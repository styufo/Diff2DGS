import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import PIL.Image
from einops import rearrange, repeat
from dataclasses import dataclass
import copy
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
    BaseOutput
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)

from libs.unet_2d_condition import UNet2DConditionModel
from libs.brushnet_CA import BrushNetModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def validate_prompt_and_embeddings(prompt, prompt_embeds, negative_prompt, negative_prompt_embeds):
    if prompt is not None and prompt_embeds is not None:
        raise ValueError("Cannot forward both `prompt` and `prompt_embeds`. Provide only one.")
    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`. Provide only one.")
    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError("`prompt_embeds` and `negative_prompt_embeds` must have the same shape.")

def validate_images_and_masks(images, masks):
    for image in images:
        if not isinstance(image, (PIL.Image.Image, torch.Tensor, np.ndarray, list)):
            raise TypeError(f"Invalid image type: {type(image)}")
    for mask in masks:
        if not isinstance(mask, (PIL.Image.Image, torch.Tensor, np.ndarray, list)):
            raise TypeError(f"Invalid mask type: {type(mask)}")

def prepare_images(images, processor, width, height, batch_size, num_images_per_prompt, device, dtype):
    processed_images = []
    for image in images:
        image = processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        repeat_by = batch_size if image.shape[0] == 1 else num_images_per_prompt
        image = image.repeat_interleave(repeat_by, dim=0).to(device=device, dtype=dtype)
        processed_images.append(image)
    return processed_images

def prepare_masks(masks, images, device, dtype):
    processed_masks = []
    for mask in masks:
        mask = (mask.sum(1)[:, None, :, :] < 0).to(dtype)
        processed_masks.append(mask)
    return processed_masks

@dataclass
class DiffInpaintingPipelineOutput(BaseOutput):
    frames: Union[torch.Tensor, np.ndarray]
    latents: Union[torch.Tensor, np.ndarray]

class StableDiffusionDiffinpaintingPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for video inpainting using Video Diffusion Model with BrushNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        brushnet ([`BrushNetModel`]`):
            Provides additional conditioning to the `unet` during the denoising process.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        brushnet: BrushNetModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            brushnet=brushnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def check_inputs(self, prompt, images, masks, callback_steps, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None, **kwargs):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(f"`callback_steps` must be a positive integer, got {callback_steps}.")
        validate_prompt_and_embeddings(prompt, prompt_embeds, negative_prompt, negative_prompt_embeds)
        validate_images_and_masks(images, masks)

    def prepare_image(self, images, width, height, batch_size, num_images_per_prompt, device, dtype):
        return prepare_images(images, self.image_processor, width, height, batch_size, num_images_per_prompt, device, dtype)

    def prepare_latents(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor, 
            width // self.vae_scale_factor
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            noise = rearrange(randn_tensor(shape, generator=generator, device=device, dtype=dtype), "b c t h w -> (b t) c h w")
        else:
            noise = latents.to(device)

        latents = noise * self.scheduler.init_noise_sigma
        return latents, noise

    @torch.no_grad()
    def __call__(self, num_frames=24, prompt=None, images=None, masks=None, height=None, width=None, num_inference_steps=50, **kwargs):
        self.check_inputs(prompt, images, masks, kwargs.get("callback_steps"), kwargs.get("negative_prompt"), kwargs.get("prompt_embeds"), kwargs.get("negative_prompt_embeds"))

        images = self.prepare_image(images, width, height, kwargs.get("batch_size"), kwargs.get("num_images_per_prompt"), self._execution_device, self.brushnet.dtype)
        masks = prepare_masks(masks, images, self._execution_device, self.brushnet.dtype)


        return DiffinpaintingPipelineOutput(frames=video, latents=latents)
