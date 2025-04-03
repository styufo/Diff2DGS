import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import PIL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    BaseOutput,
    logging,
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class PipelineConfig:
    """Configuration parameters for the Pipeline"""
    vae_scale_factor: int = 8
    guidance_scale: float = 7.5
    clip_skip: Optional[int] = None
    cross_attention_kwargs: Optional[Dict] = None
    num_timesteps: int = 1000

    # Video related configuration
    num_frames_per_clip: int = 8
    overlap: int = 4
    num_inference_steps: int = 50

    # Model related configuration
    height: int = 512
    width: int = 512
    dtype: torch.dtype = torch.float32

@dataclass
class DiffInpaintingPipelineOutput(BaseOutput):
    """Pipeline output class"""
    frames: Union[torch.Tensor, np.ndarray]
    latents: Optional[Union[torch.Tensor, np.ndarray]] = None

class PipelineUtils:
    """Utility class providing common methods"""

    @staticmethod
    def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None):
        """Retrieve timesteps for the diffusion process"""
        if timesteps is None:
            timesteps = scheduler.timesteps
        elif isinstance(timesteps, int):
            timesteps = [timesteps]

        timesteps = torch.tensor(timesteps, dtype=torch.long, device=device)
        return timesteps

    @staticmethod
    def get_frames_context_swap(total_frames: int, overlap: int, num_frames_per_clip: int) -> List[Tuple[int, int]]:
        """Calculate frame context swapping intervals"""
        if total_frames <= num_frames_per_clip:
            return [(0, total_frames)]

        chunks = []
        start = 0
        while start < total_frames:
            end = min(start + num_frames_per_clip, total_frames)
            chunks.append((start, end))
            start = end - overlap

        return chunks

class InputValidator:
    """Input validation class"""

    @staticmethod
    def validate_inputs(
        images: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]],
        masks: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]],
        height: Optional[int],
        width: Optional[int],
        prompt: Union[str, List[str]],
        prompt_embeds: Optional[torch.Tensor] = None,
    ):
        """Validate input parameters and format"""
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot provide both `prompt` and `prompt_embeds`.")

        if prompt is None and prompt_embeds is None:
            raise ValueError("Must provide either `prompt` or `prompt_embeds`.")

        if isinstance(images, list) and isinstance(masks, list):
            if len(images) != len(masks):
                raise ValueError("Number of images and masks must match.")

        if height is not None and width is not None:
            if isinstance(images, list):
                for image in images:
                    if isinstance(image, PIL.Image.Image):
                        image = image.resize((width, height), PIL_INTERPOLATION["lanczos"])
            elif isinstance(images, PIL.Image.Image):
                images = images.resize((width, height), PIL_INTERPOLATION["lanczos"])

        return images, masks

class DataProcessor:
    """Data processing class"""

    def __init__(self, vae: AutoencoderKL, image_processor: CLIPImageProcessor):
        self.vae = vae
        self.image_processor = image_processor

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prepare latent vectors for the diffusion process"""
        shape = (batch_size, num_channels, frames, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        return latents

    def prepare_image(
        self,
        image: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]],
        width: int,
        height: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare and format input images"""
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"]) for i in image]
                image = [np.array(i)[None, :] for i in image]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            else:
                raise ValueError(f"Invalid image type: {type(image)}")

        image = image.to(device=device, dtype=dtype)
        return image

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to images"""
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents

        latents = latents.permute(0, 2, 1, 3, 4).reshape(
            (latents.shape[0] * video_length, latents.shape[1], latents.shape[3], latents.shape[4])
        )

        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        image = image.reshape(
            (latents.shape[0] // video_length, video_length, image.shape[1], image.shape[2], image.shape[3])
        ).permute(0, 2, 1, 3, 4)

        return image

class Encoder:
    """Encoder class for text and image encoding"""

    def __init__(
        self,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode input prompts to embeddings"""
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = self.text_encoder(text_input_ids.to(device))

            if clip_skip is not None:
                prompt_embeds = prompt_embeds[clip_skip - 1]
            else:
                prompt_embeds = prompt_embeds.last_hidden_state

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = negative_prompt if negative_prompt is not None else [""] * batch_size
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device))

            if clip_skip is not None:
                negative_prompt_embeds = negative_prompt_embeds[clip_skip - 1]
            else:
                negative_prompt_embeds = negative_prompt_embeds.last_hidden_state

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return prompt_embeds

class DiffusionModel:
    """Diffusion model class handling core diffusion logic"""

    def __init__(
        self,
        unet: UNet2DConditionModel,
        brushnet: Any,  # BrushNet type
        scheduler: KarrasDiffusionSchedulers,
    ):
        self.unet = unet
        self.brushnet = brushnet
        self.scheduler = scheduler

    def denoise_step(
        self,
        latents: torch.Tensor,
        timestep: int,
        prompt_embeds: torch.Tensor,
        guidance_scale: float,
        cross_attention_kwargs: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Execute a single denoising step"""
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        # Predict noise residual
        noise_pred = self.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        # Perform guidance
        if guidance_scale > 1:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Apply BrushNet guidance
        if self.brushnet is not None:
            noise_pred = self.brushnet(noise_pred)

        # Calculate previous latents
        latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

        return latents

class StableDiffusionDiffInpaintingPipeline(DiffusionPipeline):
    """Main Pipeline class for diffusion-based inpainting"""

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        brushnet: Any,  # BrushNet type
        scheduler: KarrasDiffusionSchedulers,
        image_processor: CLIPImageProcessor,
    ):
        super().__init__()

        self.config = PipelineConfig()
        self.data_processor = DataProcessor(vae, image_processor)
        self.encoder = Encoder(text_encoder, tokenizer)
        self.diffusion_model = DiffusionModel(unet, brushnet, scheduler)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            brushnet=brushnet,
            scheduler=scheduler,
            image_processor=image_processor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        images: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]],
        masks: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
    ) -> Union[DiffInpaintingPipelineOutput, Tuple]:
        """Main entry point for pipeline execution"""

        # Validate and prepare parameters
        images, masks = InputValidator.validate_inputs(
            images, masks, height, width, prompt, prompt_embeds
        )

        device = self._execution_device
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare prompt embeddings
        prompt_embeds = self.encoder.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )

        # Prepare images and masks
        images = self.data_processor.prepare_image(
            images,
            width,
            height,
            1,
            device,
            prompt_embeds.dtype,
        )

        masks = self.data_processor.prepare_image(
            masks,
            width,
            height,
            1,
            device,
            prompt_embeds.dtype,
        )

        # Prepare latents
        latents = self.data_processor.prepare_latents(
            1,
            self.unet.config.in_channels,
            images.shape[2],
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Diffusion process
        for i, t in enumerate(self.progress_bar(timesteps)):
            latents = self.diffusion_model.denoise_step(
                latents=latents,
                timestep=t,
                prompt_embeds=prompt_embeds,
                guidance_scale=guidance_scale,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # Decode results
        image = self.data_processor.decode_latents(latents)

        # Post-processing
        if output_type == "pil":
            image = self.image_processor.numpy_to_pil(image)

        if not return_dict:
            return (image, latents)

        return DiffInpaintingPipelineOutput(frames=image, latents=latents)
