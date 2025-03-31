import gc
import copy
import cv2
import os
import numpy as np
import torch
import torchvision
from einops import repeat
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
)
from diffusers.schedulers import TCDScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, PretrainedConfig

from libs.unet_motion_model import MotionAdapter, UNetMotionModel
from libs.brushnet_CA import BrushNetModel
from diffinpainting.pipeline_diffinpainting import StableDiffusionInpaintingPipeline


class Diffinpainting:
    def __init__(self, device, base_model_path, vae_path, diffinpaint_path, revision=None,
                 ckpt="Normal CFG 4-Step", mode="sd15", loaded=None):
        self.device = device
        self._load_models(base_model_path, vae_path, diffinpaint_path, revision)
        self._initialize_pipeline(base_model_path, ckpt, mode, loaded)

    def _load_models(self, base_model_path, vae_path, diffinpaint_path, revision):
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            base_model_path, subfolder="scheduler", prediction_type="v_prediction",
            timestep_spacing="trailing", rescale_betas_zero_snr=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer", use_fast=False)
        text_encoder_cls = self._import_model_class(base_model_path, revision)
        self.text_encoder = text_encoder_cls.from_pretrained(base_model_path, subfolder="text_encoder")
        self.brushnet = BrushNetModel.from_pretrained(diffinpaint_path, subfolder="brushnet")
        self.unet_main = UNetMotionModel.from_pretrained(diffinpaint_path, subfolder="unet_main")

    def _import_model_class(self, pretrained_model_name_or_path, revision):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
        )
        model_class = text_encoder_config.architectures[0]
        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel
            return CLIPTextModel
        elif model_class == "RobertaSeriesModelWithTransformation":
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
            return RobertaSeriesModelWithTransformation
        else:
            raise ValueError(f"{model_class} is not supported.")

    def _initialize_pipeline(self, base_model_path, ckpt, mode, loaded):
        self.pipeline = StableDiffusiondiffinpaintPipeline.from_pretrained(
            base_model_path, vae=self.vae, text_encoder=self.text_encoder,
            tokenizer=self.tokenizer, unet=self.unet_main, brushnet=self.brushnet
        ).to(self.device, torch.float16)
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.set_progress_bar_config(disable=True)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)

        self._load_pcm_weights(ckpt, mode, loaded)

    def _load_pcm_weights(self, ckpt, mode, loaded):
        checkpoints = {
            "2-Step": ["pcm_{}_smallcfg_2step_converted.safetensors", 2, 0.0],
            "4-Step": ["pcm_{}_smallcfg_4step_converted.safetensors", 4, 0.0],
            "8-Step": ["pcm_{}_smallcfg_8step_converted.safetensors", 8, 0.0],
            "16-Step": ["pcm_{}_smallcfg_16step_converted.safetensors", 16, 0.0],
            "Normal CFG 4-Step": ["pcm_{}_normalcfg_4step_converted.safetensors", 4, 7.5],
            "Normal CFG 8-Step": ["pcm_{}_normalcfg_8step_converted.safetensors", 8, 7.5],
            "Normal CFG 16-Step": ["pcm_{}_normalcfg_16step_converted.safetensors", 16, 7.5],
            "LCM-Like LoRA": ["pcm_{}_lcmlike_lora_converted.safetensors", 4, 0.0],
        }
        self.ckpt = ckpt
        PCM_ckpts = checkpoints[ckpt][0].format(mode)
        self.guidance_scale = checkpoints[ckpt][2]
        if loaded != (ckpt + mode):
            self.pipeline.load_lora_weights("weights/PCM_Weights", weight_name=PCM_ckpts, subfolder=mode)
            loaded = ckpt + mode
            self.pipeline.scheduler = LCMScheduler() if ckpt == "LCM-Like LoRA" else TCDScheduler(
                num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                beta_schedule="scaled_linear", timestep_spacing="trailing"
            )
        self.num_inference_steps = checkpoints[ckpt][1]

    def forward(self, validation_image, validation_mask, priori, output_path, **kwargs):
        frames, fps, img_size, n_clip, n_total_frames = self._read_video(validation_image, **kwargs)
        masks, masked_images = self._read_mask(validation_mask, fps, len(frames), img_size, kwargs["mask_dilation_iter"], frames)
        prioris = self._read_priori(priori, fps, n_total_frames, img_size)

        n_total_frames = min(len(frames), len(masks), len(prioris))
        masks, masked_images, frames, prioris = self._resize_inputs(masks, masked_images, frames, prioris, n_total_frames)

        latents, noise = self._prepare_latents(prioris, img_size, n_clip, kwargs["nframes"], kwargs["seed"])
        self._pre_inference(latents, masks, masked_images, frames, kwargs["nframes"], n_total_frames)

        images = self._frame_by_frame_inference(latents, masks, masked_images, kwargs["nframes"], kwargs["guidance_scale"])
        self._compose_and_save(images, masks, frames, output_path, fps, kwargs["blended"])

        return output_path
