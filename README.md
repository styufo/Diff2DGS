# Diff2DGS
Official implement for Diff2DGS: Reliable Reconstruction of Occluded Surgical Scenes via 2D Gaussian Splatting


code is coming soon

# Introduction
Theability to dynamically reconstruct surgical scenes is paramount in computer-assisted surgery. Although existing methods have achieved relatively fast reconstruction, the reconstruction result for occlusion parts in surgical scenes is not ideal. We propose Diff2DGS a novel two-stage framework for addressing the challenges of 3D reconstruction in occluded surgical scenes. The first stage leverages a diffusion-based video inpainting module, enhanced with temporal priors, to restore tissues occluded by surgical instruments with high spatiotemporal consistency. The second stage adapts 2D Gaussian Splatting (2DGS) to surgical scenarios by incorporating a Learnable Deformation Model (LDM), which explicitly models dynamic tissue deformation and anatomical geometry. Experimental results demonstrate that Diff2DGS outperforms state-of-the-art methods in both reconstruction accuracy and efficiency, particularly in
 highly occluded regions. Quantitatively, Diff2DGS achieves the state-ofthe-art PSNR and SSIM on both StereoMIS and EndoNeRF dataset.

# Getting Started
## Download the pre-trained models
Place the weight under the ./weights directory, the structure of the directory will be arranged as:
weights
 
- |- diffinpaint
  - |-brushnet
  - |-unet_main
- |- stable-diffusion-v1-5
  - |-feature_extractor
  - |-...
- |- PCM_Weights
  - |-sd15  
- |- propainter
  - |-ProPainter.pth
  - |-raft-things.pth
  - |-recurrent_flow_completion.pth
- |- sd-vae-ft-mse
  - |-diffusion_pytorch_model.bin
  - |-...
- |- README.md


1. Download the weights of diffinpaint in [Google Drive Link](https://drive.google.com/drive/folders/1TZPRpgjMtV274dyqo3XBy_0PB93upHSy?usp=sharing):

2. Download the pretrained weight of based models and other components:
* stable-diffusion-v1-5：[Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)  
  The full folder size of stable-diffusion-v1-5 is quite large(>30 GB), you can download the necessary folders and files: #feature_extractor, model_index.json, safety_checker, scheduler, text_encoder, and tokenizer#
* PCM_Weights: [Hugging Face](https://huggingface.co/wangfuyun/PCM_Weights)
* propainter: [Github link](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0)
* sd-vae-ft-mse: [Hugging Face](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)
