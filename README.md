# Diff2DGS
Official implement for Diff2DGS: Reliable Reconstruction of Occluded Surgical Scenes via 2D Gaussian Splatting


code is coming soon

# Introduction
The ability to dynamically reconstruct surgical scenes is paramount in computer-assisted surgery. Although existing methods have achieved relatively fast reconstruction, the reconstruction result for occlusion parts in surgical scenes is not ideal. We propose Diff2DGS a novel two-stage framework for addressing the challenges of 3D reconstruction in occluded surgical scenes. The first stage leverages a diffusion-based video inpainting module, enhanced with temporal priors, to restore tissues occluded by surgical instruments with high spatiotemporal consistency. The second stage adapts 2D Gaussian Splatting (2DGS) to surgical scenarios by incorporating a Learnable Deformation Model (LDM), which explicitly models dynamic tissue deformation and anatomical geometry. Experimental results demonstrate that Diff2DGS outperforms state-of-the-art methods in both reconstruction accuracy and efficiency, particularly in
 highly occluded regions. Quantitatively, Diff2DGS achieves the state-ofthe-art PSNR and SSIM on both StereoMIS and EndoNeRF datasets.


Here's a demo video of our result:

![Sample Video](https://github.com/styufo/Diff2DGS/blob/main/assets/demo.mp4)

 # Architecture
![GitHub Logo](https://github.com/styufo/Diff2DGS/blob/main/arti.png)
Diff2DGS consists of Surgical Instrument Inpainting, Point Cloud Initialization, Deformation Modeling, and 2D Gaussian Splatting. First, we use a pre-trained surgical instrument inpainting model to generate the occlusion part of the surgical instrument. This process tracks the surgical instrument across frames and generates high-quality tissue images with spatiotemporal consistency using stable diffusion and temporal attention. Next, the images without occlusion are combined with depth information to initialize the Gaussian point cloud, and tissue deformation is simulated by a Learnable Deformation Model. Subsequently, a 2D Gaussian splatting model is applied to generate a color image and depth map from the perspective of the given camera. Finally, the optimization loss is computed against the ground truth to refine the framework further.

# Getting Started
## Setup the Environment
First, you need to create a corresponding conda environment：
```
conda create -n Diff2dgs python=3.12
conda activate Diff2dgs
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
git clone https://github.com/styufo/Diff2DGS.git
cd Diff2DGS
pip install -r requirements.txt
```
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
  The full folder size of stable-diffusion-v1-5 is quite large(>30 GB), you can download the necessary folders and files: **feature_extractor, model_index.json, safety_checker, scheduler, text_encoder, and tokenizer**
* PCM_Weights: [Hugging Face](https://huggingface.co/wangfuyun/PCM_Weights)
* propainter: [Github link](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0)
* sd-vae-ft-mse: [Hugging Face](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)

## Data Preparation
Plaese download the EndoNeRF and StereoMIS dataset from these two links：[EndoNeRF](https://github.com/med-air/EndoNeRF?tab=readme-ov-file), [StrereoMIS](https://zenodo.org/records/7727692)


To use the StereoMIS dataset, please follow this [github repo](https://github.com/aimi-lab/robust-pose-estimator) to preprocess the dataset. After that, run the script stereomis2endonerf.py provided in [Deform3DGS](https://github.com/jinlab-imvr/Deform3DGS/blob/main/stereomis2endonerf.py) to extract clips from the StereoMIS dataset and organize the depth, masks, images, intrinsic and extrinsic parameters in the same format as EndoNeRF.

And then，set the data structure is as follows:
```
data
| - endonerf_full_datasets
|   | - cutting_tissues_twice
|   |   | -  depth/
|   |   | -  images/
|   |   | -  masks/
|   |   | -  pose_bounds.npy 
|   | - pushing_soft_tissues
| - StereoMIS
|   | - stereo_seq_1
|   | - stereo_seq_2
```
