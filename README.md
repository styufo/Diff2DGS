# Diff2DGS
Official implement for Diff2DGS: Reliable Reconstruction of Occluded Surgical Scenes via 2D Gaussian Splatting


code is coming soon

# Introduction
Theability to dynamically reconstruct surgical scenes is paramount in computer-assisted surgery. Although existing methods have achieved relatively fast reconstruction, the reconstruction result for occlusion parts in surgical scenes is not ideal. We propose Diff2DGS a novel two-stage framework for addressing the challenges of 3D reconstruction in occluded surgical scenes. The first stage leverages a diffusion-based video inpainting module, enhanced with temporal priors, to restore tissues occluded by surgical instruments with high spatiotemporal consistency. The second stage adapts 2D Gaussian Splatting (2DGS) to surgical scenarios by incorporating a Learnable Deformation Model (LDM), which explicitly models dynamic tissue deformation and anatomical geometry. Experimental results demonstrate that Diff2DGS outperforms state-of-the-art methods in both reconstruction accuracy and efficiency, particularly in
 highly occluded regions. Quantitatively, Diff2DGS achieves the state-ofthe-art PSNR and SSIM on both StereoMIS and EndoNeRF dataset.
