import os
import time
import argparse
import torch
from diffinpaint.diffinpaint import diffinpaint
from propainter.inference import Propainter, get_device


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DiffInpaint with ProPainter.")
    parser.add_argument('--input_video', type=str, default="examples/example0/video.mp4", help='Path to the input video')
    parser.add_argument('--input_mask', type=str, default="examples/example0/mask.mp4", help='Path to the input mask')
    parser.add_argument('--video_length', type=int, default=10, help='The maximum length of output video')
    parser.add_argument('--mask_dilation_iter', type=int, default=8, help='Adjust it to change the degree of mask expansion')
    parser.add_argument('--max_img_size', type=int, default=960, help='The maximum length of output width and height')
    parser.add_argument('--save_path', type=str, default="results", help='Path to the output')
    parser.add_argument('--ref_stride', type=int, default=10, help='ProPainter parameter')
    parser.add_argument('--neighbor_length', type=int, default=10, help='ProPainter parameter')
    parser.add_argument('--subvideo_length', type=int, default=50, help='ProPainter parameter')
    parser.add_argument('--base_model_path', type=str, default="weights/stable-diffusion-v1-5", help='Path to SD1.5 base model')
    parser.add_argument('--vae_path', type=str, default="weights/sd-vae-ft-mse", help='Path to VAE')
    parser.add_argument('--diffinpaint_path', type=str, default="weights/diffinpaint", help='Path to DiffInpaint')
    parser.add_argument('--propainter_model_dir', type=str, default="weights/propainter", help='Path to ProPainter model')
    return parser.parse_args()


def initialize_models(args):
    """Initialize models for DiffInpaint and ProPainter."""
    device = get_device()
    video_inpainting_sd = diffinpaint(
        device, args.base_model_path, args.vae_path, args.diffinpaint_path, ckpt="2-Step"
    )
    propainter = Propainter(args.propainter_model_dir, device=device)
    return video_inpainting_sd, propainter, device


def create_output_paths(save_path):
    """Create output directories and return paths for output files."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    priori_path = os.path.join(save_path, "priori.mp4")
    output_path = os.path.join(save_path, "removetools_result.mp4")
    return priori_path, output_path


def run_propaint(propainter, args, priori_path):
    """Run ProPainter to generate the priori video."""
    propainter.forward(
        args.input_video, args.input_mask, priori_path,
        video_length=args.video_length, ref_stride=args.ref_stride,
        neighbor_length=args.neighbor_length, subvideo_length=args.subvideo_length,
        mask_dilation=args.mask_dilation_iter
    )


def run_diffinpaint(video_inpainting_sd, args, priori_path, output_path):
    """Run DiffInpaint to generate the final output video."""
    video_inpainting_sd.forward(
        args.input_video, args.input_mask, priori_path, output_path,
        max_img_size=args.max_img_size, video_length=args.video_length,
        mask_dilation_iter=args.mask_dilation_iter, guidance_scale=None
    )


def main():
    args = parse_arguments()
    video_inpainting_sd, propainter, _ = initialize_models(args)
    priori_path, output_path = create_output_paths(args.save_path)

    start_time = time.time()

    run_propaint(propainter, args, priori_path)
    run_diffinpaint(video_inpainting_sd, args, priori_path, output_path)

    end_time = time.time()
    print(f"DiffInpaint inference time: {end_time - start_time:.4f} s")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
