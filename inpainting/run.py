"""Command-line entry point for the Diff2DGS inpainting stage."""

import argparse
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from inpainting.diffueraser import DiffuEraser, checkpoints  # noqa: E402
from propainter.inference import Propainter, get_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Diff2DGS video inpainting")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--input-mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default=str(REPO_ROOT / "weights"))
    parser.add_argument("--video-length", type=float, required=True)
    parser.add_argument("--mask-dilation", type=int, default=8)
    parser.add_argument("--max-image-size", type=int, default=960)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument("--neighbor-length", type=int, default=10)
    parser.add_argument("--subvideo-length", type=int, default=50)
    parser.add_argument("--checkpoint", choices=tuple(checkpoints), default="2-Step")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--keep-prior", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = output.with_name(f"{output.stem}_prior.mp4")
    weights = Path(args.weights).resolve()

    required = {
        "base model": weights / "stable-diffusion-v1-5",
        "VAE": weights / "sd-vae-ft-mse",
        "inpainting model": weights / "diffinpaint",
        "PCM": weights / "PCM_Weights",
        "ProPainter": weights / "propainter",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model weights:\n  " + "\n  ".join(missing))

    required_files = {
        "BrushNet checkpoint": required["inpainting model"] / "brushnet",
        "video UNet checkpoint": required["inpainting model"] / "unet_main",
        "PCM checkpoint": required["PCM"]
        / "sd15"
        / checkpoints[args.checkpoint][0].format("sd15"),
        "ProPainter checkpoint": required["ProPainter"] / "ProPainter.pth",
        "RAFT checkpoint": required["ProPainter"] / "raft-things.pth",
        "flow-completion checkpoint": required["ProPainter"]
        / "recurrent_flow_completion.pth",
    }
    missing_files = [
        f"{name}: {path}" for name, path in required_files.items() if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError("Missing model checkpoints:\n  " + "\n  ".join(missing_files))

    device = get_device()
    if device.type != "cuda":
        raise RuntimeError("Diff2DGS video inpainting requires a CUDA-capable GPU")
    model = DiffuEraser(
        device,
        str(required["base model"]),
        str(required["VAE"]),
        str(required["inpainting model"]),
        pcm_path=str(required["PCM"]),
        ckpt=args.checkpoint,
    )
    propainter = Propainter(str(required["ProPainter"]), device=device)

    started = time.time()
    propainter.forward(
        args.input_video,
        args.input_mask,
        str(prior),
        video_length=args.video_length,
        ref_stride=args.ref_stride,
        neighbor_length=args.neighbor_length,
        subvideo_length=args.subvideo_length,
        mask_dilation=args.mask_dilation,
    )
    model.forward(
        args.input_video,
        args.input_mask,
        str(prior),
        str(output),
        max_img_size=args.max_image_size,
        video_length=args.video_length,
        mask_dilation_iter=args.mask_dilation,
        seed=args.seed,
        guidance_scale=None,
    )
    if not args.keep_prior and prior.exists():
        prior.unlink()
    torch.cuda.empty_cache()
    print(f"Inpainting completed in {time.time() - started:.1f}s: {output}")


if __name__ == "__main__":
    main()
