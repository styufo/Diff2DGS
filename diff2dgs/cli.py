"""Public command-line interface for Diff2DGS."""

import argparse
from pathlib import Path

from .pipeline import Diff2DGSPipeline, PipelineOptions


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = Diff2DGSPipeline.stages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diff2dgs", description="End-to-end surgical video inpainting and deformable 2DGS reconstruction"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run the staged end-to-end pipeline")
    run.add_argument("--data", type=Path, required=True, help="EndoNeRF-format scene directory")
    run.add_argument("--workspace", type=Path, required=True, help="Isolated output workspace")
    run.add_argument("--weights", type=Path, default=REPO_ROOT / "weights")
    run.add_argument(
        "--config", type=Path, default=REPO_ROOT / "reconstruction/arguments/endonerf/default.py"
    )
    run.add_argument("--fps", type=float, default=30.0)
    run.add_argument(
        "--dataset-type",
        choices=("auto", "endonerf", "stereomis"),
        default="auto",
        help="Camera convention; auto detects common EndoNeRF and StereoMIS layouts",
    )
    run.add_argument("--mask-foreground", choices=("white", "black"), default="white")
    run.add_argument("--gpu", default="0", help="CUDA device index exposed to each stage")
    run.add_argument("--skip-inpainting", action="store_true", help="Reconstruct the input images directly")
    run.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    run.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    run.add_argument("--force", action="append", choices=STAGES, default=[])
    run.add_argument("--export-ply", action="store_true")
    run.add_argument(
        "--depth-strategy",
        choices=("fixed", "adaptive_ratio", "adaptive_ratio_ema", "adaptive_ratio_warmup"),
        default="adaptive_ratio",
    )
    run.add_argument("--depth-weight", type=float, default=1.0)
    run.add_argument("--depth-weight-init", type=float, default=10.0)
    run.add_argument("--depth-weight-alpha", type=float, default=0.8)
    run.add_argument("--depth-weight-beta", type=float, default=0.25)
    run.add_argument("--depth-weight-min", type=float, default=1.0)
    run.add_argument("--depth-weight-max", type=float, default=10.0)

    visualize = subparsers.add_parser(
        "visualize", help="Play an exported per-frame PLY sequence"
    )
    visualize.add_argument("--input", type=Path, required=True)
    visualize.add_argument("--fps", type=float, default=10.0)
    visualize.add_argument("--output", type=Path)
    visualize.add_argument("--width", type=int, default=1280)
    visualize.add_argument("--height", type=int, default=720)
    visualize.add_argument("--point-size", type=float, default=2.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "visualize":
        from .visualize import play_sequence

        play_sequence(
            args.input,
            fps=args.fps,
            output=args.output,
            width=args.width,
            height=args.height,
            point_size=args.point_size,
        )
        return
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.depth_weight_min > args.depth_weight_max:
        parser.error("--depth-weight-min must not exceed --depth-weight-max")
    options = PipelineOptions(
        repo=REPO_ROOT,
        data=args.data,
        workspace=args.workspace,
        weights=args.weights.resolve(),
        config=args.config.resolve(),
        fps=args.fps,
        dataset_type=args.dataset_type,
        mask_foreground=args.mask_foreground,
        gpu=args.gpu,
        skip_inpainting=args.skip_inpainting,
        force=frozenset(args.force),
        export_ply=args.export_ply,
        depth_strategy=args.depth_strategy,
        depth_weight=args.depth_weight,
        depth_weight_init=args.depth_weight_init,
        depth_weight_alpha=args.depth_weight_alpha,
        depth_weight_beta=args.depth_weight_beta,
        depth_weight_min=args.depth_weight_min,
        depth_weight_max=args.depth_weight_max,
    )
    Diff2DGSPipeline(options).run(args.from_stage, args.to_stage)


if __name__ == "__main__":
    main()
