"""Staged, resumable execution of the Diff2DGS pipeline."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path) -> list[object]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", path.name)]


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    files = sorted(
        (path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_key,
    )
    if not files:
        raise ValueError(f"No supported images found in {directory}")
    return files


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    images: tuple[Path, ...]
    masks: tuple[Path, ...]
    depths: tuple[Path, ...]

    @classmethod
    def inspect(cls, root: Path) -> "DatasetLayout":
        root = root.resolve()
        pose_file = root / "poses_bounds.npy"
        if not pose_file.is_file():
            raise FileNotFoundError(f"Missing camera poses: {pose_file}")

        images = tuple(image_files(root / "images"))
        masks = tuple(image_files(root / "masks"))
        depths = tuple(image_files(root / "depth"))
        counts = {"images": len(images), "masks": len(masks), "depth": len(depths)}
        if len(set(counts.values())) != 1:
            raise ValueError(f"Frame counts do not match: {counts}")

        pose_count = np.load(pose_file, mmap_mode="r").shape[0]
        if pose_count != len(images):
            raise ValueError(
                f"Camera pose count ({pose_count}) does not match frame count ({len(images)})"
            )

        expected_size = None
        for image_path, mask_path, depth_path in zip(images, masks, depths):
            with Image.open(image_path) as image, Image.open(mask_path) as mask, Image.open(
                depth_path
            ) as depth:
                expected_size = expected_size or image.size
                sizes = {image.size, mask.size, depth.size, expected_size}
            if len(sizes) != 1:
                raise ValueError(
                    "Frame dimensions do not match for "
                    f"{image_path.name}, {mask_path.name}, and {depth_path.name}: {sizes}"
                )
        return cls(root=root, images=images, masks=masks, depths=depths)


class StageState:
    def __init__(self, workspace: Path):
        self.path = workspace / "pipeline_state.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"completed": [], "commands": []}

    def completed(self, stage: str) -> bool:
        return stage in self.data["completed"]

    def add_command(self, stage: str, command: Sequence[str]) -> None:
        self.data["commands"].append({"stage": stage, "command": list(command)})
        self.save()

    def mark_completed(self, stage: str) -> None:
        if stage not in self.data["completed"]:
            self.data["completed"].append(stage)
        self.save()

    def clear(self, stages: Iterable[str]) -> None:
        invalidated = set(stages)
        completed = [stage for stage in self.data["completed"] if stage not in invalidated]
        if completed != self.data["completed"]:
            self.data["completed"] = completed
            self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")


def require_program(program: str) -> str:
    resolved = shutil.which(program)
    if resolved is None:
        raise RuntimeError(f"Required executable is not available: {program}")
    return resolved


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stage: str,
    state: StageState,
) -> None:
    printable = " ".join(str(value) for value in command)
    print(f"\n[{stage}] {printable}", flush=True)
    state.add_command(stage, command)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _ffmpeg_concat_manifest(files: Iterable[Path], path: Path, fps: float) -> None:
    duration = 1.0 / fps
    lines: list[str] = []
    for file_path in files:
        escaped = str(file_path.resolve()).replace("'", "'\\''")
        lines.extend([f"file '{escaped}'", f"duration {duration:.12f}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def encode_video(
    files: Sequence[Path], output: Path, fps: float, manifest: Path, *, grayscale: bool
) -> None:
    _ffmpeg_concat_manifest(files, manifest, fps)
    pixel_format = "gray" if grayscale else "yuv444p"
    subprocess.run(
        [
            require_program("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-frames:v",
            str(len(files)),
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "0",
            "-pix_fmt",
            pixel_format,
            str(output),
        ],
        check=True,
    )


def normalize_masks(
    masks: Sequence[Path], output_dir: Path, foreground: str
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []
    for index, source in enumerate(masks):
        with Image.open(source) as image:
            binary = image.convert("L").point(
                (lambda value: 255 if value > 127 else 0), mode="1"
            ).convert("L")
            if foreground == "black":
                binary = binary.point(lambda value: 255 - value)
            destination = output_dir / f"{index:06d}.png"
            binary.save(destination)
            output.append(destination)
    return output


def create_dataset_workspace(layout: DatasetLayout, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    generated_point_cloud = destination / "points3d.ply"
    if generated_point_cloud.is_symlink():
        generated_point_cloud.unlink()
    for source in layout.root.iterdir():
        if source.name in {"images", "points3d.ply"}:
            continue
        target = destination / source.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    (destination / "images").mkdir(exist_ok=True)


def extract_frames(video: Path, names: Sequence[str], output_dir: Path) -> None:
    temporary = output_dir.parent / "extracted_frames"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            require_program("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vsync",
            "0",
            "-frames:v",
            str(len(names)),
            str(temporary / "%06d.png"),
        ],
        check=True,
    )
    frames = sorted(temporary.glob("*.png"), key=natural_key)
    if len(frames) != len(names):
        raise RuntimeError(
            f"Inpainted video yielded {len(frames)} frames; expected {len(names)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in output_dir.iterdir():
        if old_frame.is_file() or old_frame.is_symlink():
            old_frame.unlink()
    for frame, name in zip(frames, names):
        destination = output_dir / name
        with Image.open(frame) as image:
            image.convert("RGB").save(destination)
    shutil.rmtree(temporary)


@dataclass
class PipelineOptions:
    repo: Path
    data: Path
    workspace: Path
    weights: Path
    config: Path
    fps: float
    dataset_type: str
    mask_foreground: str
    gpu: str
    skip_inpainting: bool
    force: frozenset[str]
    export_ply: bool
    depth_strategy: str
    depth_weight: float
    depth_weight_init: float
    depth_weight_alpha: float
    depth_weight_beta: float
    depth_weight_min: float
    depth_weight_max: float


class Diff2DGSPipeline:
    stages = ("prepare", "inpaint", "extract", "train", "render", "evaluate")

    def __init__(self, options: PipelineOptions):
        self.options = options
        self.layout = DatasetLayout.inspect(options.data)
        self.workspace = options.workspace.resolve()
        if self.workspace == self.layout.root or self.layout.root in self.workspace.parents:
            raise ValueError("--workspace must be outside the source dataset directory")
        self.workspace.mkdir(parents=True, exist_ok=True)
        manifest_path = self.workspace / "dataset_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if Path(manifest["source"]).resolve() != self.layout.root:
                raise ValueError(
                    "This workspace belongs to a different source dataset; use a new workspace"
                )
        self.state = StageState(self.workspace)
        self.media = self.workspace / "media"
        self.dataset = self.workspace / "dataset"
        self.model = self.workspace / "model"
        self.reconstruction = options.repo / "reconstruction"
        self.env = os.environ.copy()
        self.env["CUDA_VISIBLE_DEVICES"] = options.gpu
        self.env["PYTHONPATH"] = os.pathsep.join(
            [str(options.repo), str(options.repo / "third_party"), self.env.get("PYTHONPATH", "")]
        )

    def should_run(self, stage: str) -> bool:
        if stage in self.options.force:
            stage_index = self.stages.index(stage)
            self.state.clear(self.stages[stage_index:])
        return not self.state.completed(stage)

    def prepare(self) -> None:
        if not self.should_run("prepare"):
            return
        self.media.mkdir(parents=True, exist_ok=True)
        create_dataset_workspace(self.layout, self.dataset)
        if self.options.skip_inpainting:
            for source in self.layout.images:
                target = self.dataset / "images" / source.name
                if not target.exists():
                    target.symlink_to(source.resolve())
        else:
            require_program("ffmpeg")
            normalized = normalize_masks(
                self.layout.masks,
                self.media / "normalized_masks",
                self.options.mask_foreground,
            )
            encode_video(
                self.layout.images,
                self.media / "input.mp4",
                self.options.fps,
                self.media / "images.ffconcat",
                grayscale=False,
            )
            encode_video(
                normalized,
                self.media / "mask.mp4",
                self.options.fps,
                self.media / "masks.ffconcat",
                grayscale=True,
            )
        metadata = {
            "source": str(self.layout.root),
            "frames": len(self.layout.images),
            "fps": self.options.fps,
            "image_names": [path.name for path in self.layout.images],
            "mask_foreground": self.options.mask_foreground,
        }
        (self.workspace / "dataset_manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        self.state.mark_completed("prepare")

    def inpaint(self) -> None:
        if self.options.skip_inpainting or not self.should_run("inpaint"):
            return
        duration = len(self.layout.images) / self.options.fps
        command = [
            sys.executable,
            "-m",
            "inpainting.run",
            "--input-video",
            str(self.media / "input.mp4"),
            "--input-mask",
            str(self.media / "mask.mp4"),
            "--output",
            str(self.media / "inpainted.mp4"),
            "--weights",
            str(self.options.weights),
            "--video-length",
            str(duration),
        ]
        run_command(
            command,
            cwd=self.options.repo,
            env=self.env,
            stage="inpaint",
            state=self.state,
        )
        self.state.mark_completed("inpaint")

    def extract(self) -> None:
        if self.options.skip_inpainting or not self.should_run("extract"):
            return
        extract_frames(
            self.media / "inpainted.mp4",
            [path.name for path in self.layout.images],
            self.dataset / "images",
        )
        self.state.mark_completed("extract")

    def train(self) -> None:
        if not self.should_run("train"):
            return
        command = [
            sys.executable,
            "train.py",
            "-s",
            str(self.dataset),
            "-m",
            str(self.model),
            "--expname",
            self.workspace.name,
            "--configs",
            str(self.options.config),
            "--dataset_type",
            self.options.dataset_type,
            "--depth_weight_strategy",
            self.options.depth_strategy,
            "--depth_weight",
            str(self.options.depth_weight),
            "--depth_weight_init",
            str(self.options.depth_weight_init),
            "--depth_weight_alpha",
            str(self.options.depth_weight_alpha),
            "--depth_weight_beta",
            str(self.options.depth_weight_beta),
            "--depth_weight_min",
            str(self.options.depth_weight_min),
            "--depth_weight_max",
            str(self.options.depth_weight_max),
        ]
        run_command(
            command,
            cwd=self.reconstruction,
            env=self.env,
            stage="train",
            state=self.state,
        )
        self.state.mark_completed("train")

    def render(self) -> None:
        if not self.should_run("render"):
            return
        command = [
            sys.executable,
            "render.py",
            "-m",
            str(self.model),
            "--skip_train",
            "--configs",
            str(self.options.config),
        ]
        if self.options.export_ply:
            command.append("--reconstruct_video")
        else:
            command.append("--skip_video")
        run_command(
            command,
            cwd=self.reconstruction,
            env=self.env,
            stage="render",
            state=self.state,
        )
        self.state.mark_completed("render")

    def evaluate(self) -> None:
        if not self.should_run("evaluate"):
            return
        command = [sys.executable, "metrics.py", "-m", str(self.model), "-p", "test"]
        run_command(
            command,
            cwd=self.reconstruction,
            env=self.env,
            stage="evaluate",
            state=self.state,
        )
        self.state.mark_completed("evaluate")

    def run(self, from_stage: str, to_stage: str) -> None:
        start = self.stages.index(from_stage)
        end = self.stages.index(to_stage)
        if start > end:
            raise ValueError("--from-stage must not come after --to-stage")
        for stage in self.stages[start : end + 1]:
            if self.options.skip_inpainting and stage in {"inpaint", "extract"}:
                continue
            getattr(self, stage)()
        print(f"\nDiff2DGS pipeline completed through '{to_stage}': {self.workspace}")
