"""Interactive playback for exported point-cloud sequences."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def natural_key(path: Path) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def play_sequence(
    input_dir: Path,
    *,
    fps: float = 10.0,
    output: Path | None = None,
    width: int = 1280,
    height: int = 720,
    point_size: float = 2.0,
) -> None:
    if fps <= 0:
        raise ValueError("--fps must be positive")
    files = sorted(input_dir.expanduser().resolve().glob("*.ply"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No PLY files found in {input_dir}")

    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(
        window_name="Diff2DGS point-cloud sequence", width=width, height=height
    ):
        raise RuntimeError("Could not create an Open3D window; an OpenGL display is required")

    writer = None
    point_cloud = o3d.io.read_point_cloud(str(files[0]))
    if not point_cloud.has_points():
        visualizer.destroy_window()
        raise ValueError(f"Point cloud is empty: {files[0]}")
    if not point_cloud.has_colors():
        point_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    visualizer.add_geometry(point_cloud)

    render_options = visualizer.get_render_option()
    render_options.background_color = np.asarray([1.0, 1.0, 1.0])
    render_options.point_size = point_size
    render_options.show_coordinate_frame = True

    view = visualizer.get_view_control()
    view.set_front([0.0, 0.0, -1.0])
    view.set_up([0.0, 1.0, 0.0])
    view.set_lookat(point_cloud.get_axis_aligned_bounding_box().get_center())
    view.set_zoom(0.7)

    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            visualizer.destroy_window()
            raise RuntimeError(f"Could not open video writer: {output}")

    frame_period = 1.0 / fps
    try:
        for path in files:
            started = time.monotonic()
            frame = o3d.io.read_point_cloud(str(path))
            if not frame.has_points():
                continue
            if not frame.has_colors():
                frame.paint_uniform_color([0.5, 0.5, 0.5])
            point_cloud.points = frame.points
            point_cloud.colors = frame.colors
            point_cloud.normals = frame.normals
            visualizer.update_geometry(point_cloud)
            if not visualizer.poll_events():
                break
            visualizer.update_renderer()

            if writer is not None:
                rgb = np.asarray(visualizer.capture_screen_float_buffer(False))
                writer.write(cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            remaining = frame_period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if writer is not None:
            writer.release()
        visualizer.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--point-size", type=float, default=2.0)
    args = parser.parse_args()
    play_sequence(
        args.input,
        fps=args.fps,
        output=args.output,
        width=args.width,
        height=args.height,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
