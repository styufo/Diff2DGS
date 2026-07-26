import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from diff2dgs.pipeline import (
    DatasetLayout,
    StageState,
    create_dataset_workspace,
    natural_key,
    normalize_masks,
)


def write_image(path: Path, value: int = 0) -> None:
    Image.fromarray(np.full((4, 6), value, dtype=np.uint8)).save(path)


class PipelineTests(unittest.TestCase):
    def make_dataset(self, root: Path, frame_count: int = 2) -> None:
        for directory in ("images", "masks", "depth"):
            (root / directory).mkdir(parents=True)
            for index in range(frame_count):
                write_image(root / directory / f"frame-{index}.png", index)
        np.save(root / "poses_bounds.npy", np.zeros((frame_count, 17)))

    def test_natural_key_orders_numeric_filenames(self) -> None:
        names = [Path("frame-10.png"), Path("frame-2.png"), Path("frame-1.png")]
        self.assertEqual(
            [path.name for path in sorted(names, key=natural_key)],
            ["frame-1.png", "frame-2.png", "frame-10.png"],
        )

    def test_dataset_layout_validates_pose_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            layout = DatasetLayout.inspect(root)
            self.assertEqual(len(layout.images), 2)
            np.save(root / "poses_bounds.npy", np.zeros((1, 17)))
            with self.assertRaisesRegex(ValueError, "pose count"):
                DatasetLayout.inspect(root)

    def test_mask_polarity_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "mask.png"
            Image.fromarray(np.asarray([[0, 255]], dtype=np.uint8)).save(source)
            white = normalize_masks([source], root / "white", "white")[0]
            black = normalize_masks([source], root / "black", "black")[0]
            np.testing.assert_array_equal(np.asarray(Image.open(white)), [[0, 255]])
            np.testing.assert_array_equal(np.asarray(Image.open(black)), [[255, 0]])

    def test_workspace_does_not_link_generated_point_cloud(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            workspace = Path(temporary) / "workspace"
            root.mkdir()
            self.make_dataset(root)
            (root / "points3d.ply").write_text("source", encoding="utf-8")
            layout = DatasetLayout.inspect(root)
            create_dataset_workspace(layout, workspace)
            self.assertFalse((workspace / "points3d.ply").exists())
            self.assertTrue((workspace / "depth").is_symlink())

    def test_state_can_invalidate_downstream_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = StageState(Path(temporary))
            for stage in ("prepare", "inpaint", "extract", "train"):
                state.mark_completed(stage)
            state.clear(("extract", "train"))
            self.assertEqual(state.data["completed"], ["prepare", "inpaint"])


if __name__ == "__main__":
    unittest.main()
