import sys
import unittest
from pathlib import Path

import numpy as np
import torch


RECONSTRUCTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECONSTRUCTION_ROOT))

from utils.graphics_utils import getProjectionMatrix2


class ProjectionTests(unittest.TestCase):
    def test_intrinsics_projection_uses_positive_camera_depth(self):
        znear = 0.03
        zfar = 250.0
        intrinsics = np.asarray(
            [[400.0, 0.0, 320.0], [0.0, 400.0, 256.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        projection = getProjectionMatrix2(znear, zfar, intrinsics, 512, 640)

        near_clip = projection @ torch.tensor([0.0, 0.0, znear, 1.0])
        far_clip = projection @ torch.tensor([0.0, 0.0, zfar, 1.0])

        self.assertGreater(float(near_clip[3]), 0.0)
        self.assertAlmostEqual(float(near_clip[2] / near_clip[3]), 0.0, places=5)
        self.assertAlmostEqual(float(far_clip[2] / far_clip[3]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
