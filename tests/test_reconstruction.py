from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gausstwin.gaussian.optimizer import build_optimizer
from gausstwin.gaussian.parameters import GaussianParameters
from gausstwin.reconstruction.camera import CameraIntrinsics, CameraView, quaternion_to_rotation
from gausstwin.reconstruction.colmap import load_colmap_model
from gausstwin.reconstruction.renderer import render_view
from gausstwin.reconstruction.trainer import load_checkpoint, save_checkpoint


class ReconstructionTests(unittest.TestCase):
    def test_camera_scaling_and_identity_rotation(self):
        intrinsics = CameraIntrinsics(1000, 500, 800.0, 800.0, 500.0, 250.0)
        scaled = intrinsics.scaled_to(500)
        self.assertEqual((scaled.width, scaled.height), (500, 250))
        self.assertEqual(scaled.fx, 400.0)
        self.assertEqual(quaternion_to_rotation((1.0, 0.0, 0.0, 0.0))[0], (1.0, 0.0, 0.0))

    def test_colmap_binary_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cameras.bin").write_bytes(
                struct.pack("<QIiQQdddd", 1, 1, 1, 640, 480, 500.0, 500.0, 320.0, 240.0)
            )
            (root / "images.bin").write_bytes(
                struct.pack("<QIdddddddI", 1, 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
                + b"frame.png\x00"
                + struct.pack("<Q", 0)
            )
            (root / "points3D.bin").write_bytes(
                struct.pack("<QQdddBBBdQ", 1, 7, 1.0, 2.0, 3.0, 10, 20, 30, 0.2, 0)
            )
            model = load_colmap_model(root)
            self.assertEqual(model.cameras[1].intrinsics.width, 640)
            self.assertEqual(model.images[0].name, "frame.png")
            self.assertEqual(model.points[0].rgb, (10, 20, 30))
            self.assertEqual(model.camera_views()[0].world_to_camera[2][3], 0.0)

    def test_gaussian_shapes_and_checkpoint_serialization(self):
        means = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        gaussians = GaussianParameters(means, colors, initial_scale=0.1, initial_opacity=0.1)
        self.assertEqual(tuple(gaussians.scales().shape), (2, 3))
        self.assertEqual(tuple(gaussians.normalized_quaternions().shape), (2, 4))
        self.assertEqual(tuple(gaussians.opacities().shape), (2,))
        optimizer = build_optimizer(
            gaussians,
            {"means": 1e-3, "scales": 1e-3, "rotations": 1e-3, "opacities": 1e-3, "colors": 1e-3},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, gaussians, optimizer, 3, {"training_loss": 0.1}, {"seed": 42})
            restored = GaussianParameters(torch.zeros_like(means), colors, initial_scale=0.1, initial_opacity=0.1)
            metadata = load_checkpoint(path, restored)
            self.assertEqual(metadata["iteration"], 3)
            self.assertTrue(torch.allclose(restored.means, means))

    def test_renderer_interface_with_mocked_gsplat(self):
        gaussians = GaussianParameters(
            torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0]]), 0.1, 0.1
        )
        view = CameraView(
            image_id=1,
            image_name="frame.png",
            intrinsics=CameraIntrinsics(8, 6, 5.0, 5.0, 4.0, 3.0),
            world_to_camera=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                             (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        )

        def rasterization(**kwargs):
            self.assertEqual(kwargs["width"], 8)
            self.assertEqual(kwargs["height"], 6)
            return torch.zeros((1, 6, 8, 3)), torch.zeros((1, 6, 8, 1)), {}

        fake_gsplat = types.ModuleType("gsplat")
        fake_gsplat.rasterization = rasterization
        with patch.dict(sys.modules, {"gsplat": fake_gsplat}):
            render, alpha = render_view(gaussians, view, "cpu")
        self.assertEqual(tuple(render.shape), (6, 8, 3))
        self.assertEqual(tuple(alpha.shape), (6, 8, 1))


if __name__ == "__main__":
    unittest.main()
