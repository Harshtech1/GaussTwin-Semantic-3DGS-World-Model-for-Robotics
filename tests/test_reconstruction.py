import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gausstwin.gaussian.optimizer import build_optimizer
from gausstwin.gaussian.parameters import GaussianParameters
from gausstwin.reconstruction.camera import CameraIntrinsics, CameraView, quaternion_to_rotation
from gausstwin.reconstruction.colmap import load_colmap_model
from gausstwin.reconstruction.dataset import load_colmap_scene
from gausstwin.reconstruction.initialization import initialize_gaussians
from gausstwin.reconstruction.renderer import render_view
from gausstwin.reconstruction.trainer import load_checkpoint, optimization_step, save_checkpoint

FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "3dgs"


def _materialize_fixture(destination: Path) -> Path:
    """Turn the checked-in JSON fixture into COLMAP's three binary sparse files."""
    spec = json.loads((FIXTURE_ROOT / "scene.json").read_text(encoding="utf-8"))
    sparse_root = destination / "sparse" / "0"
    images_root = destination / "images"
    sparse_root.mkdir(parents=True)
    images_root.mkdir()
    shutil.copy2(FIXTURE_ROOT / "images" / "frame.ppm", images_root / "frame.ppm")
    camera = spec["camera"]
    image = spec["image"]
    points = spec["points"]
    (sparse_root / "cameras.bin").write_bytes(
        struct.pack(
            "<QIiQQ" + "d" * len(camera["params"]),
            1,
            camera["id"],
            camera["model_id"],
            camera["width"],
            camera["height"],
            *camera["params"],
        )
    )
    (sparse_root / "images.bin").write_bytes(
        struct.pack("<QIdddddddI", 1, image["id"], *image["qvec"], *image["tvec"], image["camera_id"])
        + image["name"].encode("utf-8")
        + b"\x00"
        + struct.pack("<Q", 0)
    )
    point_records = bytearray(struct.pack("<Q", len(points)))
    for point in points:
        point_records.extend(
            struct.pack("<QdddBBBdQ", point["id"], *point["xyz"], *point["rgb"], point["error"], 0)
        )
    (sparse_root / "points3D.bin").write_bytes(point_records)
    return destination


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

    def test_local_gaussian_scales_are_finite_positive_and_density_aware(self):
        dense = [(0.00, 0.00, 0.00), (0.01, 0.00, 0.00), (0.00, 0.01, 0.00), (0.00, 0.00, 0.01)]
        sparse = [(10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (10.0, 1.0, 0.0), (10.0, 0.0, 1.0)]
        points = [SimpleNamespace(xyz=xyz, rgb=(128, 128, 128)) for xyz in dense + sparse]
        gaussians = GaussianParameters.from_sparse_points(
            points, device="cpu", max_points=10, initial_scale=None, initial_opacity=0.1
        )
        scales = gaussians.scales()[:, 0]
        self.assertTrue(torch.isfinite(scales).all())
        self.assertTrue((scales > 0).all())
        self.assertLess(scales[:4].median(), scales[4:].median())

    def test_duplicate_coordinates_do_not_create_zero_or_invalid_scales(self):
        points = [SimpleNamespace(xyz=(0.0, 0.0, 0.0), rgb=(255, 0, 0)) for _ in range(4)]
        gaussians = GaussianParameters.from_sparse_points(
            points, device="cpu", max_points=10, initial_scale=None, initial_opacity=0.1
        )
        scales = gaussians.scales()
        self.assertTrue(torch.isfinite(scales).all())
        self.assertTrue((scales > 0).all())

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

    def test_minimal_3dgs_training_contract(self):
        """CPU contract: real COLMAP/init/optimizer/checkpoint, mocked gsplat only."""
        with tempfile.TemporaryDirectory() as directory:
            scene = load_colmap_scene(_materialize_fixture(Path(directory)))
            self.assertEqual(len(scene.views), 1)
            self.assertEqual(len(scene.points), 2)
            config = {
                "reconstruction": {"max_points": 10, "initial_scale": 0.1, "initial_opacity": 0.1},
                "training": {"loss": {"l1_weight": 0.8, "ssim_weight": 0.2}},
            }
            gaussians = initialize_gaussians(scene, config, device="cpu")
            optimizer = build_optimizer(
                gaussians,
                {"means": 1e-3, "scales": 1e-3, "rotations": 1e-3, "opacities": 1e-3, "colors": 1e-2},
            )
            target = torch.zeros((8, 8, 3), dtype=torch.float32)
            colors_before = gaussians.color_logits.detach().clone()

            def mock_renderer(parameters, view, device):
                self.assertEqual(device, "cpu")
                self.assertEqual((view.intrinsics.width, view.intrinsics.height), (8, 8))
                rgb = parameters.colors().mean(dim=0).view(1, 1, 3).expand(8, 8, 3)
                return rgb, torch.ones((8, 8, 1), dtype=rgb.dtype)

            prediction, loss = optimization_step(
                gaussians, optimizer, scene.views[0], target, "cpu", config["training"], mock_renderer
            )
            self.assertEqual(tuple(prediction.shape), (8, 8, 3))
            self.assertTrue(torch.isfinite(loss))
            self.assertFalse(torch.equal(colors_before, gaussians.color_logits.detach()))

            checkpoint_path = Path(directory) / "integration.pt"
            save_checkpoint(checkpoint_path, gaussians, optimizer, 1, {"training_loss": loss.item()}, config)
            means_after_step = gaussians.means.detach().clone()
            with torch.no_grad():
                gaussians.means.add_(1.0)
            restored = load_checkpoint(checkpoint_path, gaussians, optimizer)
            self.assertEqual(restored["iteration"], 1)
            self.assertTrue(torch.allclose(gaussians.means, means_after_step))


if __name__ == "__main__":
    unittest.main()
