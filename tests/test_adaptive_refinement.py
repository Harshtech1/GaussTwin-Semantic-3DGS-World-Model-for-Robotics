from pathlib import Path
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gausstwin.gaussian.adaptive import AdaptiveGaussianParameters
from gausstwin.gaussian.optimizer import build_optimizer
from gausstwin.gaussian.parameters import GaussianParameters
from gausstwin.gaussian.refinement import (
    RefinementStatistics,
    compute_refinement_diagnostics,
    inspect_rasterization_metadata,
    pruning_mask,
    refine,
    select_candidates,
)
from gausstwin.gaussian.trajectory import TrajectoryLogger
from gausstwin.reconstruction.adaptive_trainer import (
    ADAPTIVE_CHECKPOINT_SCHEMA_VERSION,
    load_adaptive_checkpoint,
    save_adaptive_checkpoint,
)
from gausstwin.reconstruction.adaptive_renderer import render_adaptive_view
from gausstwin.reconstruction.camera import CameraIntrinsics, CameraView
from gausstwin.reconstruction.trainer import save_checkpoint


def _gaussians() -> AdaptiveGaussianParameters:
    means = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]])
    log_scales = torch.tensor([[0.0, 0.0, 0.0], [-4.0, -4.0, -4.0], [-4.0, -4.0, -4.0], [0.0, 0.0, 0.0]])
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)
    opacities = torch.tensor([0.0, 0.0, -10.0, 0.0])
    colors = torch.zeros((4, 3))
    return AdaptiveGaussianParameters(means, log_scales, quaternions, opacities, colors)


def _settings(**overrides):
    settings = {
        "min_gaussian_count": 2,
        "max_gaussian_count": 8,
        "prune_opacity_threshold": 0.01,
        "min_observations": 2,
        "gradient_threshold": 0.5,
        "max_candidates": 4,
        "max_candidate_fraction": 1.0,
        "split_scale_threshold": 0.5,
        "split_scale_factor": 0.8,
        "split_jitter_factor": 0.5,
        "duplicate_jitter_factor": 0.25,
    }
    settings.update(overrides)
    return settings


class AdaptiveRefinementTests(unittest.TestCase):
    def test_gradient_accumulation_and_visibility_observations(self):
        statistics = RefinementStatistics(3, "cpu")
        means2d = SimpleNamespace(absgrad=torch.tensor([[[3.0, 4.0], [6.0, 8.0], [5.0, 12.0]]]))
        means2d.shape = torch.Size((1, 3, 2))
        statistics.accumulate({"means2d": means2d, "radii": torch.tensor([[[1.0, 1.0], [0.0, 1.0], [2.0, 2.0]]])})
        statistics.accumulate({"means2d": means2d, "radii": torch.tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]]])})
        self.assertTrue(torch.equal(statistics.observations, torch.tensor([2, 1, 1])))
        self.assertTrue(torch.allclose(statistics.mean_gradient(), torch.tensor([5.0, 10.0, 13.0])))

    def test_nonpacked_metadata_shapes_count_one_view_slot(self):
        means2d = SimpleNamespace(absgrad=torch.ones((1, 4, 2)))
        means2d.shape = torch.Size((1, 4, 2))
        metadata = {"means2d": means2d, "radii": torch.tensor([[[1.0, 1.0], [0.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])}
        diagnostics = inspect_rasterization_metadata(metadata, 4)
        self.assertEqual(diagnostics.means2d_shape, (1, 4, 2))
        self.assertEqual(diagnostics.radii_shape, (1, 4, 2))
        self.assertEqual(diagnostics.observation_slots, 1)
        statistics = RefinementStatistics(4, "cpu")
        for _ in range(50):
            statistics.accumulate(metadata)
        self.assertEqual(statistics.observations.max().item(), 50)

    def test_metadata_shape_mismatch_is_reported(self):
        means2d = SimpleNamespace(absgrad=torch.ones((1, 3, 2)))
        means2d.shape = torch.Size((1, 3, 2))
        with self.assertRaises(ValueError):
            inspect_rasterization_metadata({"means2d": means2d, "radii": torch.ones((1, 3, 2))}, 4)

    def test_two_camera_slots_and_forward_metadata_without_absgrad(self):
        means2d = SimpleNamespace()
        means2d.shape = torch.Size((2, 3, 2))
        metadata = {"means2d": means2d, "radii": torch.ones((2, 3, 2))}
        diagnostics = inspect_rasterization_metadata(metadata, 3)
        self.assertEqual(diagnostics.observation_slots, 2)
        self.assertIsNone(diagnostics.absgrad_shape)
        with self.assertRaises(RuntimeError):
            RefinementStatistics(3, "cpu").accumulate(metadata)

    def test_candidate_ranking_requires_gradient_and_observations(self):
        gaussians = _gaussians()
        statistics = RefinementStatistics(4, "cpu")
        statistics.gradient_sum = torch.tensor([2.0, 20.0, 100.0, 0.9])
        statistics.observations = torch.tensor([2, 10, 1, 2])
        selection = select_candidates(gaussians, statistics, _settings())
        self.assertEqual(selection.split_indices.tolist(), [0])
        self.assertEqual(selection.duplicate_indices.tolist(), [1])
        self.assertEqual(selection.prune_indices.tolist(), [])
        self.assertEqual(selection.diagnostics.eligible_count, 2)
        self.assertEqual(selection.diagnostics.candidate_limit, 4)

    def test_refinement_diagnostics_percentiles_and_empty_safety(self):
        diagnostics = compute_refinement_diagnostics(
            torch.tensor([0.0, 1.0, 2.0, 3.0]),
            torch.tensor([0, 2, 4, 6]),
            eligible_count=3,
            candidate_limit=2,
        )
        self.assertEqual(diagnostics.gradient_min, 0.0)
        self.assertEqual(diagnostics.gradient_p50, 1.5)
        self.assertAlmostEqual(diagnostics.gradient_p90, 2.7, places=5)
        self.assertEqual(diagnostics.gradient_max, 3.0)
        self.assertEqual(diagnostics.observation_min, 0)
        self.assertEqual(diagnostics.observation_p50, 3.0)
        self.assertEqual(diagnostics.observation_max, 6)
        self.assertEqual(diagnostics.eligible_count, 3)
        empty = compute_refinement_diagnostics(
            torch.tensor([]), torch.tensor([], dtype=torch.long), eligible_count=0, candidate_limit=0
        )
        self.assertIsNone(empty.gradient_p50)
        self.assertIsNone(empty.observation_p90)

    def test_deterministic_split_and_duplicate_and_count_change(self):
        first, second = _gaussians(), _gaussians()
        result_one = first.mutate(
            torch.tensor([0]), torch.tensor([1]), torch.tensor([], dtype=torch.long),
            seed=9, split_scale_factor=0.8, split_jitter_factor=0.5, duplicate_jitter_factor=0.25,
        )
        result_two = second.mutate(
            torch.tensor([0]), torch.tensor([1]), torch.tensor([], dtype=torch.long),
            seed=9, split_scale_factor=0.8, split_jitter_factor=0.5, duplicate_jitter_factor=0.25,
        )
        self.assertEqual((result_one.old_count, result_one.added_count, result_one.replaced_count), (4, 3, 1))
        self.assertEqual(first.count, 6)
        self.assertTrue(torch.allclose(first.means, second.means))
        self.assertTrue(torch.allclose(first.log_scales, second.log_scales))

    def test_pruning_mask_minimum_count_and_hard_maximum_safeguards(self):
        gaussians = _gaussians()
        statistics = RefinementStatistics(4, "cpu")
        statistics.observations.fill_(2)
        statistics.gradient_sum[0] = 10.0
        mask = pruning_mask(gaussians.opacities(), statistics.observations, opacity_threshold=0.01, min_observations=2)
        self.assertEqual(mask.tolist(), [False, False, True, False])
        selection = select_candidates(gaussians, statistics, _settings(min_gaussian_count=4, max_gaussian_count=4))
        self.assertEqual(selection.prune_indices.tolist(), [])
        self.assertEqual(selection.split_indices.tolist(), [])
        self.assertEqual(selection.duplicate_indices.tolist(), [])

    def test_adaptive_renderer_requests_explicit_indexing_and_absgrad(self):
        gaussians = _gaussians()
        view = CameraView(
            image_id=1,
            image_name="unused.png",
            intrinsics=CameraIntrinsics(4, 4, 2.0, 2.0, 2.0, 2.0),
            world_to_camera=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                             (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        )

        def rasterization(**kwargs):
            self.assertFalse(kwargs["packed"])
            self.assertTrue(kwargs["absgrad"])
            return torch.zeros((1, 4, 4, 3)), torch.zeros((1, 4, 4, 1)), {"radii": torch.ones((1, 4))}

        fake_gsplat = types.ModuleType("gsplat")
        fake_gsplat.rasterization = rasterization
        with patch.dict(sys.modules, {"gsplat": fake_gsplat}):
            result = render_adaptive_view(gaussians, view, "cpu")
        self.assertEqual(tuple(result.rgb.shape), (4, 4, 3))

    def test_refinement_resets_are_external_and_event_counts_match(self):
        gaussians = _gaussians()
        statistics = RefinementStatistics(4, "cpu")
        statistics.gradient_sum = torch.tensor([10.0, 10.0, 0.0, 0.0])
        statistics.observations = torch.tensor([2, 2, 2, 2])
        result, _ = refine(gaussians, statistics, _settings(), seed=3)
        self.assertEqual(result.resulting_count, gaussians.count)
        self.assertEqual(result.resulting_count, result.old_count + result.added_count - result.replaced_count - result.pruned_count)
        statistics.reset(gaussians.count)
        self.assertEqual(statistics.count, gaussians.count)
        self.assertEqual(statistics.observations.sum().item(), 0)

    def test_append_only_trajectory_and_unique_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            trajectory = TrajectoryLogger(directory)
            trajectory.append({"iteration": 250, "loss": 1.0, "psnr": 2.0, "ssim": 0.1, "gaussian_count": 4, "refinement_event": None})
            with self.assertRaises(FileExistsError):
                trajectory.append({"iteration": 250})
            path = trajectory.write_evaluation(250, {"psnr": 2.0, "ssim": 0.1})
            self.assertEqual(path.name, "evaluation_iter_000250.json")
            with self.assertRaises(FileExistsError):
                trajectory.write_evaluation(250, {"psnr": 3.0})

    def test_checkpoint_schema_version_compatibility(self):
        fixed = GaussianParameters(torch.tensor([[0.0, 0.0, 1.0]]), torch.ones((1, 3)), 0.1, 0.1)
        legacy_optimizer = build_optimizer(
            fixed, {"means": 1e-3, "scales": 1e-3, "rotations": 1e-3, "opacities": 1e-3, "colors": 1e-3}
        )
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / "legacy.pt"
            save_checkpoint(legacy_path, fixed, legacy_optimizer, 1, {}, {})
            adaptive = AdaptiveGaussianParameters.from_fixed(fixed)
            legacy = load_adaptive_checkpoint(legacy_path, adaptive)
            self.assertEqual(legacy["schema_version"], 1)
            optimizer = build_optimizer(
                adaptive, {"means": 1e-3, "scales": 1e-3, "rotations": 1e-3, "opacities": 1e-3, "colors": 1e-3}
            )
            adaptive_path = Path(directory) / "adaptive.pt"
            save_adaptive_checkpoint(adaptive_path, adaptive, optimizer, 2, {}, {})
            loaded = load_adaptive_checkpoint(adaptive_path, adaptive, optimizer)
            self.assertEqual(loaded["schema_version"], ADAPTIVE_CHECKPOINT_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
