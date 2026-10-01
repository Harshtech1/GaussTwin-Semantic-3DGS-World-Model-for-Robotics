"""Single-GPU adaptive 3DGS trainer for GaussTwin-002."""

from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from gausstwin.gaussian.adaptive import AdaptiveGaussianParameters
from gausstwin.gaussian.losses import photometric_loss, psnr, ssim
from gausstwin.gaussian.optimizer import build_optimizer
from gausstwin.gaussian.refinement import RefinementStatistics, refine
from gausstwin.gaussian.trajectory import TrajectoryLogger

from .adaptive_renderer import render_adaptive_view
from .dataset import ColmapScene
from .initialization import initialize_gaussians
from .trainer import OutputPaths, output_paths

ADAPTIVE_CHECKPOINT_SCHEMA_VERSION = 2


def save_adaptive_checkpoint(
    path: str | Path,
    gaussians: AdaptiveGaussianParameters,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    payload: dict[str, Any],
    config: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": ADAPTIVE_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": "adaptive_3dgs",
            "iteration": iteration,
            "gaussians": gaussians.state_dict(),
            "optimizer": optimizer.state_dict(),
            "payload": payload,
            "config": config,
        },
        path,
    )


def load_adaptive_checkpoint(
    path: str | Path, gaussians: AdaptiveGaussianParameters, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=gaussians.means.device, weights_only=False)
    schema_version = int(checkpoint.get("schema_version", 1))
    if schema_version not in {1, ADAPTIVE_CHECKPOINT_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported checkpoint schema version: {schema_version}")
    gaussians.load_state_dict(checkpoint["gaussians"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint["schema_version"] = schema_version
    return checkpoint


def _git_commit() -> str:
    try:
        repository = Path(__file__).resolve().parents[3]
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class AdaptiveThreeDGSTrainer:
    """GaussTwin-002: fixed rendering path, adaptive Gaussian representation."""

    def __init__(self, scene: ColmapScene, config: dict[str, Any], device: str | None = None) -> None:
        self.scene, self.config = scene, config
        self.training = config["training"]
        self.refinement = config["refinement"]
        self.device = device or self.training["device"]
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("GaussTwin-002 requires one CUDA device in Kaggle")
        index = torch.device(self.device).index
        if index is None or index >= torch.cuda.device_count():
            raise ValueError(f"Requested device {self.device} is not visible")
        self._seed(int(self.training["seed"]))
        self.views = [view.scaled_to(config["reconstruction"]["image_resolution"]) for view in scene.views]
        if not self.views:
            raise ValueError("COLMAP model contains no images")
        self.train_views, self.validation_views = self._split_views(self.views)
        fixed = initialize_gaussians(scene, config, self.device)
        self.gaussians = AdaptiveGaussianParameters.from_fixed(fixed)
        self.optimizer = build_optimizer(self.gaussians, self.training["learning_rates"])
        self.statistics = RefinementStatistics(self.gaussians.count, self.device)
        self.paths: OutputPaths = output_paths(config)
        self.paths.create()
        self.trajectory = TrajectoryLogger(self.paths.metrics)
        self.checkpoints: list[str] = []
        self.peak_memory_bytes = 0

    def _seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _split_views(self, views):
        fraction = float(self.config["reconstruction"]["validation_fraction"])
        if len(views) < 2 or fraction <= 0:
            return views, []
        count = min(max(1, round(len(views) * fraction)), len(views) - 1)
        return views[count:], views[:count]

    def _target(self, view) -> torch.Tensor:
        return self.scene.load_image(view, max_dimension=None).to(self.device, non_blocking=True)

    def _manifest(self) -> dict[str, Any]:
        return {
            "experiment": "GaussTwin-002",
            "source_git_commit": _git_commit(),
            "resolved_config": self.config,
            "device": self.device,
            "gpu_name": torch.cuda.get_device_name(self.device),
            "seed": self.training["seed"],
            "initialization_mode": "COLMAP local-neighborhood scale initialization",
            "refinement_settings": self.refinement,
            "checkpoint_schema_version": ADAPTIVE_CHECKPOINT_SCHEMA_VERSION,
            "checkpoints": self.checkpoints,
            "gaussian_count_trajectory": [entry["gaussian_count"] for entry in self.trajectory.entries],
            "metric_trajectory": self.trajectory.entries,
        }

    @torch.no_grad()
    def evaluate(self, iteration: int) -> dict[str, Any]:
        views = self.validation_views[: int(self.training["validation_views"])]
        if not views:
            return {"iteration": iteration, "validation_view_count": 0, "psnr": None, "ssim": None}
        scores: list[tuple[float, float]] = []
        for view in views:
            target = self._target(view)
            output = render_adaptive_view(self.gaussians, view, self.device, absgrad=False)
            scores.append((float(psnr(output.rgb, target)), float(ssim(output.rgb, target))))
        return {
            "iteration": iteration,
            "validation_view_count": len(scores),
            "psnr": sum(score[0] for score in scores) / len(scores),
            "ssim": sum(score[1] for score in scores) / len(scores),
        }

    def train(self) -> dict[str, Any]:
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)
        for iteration in range(1, int(self.training["iterations"]) + 1):
            view = self.train_views[(iteration - 1) % len(self.train_views)]
            target = self._target(view)
            self.optimizer.zero_grad(set_to_none=True)
            output = render_adaptive_view(self.gaussians, view, self.device)
            loss = photometric_loss(
                output.rgb, target, self.training["loss"]["l1_weight"], self.training["loss"]["ssim_weight"]
            )
            loss.backward()
            self.statistics.accumulate(output.meta)
            self.optimizer.step()
            event: dict[str, Any] | None = None
            if iteration >= int(self.refinement["warmup_iterations"]) and iteration % int(self.refinement["interval"]) == 0:
                result, selection = refine(self.gaussians, self.statistics, self.refinement, int(self.training["seed"]) + iteration)
                self.optimizer = build_optimizer(self.gaussians, self.training["learning_rates"])
                self.statistics.reset(self.gaussians.count)
                event = {
                    "iteration": iteration,
                    "old_gaussian_count": result.old_count,
                    "added_count": result.added_count,
                    "pruned_count": result.pruned_count,
                    "replaced_count": result.replaced_count,
                    "resulting_gaussian_count": result.resulting_count,
                    "split_count": len(selection.split_indices),
                    "duplicate_count": len(selection.duplicate_indices),
                    **selection.diagnostics.as_dict(),
                }
            self.peak_memory_bytes = max(self.peak_memory_bytes, torch.cuda.max_memory_allocated(self.device))
            entry = {
                "iteration": iteration,
                "loss": float(loss.detach()),
                "psnr": float(psnr(output.rgb.detach(), target).detach()),
                "ssim": float(ssim(output.rgb.detach(), target).detach()),
                "gaussian_count": self.gaussians.count,
                "refinement_event": event,
            }
            self.trajectory.append(entry)
            if iteration % int(self.training["checkpoint_interval"]) == 0 or iteration == int(self.training["iterations"]):
                evaluation = self.evaluate(iteration)
                self.trajectory.write_evaluation(iteration, evaluation)
                checkpoint = self.paths.checkpoints / f"iter_{iteration:06d}.pt"
                if checkpoint.exists():
                    raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint}")
                save_adaptive_checkpoint(checkpoint, self.gaussians, self.optimizer, iteration, evaluation, self.config)
                self.checkpoints.append(str(checkpoint))
                self.trajectory.write_manifest(self._manifest())
        summary = {
            "training_time_seconds": time.perf_counter() - start,
            "peak_vram_bytes": self.peak_memory_bytes,
            "gaussian_count": self.gaussians.count,
            "checkpoint_count": len(self.checkpoints),
        }
        (self.paths.metrics / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self.trajectory.write_manifest(self._manifest())
        return summary
