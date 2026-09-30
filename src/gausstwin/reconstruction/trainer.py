"""A small, reproducible, one-GPU-at-a-time gsplat training loop."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gausstwin.gaussian.losses import photometric_loss, psnr, ssim
from gausstwin.gaussian.optimizer import build_optimizer
from gausstwin.gaussian.parameters import GaussianParameters

from .dataset import ColmapScene
from .initialization import initialize_gaussians
from .renderer import render_view


@dataclass(frozen=True)
class OutputPaths:
    checkpoints: Path
    renders: Path
    metrics: Path

    def create(self) -> None:
        for path in (self.checkpoints, self.renders, self.metrics):
            path.mkdir(parents=True, exist_ok=True)


def output_paths(config: dict) -> OutputPaths:
    values = config["outputs"]
    return OutputPaths(
        checkpoints=Path(values["checkpoints"]),
        renders=Path(values["renders"]),
        metrics=Path(values["metrics"]),
    )


def save_checkpoint(
    path: str | Path,
    gaussians: GaussianParameters,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    metrics: dict[str, Any],
    config: dict,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "gaussians": gaussians.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, gaussians: GaussianParameters, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=gaussians.means.device, weights_only=False)
    gaussians.load_state_dict(checkpoint["gaussians"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def _save_image(path: Path, image: torch.Tensor) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write renders") from exc
    pixels = image.detach().clamp(0, 1).mul(255).byte().cpu().numpy()
    Image.fromarray(pixels, mode="RGB").save(path)


class ThreeDGSTrainer:
    """No-densification baseline trainer intended for a single T4 device."""

    def __init__(self, scene: ColmapScene, config: dict, device: str | None = None) -> None:
        self.scene = scene
        self.config = config
        self.training = config["training"]
        self.device = device or self.training["device"]
        if not self.device.startswith("cuda"):
            raise ValueError("GaussTwin-001 requires an explicit CUDA device, e.g. cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run GaussTwin-001 in Kaggle")
        index = torch.device(self.device).index
        if index is None or index >= torch.cuda.device_count():
            raise ValueError(f"Requested device {self.device} is not visible")
        self._seed(int(self.training["seed"]))
        self.views = [view.scaled_to(config["reconstruction"]["image_resolution"]) for view in scene.views]
        if not self.views:
            raise ValueError("COLMAP model contains no images")
        self.train_views, self.validation_views = self._split_views(self.views)
        self.gaussians = initialize_gaussians(scene, config, self.device)
        self.optimizer = build_optimizer(self.gaussians, self.training["learning_rates"])
        self.paths = output_paths(config)
        self.paths.create()
        self.peak_memory_bytes = 0

    def _seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _split_views(self, views):
        fraction = float(self.config["reconstruction"]["validation_fraction"])
        if len(views) < 2 or fraction <= 0:
            return views, []
        validation_count = min(max(1, round(len(views) * fraction)), len(views) - 1)
        return views[validation_count:], views[:validation_count]

    def _target(self, view) -> torch.Tensor:
        return self.scene.load_image(view, max_dimension=None).to(self.device, non_blocking=True)

    def train(self) -> dict[str, Any]:
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)
        iterations = int(self.training["iterations"])
        checkpoint_interval = int(self.training["checkpoint_interval"])
        render_interval = int(self.training["render_interval"])
        last_loss = float("nan")
        for iteration in range(1, iterations + 1):
            view = self.train_views[(iteration - 1) % len(self.train_views)]
            target = self._target(view)
            self.optimizer.zero_grad(set_to_none=True)
            prediction, _ = render_view(self.gaussians, view, self.device)
            loss = photometric_loss(
                prediction,
                target,
                self.training["loss"]["l1_weight"],
                self.training["loss"]["ssim_weight"],
            )
            loss.backward()
            self.optimizer.step()
            last_loss = float(loss.detach().item())
            self.peak_memory_bytes = max(self.peak_memory_bytes, torch.cuda.max_memory_allocated(self.device))
            if iteration % render_interval == 0 or iteration == iterations:
                _save_image(self.paths.renders / f"train_{iteration:06d}.png", prediction)
            if iteration % checkpoint_interval == 0 or iteration == iterations:
                save_checkpoint(
                    self.paths.checkpoints / f"iter_{iteration:06d}.pt",
                    self.gaussians,
                    self.optimizer,
                    iteration,
                    {"training_loss": last_loss},
                    self.config,
                )
        duration = time.perf_counter() - start
        summary = self.evaluate()
        summary.update(
            {
                "iteration_count": iterations,
                "gaussian_count": self.gaussians.count,
                "image_resolution": self.config["reconstruction"]["image_resolution"],
                "final_loss": last_loss,
                "training_time_seconds": duration,
                "device": self.device,
                "gpu_name": torch.cuda.get_device_name(self.device),
                "peak_vram_bytes": self.peak_memory_bytes,
                "lpips": None,
                "lpips_status": "not computed; no LPIPS dependency is installed",
            }
        )
        (self.paths.metrics / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        views = self.validation_views[: int(self.training["validation_views"])]
        if not views:
            return {"validation_view_count": 0, "psnr": None, "ssim": None}
        values: list[tuple[float, float]] = []
        for view in views:
            target = self._target(view)
            prediction, _ = render_view(self.gaussians, view, self.device)
            values.append((float(psnr(prediction, target).item()), float(ssim(prediction, target).item())))
            _save_image(self.paths.renders / f"validation_{view.image_id:06d}.png", prediction)
        return {
            "validation_view_count": len(values),
            "psnr": sum(value[0] for value in values) / len(values),
            "ssim": sum(value[1] for value in values) / len(values),
        }
