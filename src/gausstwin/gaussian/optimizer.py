"""Optimizer construction for the GaussTwin-001 parameterization."""

from __future__ import annotations

from typing import Any

import torch

from .parameters import GaussianParameters


def build_optimizer(gaussians: GaussianParameters, learning_rates: dict[str, Any]) -> torch.optim.Optimizer:
    required = {"means", "scales", "rotations", "opacities", "colors"}
    missing = required - learning_rates.keys()
    if missing:
        raise ValueError(f"Missing Gaussian learning rates: {sorted(missing)}")
    return torch.optim.Adam(
        [
            {"params": [gaussians.means], "lr": learning_rates["means"]},
            {"params": [gaussians.log_scales], "lr": learning_rates["scales"]},
            {"params": [gaussians.quaternions], "lr": learning_rates["rotations"]},
            {"params": [gaussians.opacity_logits], "lr": learning_rates["opacities"]},
            {"params": [gaussians.color_logits], "lr": learning_rates["colors"]},
        ]
    )
