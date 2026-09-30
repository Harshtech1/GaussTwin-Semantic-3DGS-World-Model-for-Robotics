"""Trainable Gaussian parameters initialized from a COLMAP sparse cloud."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional


def _inverse_sigmoid(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(1e-6, 1 - 1e-6)
    return torch.log(values / (1 - values))


class GaussianParameters(nn.Module):
    """The minimal gsplat parameter set: position, scale, rotation, alpha, and RGB."""

    def __init__(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        initial_scale: float,
        initial_opacity: float,
    ) -> None:
        super().__init__()
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError("means must have shape [N, 3]")
        if colors.shape != means.shape:
            raise ValueError("colors must have shape [N, 3]")
        if len(means) == 0:
            raise ValueError("at least one Gaussian is required")
        self.means = nn.Parameter(means.float())
        self.log_scales = nn.Parameter(torch.full_like(means, float(initial_scale)).log())
        quaternions = torch.zeros((len(means), 4), dtype=torch.float32, device=means.device)
        quaternions[:, 0] = 1.0
        self.quaternions = nn.Parameter(quaternions)
        self.opacity_logits = nn.Parameter(
            _inverse_sigmoid(torch.full((len(means),), initial_opacity, device=means.device))
        )
        self.color_logits = nn.Parameter(_inverse_sigmoid(colors.float()))

    @classmethod
    def from_sparse_points(
        cls,
        points: list[Any],
        *,
        device: str | torch.device,
        max_points: int,
        initial_scale: float | None,
        initial_opacity: float,
    ) -> "GaussianParameters":
        if not points:
            raise ValueError("COLMAP sparse point cloud is empty")
        selected = points[:max_points]
        means = torch.tensor([point.xyz for point in selected], dtype=torch.float32, device=device)
        colors = torch.tensor([point.rgb for point in selected], dtype=torch.float32, device=device) / 255.0
        if initial_scale is None:
            extent = (means.max(dim=0).values - means.min(dim=0).values).mean().item()
            initial_scale = max(extent / max(len(selected), 1) ** (1 / 3) * 0.5, 1e-4)
        return cls(means, colors, initial_scale=initial_scale, initial_opacity=initial_opacity)

    @property
    def count(self) -> int:
        return self.means.shape[0]

    def scales(self) -> torch.Tensor:
        return self.log_scales.exp()

    def normalized_quaternions(self) -> torch.Tensor:
        return functional.normalize(self.quaternions, dim=-1)

    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    def colors(self) -> torch.Tensor:
        return torch.sigmoid(self.color_logits)
