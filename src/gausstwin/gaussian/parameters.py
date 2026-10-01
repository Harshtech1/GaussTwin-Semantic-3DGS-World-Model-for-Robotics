"""Trainable Gaussian parameters initialized from a COLMAP sparse cloud."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

_LOCAL_NEIGHBOR_COUNT = 3
_DISTANCE_EPSILON = 1e-12
_MIN_INITIAL_SCALE = 1e-5
_DISTANCE_CHUNK_SIZE = 512


def _inverse_sigmoid(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(1e-6, 1 - 1e-6)
    return torch.log(values / (1 - values))


class GaussianParameters(nn.Module):
    """The minimal gsplat parameter set: position, scale, rotation, alpha, and RGB."""

    def __init__(
        self,
        means: torch.Tensor,
        colors: torch.Tensor,
        initial_scale: float | torch.Tensor,
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
        if isinstance(initial_scale, torch.Tensor):
            if initial_scale.ndim == 1:
                initial_scale = initial_scale.unsqueeze(-1).expand_as(means)
            if initial_scale.shape != means.shape:
                raise ValueError("per-Gaussian initial_scale must have shape [N] or [N, 3]")
            scales = initial_scale.to(device=means.device, dtype=means.dtype)
        else:
            scales = torch.full_like(means, float(initial_scale))
        self.log_scales = nn.Parameter(scales.clamp_min(_MIN_INITIAL_SCALE).log())
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
            initial_scale = _local_neighbor_scales(means)
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


@torch.no_grad()
def _local_neighbor_scales(means: torch.Tensor) -> torch.Tensor:
    """Estimate one isotropic scale per point from its nearest non-zero neighbors.

    This follows the local-spacing principle of 3DGS initialization while avoiding
    a global scene-extent heuristic. Distances are evaluated in chunks so a sparse
    COLMAP cloud does not require a full ``N x N`` distance matrix in memory.
    """
    point_count = len(means)
    if point_count == 0:
        raise ValueError("at least one point is required to estimate local scales")
    neighbor_count = min(_LOCAL_NEIGHBOR_COUNT, point_count)
    local_scales = torch.empty(point_count, dtype=means.dtype, device=means.device)
    valid_counts = torch.zeros(point_count, dtype=torch.long, device=means.device)
    for start in range(0, point_count, _DISTANCE_CHUNK_SIZE):
        end = min(start + _DISTANCE_CHUNK_SIZE, point_count)
        distances = torch.cdist(means[start:end], means)
        distances.masked_fill_(distances <= _DISTANCE_EPSILON, torch.inf)
        nearest = torch.topk(distances, k=neighbor_count, largest=False, sorted=True).values
        valid = torch.isfinite(nearest)
        counts = valid.sum(dim=1)
        valid_counts[start:end] = counts
        safe_nearest = torch.where(valid, nearest, torch.zeros_like(nearest))
        # Median is robust when all three local neighbors exist; use the mean of
        # available neighbors for boundary/very-small point clouds.
        median = nearest[:, min(1, neighbor_count - 1)]
        mean_available = safe_nearest.sum(dim=1) / counts.clamp_min(1)
        local_scales[start:end] = torch.where(counts >= _LOCAL_NEIGHBOR_COUNT, median, mean_available)

    usable = local_scales[valid_counts > 0]
    fallback = usable.median() if len(usable) else means.new_tensor(_MIN_INITIAL_SCALE)
    local_scales = torch.where(valid_counts > 0, local_scales, fallback)
    return local_scales.clamp_min(_MIN_INITIAL_SCALE)
