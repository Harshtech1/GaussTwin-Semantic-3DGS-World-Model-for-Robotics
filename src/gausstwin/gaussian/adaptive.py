"""Structurally mutable Gaussian parameters for the GaussTwin-002 baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional

from .parameters import GaussianParameters


def _inverse_sigmoid(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(1e-6, 1 - 1e-6)
    return torch.log(values / (1 - values))


def _half_opacity(logits: torch.Tensor) -> torch.Tensor:
    return _inverse_sigmoid(torch.sigmoid(logits) * 0.5)


def _deterministic_directions(indices: torch.Tensor, seed: int) -> torch.Tensor:
    """Generate stable unit vectors without consuming global RNG state."""
    values = indices.to(dtype=torch.float32) + float(seed) * 0.137
    directions = torch.stack(
        (torch.sin(values * 12.9898), torch.cos(values * 78.233), torch.sin(values * 37.719)), dim=1
    )
    return functional.normalize(directions, dim=1)


@dataclass(frozen=True)
class MutationResult:
    old_count: int
    added_count: int
    pruned_count: int
    replaced_count: int
    resulting_count: int


class AdaptiveGaussianParameters(nn.Module):
    """Gaussian tensors with deterministic split, duplicate, and prune operations."""

    def __init__(
        self,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        quaternions: torch.Tensor,
        opacity_logits: torch.Tensor,
        color_logits: torch.Tensor,
    ) -> None:
        super().__init__()
        self._validate(means, log_scales, quaternions, opacity_logits, color_logits)
        self.means = nn.Parameter(means.detach().clone())
        self.log_scales = nn.Parameter(log_scales.detach().clone())
        self.quaternions = nn.Parameter(quaternions.detach().clone())
        self.opacity_logits = nn.Parameter(opacity_logits.detach().clone())
        self.color_logits = nn.Parameter(color_logits.detach().clone())

    @classmethod
    def from_fixed(cls, fixed: GaussianParameters) -> "AdaptiveGaussianParameters":
        return cls(
            fixed.means,
            fixed.log_scales,
            fixed.quaternions,
            fixed.opacity_logits,
            fixed.color_logits,
        )

    @staticmethod
    def _validate(
        means: torch.Tensor,
        log_scales: torch.Tensor,
        quaternions: torch.Tensor,
        opacity_logits: torch.Tensor,
        color_logits: torch.Tensor,
    ) -> None:
        count = means.shape[0]
        if means.ndim != 2 or means.shape[1] != 3 or count == 0:
            raise ValueError("means must have non-empty shape [N, 3]")
        if log_scales.shape != means.shape or color_logits.shape != means.shape:
            raise ValueError("scales and colors must have shape [N, 3]")
        if quaternions.shape != (count, 4) or opacity_logits.shape != (count,):
            raise ValueError("invalid quaternion or opacity shape")

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

    def mutate(
        self,
        split_indices: torch.Tensor,
        duplicate_indices: torch.Tensor,
        prune_indices: torch.Tensor,
        *,
        seed: int,
        split_scale_factor: float,
        split_jitter_factor: float,
        duplicate_jitter_factor: float,
    ) -> MutationResult:
        """Apply one deterministic structural event and replace parameter tensors."""
        device = self.means.device
        split_indices = torch.unique(split_indices.to(device=device, dtype=torch.long), sorted=True)
        duplicate_indices = torch.unique(duplicate_indices.to(device=device, dtype=torch.long), sorted=True)
        prune_indices = torch.unique(prune_indices.to(device=device, dtype=torch.long), sorted=True)
        for indices in (split_indices, duplicate_indices, prune_indices):
            if len(indices) and (indices.min() < 0 or indices.max() >= self.count):
                raise IndexError("Gaussian mutation index out of range")
        if len(torch.unique(torch.cat((split_indices, duplicate_indices)))) != len(split_indices) + len(duplicate_indices):
            raise ValueError("split and duplicate candidates must be disjoint")
        if len(torch.unique(torch.cat((split_indices, prune_indices)))) != len(split_indices) + len(prune_indices):
            raise ValueError("split and prune candidates must be disjoint")
        if len(torch.unique(torch.cat((duplicate_indices, prune_indices)))) != len(duplicate_indices) + len(prune_indices):
            raise ValueError("duplicate and prune candidates must be disjoint")

        old_count = self.count
        keep = torch.ones(old_count, dtype=torch.bool, device=device)
        keep[split_indices] = False
        keep[prune_indices] = False
        means, log_scales = self.means.detach().clone(), self.log_scales.detach().clone()
        quaternions, opacities, colors = (
            self.quaternions.detach().clone(),
            self.opacity_logits.detach().clone(),
            self.color_logits.detach().clone(),
        )
        if len(duplicate_indices):
            opacities[duplicate_indices] = _half_opacity(opacities[duplicate_indices])

        child_means: list[torch.Tensor] = []
        child_scales: list[torch.Tensor] = []
        child_quaternions: list[torch.Tensor] = []
        child_opacities: list[torch.Tensor] = []
        child_colors: list[torch.Tensor] = []
        if len(split_indices):
            directions = _deterministic_directions(split_indices, seed)
            offset = self.scales().detach()[split_indices].mean(dim=1, keepdim=True) * split_jitter_factor
            parent_means = means[split_indices]
            child_means.extend((parent_means + directions * offset, parent_means - directions * offset))
            child_scales.extend((log_scales[split_indices] + math.log(split_scale_factor),) * 2)
            child_quaternions.extend((quaternions[split_indices],) * 2)
            half = _half_opacity(opacities[split_indices])
            child_opacities.extend((half,) * 2)
            child_colors.extend((colors[split_indices],) * 2)
        if len(duplicate_indices):
            directions = _deterministic_directions(duplicate_indices, seed + 1)
            offset = self.scales().detach()[duplicate_indices].mean(dim=1, keepdim=True) * duplicate_jitter_factor
            child_means.append(means[duplicate_indices] + directions * offset)
            child_scales.append(log_scales[duplicate_indices])
            child_quaternions.append(quaternions[duplicate_indices])
            child_opacities.append(opacities[duplicate_indices])
            child_colors.append(colors[duplicate_indices])

        def combine(original: torch.Tensor, children: list[torch.Tensor]) -> torch.Tensor:
            return torch.cat((original[keep], *children), dim=0) if children else original[keep]

        self.means = nn.Parameter(combine(means, child_means))
        self.log_scales = nn.Parameter(combine(log_scales, child_scales))
        self.quaternions = nn.Parameter(combine(quaternions, child_quaternions))
        self.opacity_logits = nn.Parameter(combine(opacities, child_opacities))
        self.color_logits = nn.Parameter(combine(colors, child_colors))
        return MutationResult(
            old_count=old_count,
            added_count=2 * len(split_indices) + len(duplicate_indices),
            pruned_count=len(prune_indices),
            replaced_count=len(split_indices),
            resulting_count=self.count,
        )
