"""Gradient-aware, conservative adaptive Gaussian refinement for GaussTwin-002."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .adaptive import AdaptiveGaussianParameters, MutationResult


@dataclass(frozen=True)
class CandidateSelection:
    split_indices: torch.Tensor
    duplicate_indices: torch.Tensor
    prune_indices: torch.Tensor
    diagnostics: "RefinementDiagnostics"


@dataclass(frozen=True)
class RefinementDiagnostics:
    gradient_min: float | None
    gradient_p50: float | None
    gradient_p90: float | None
    gradient_p95: float | None
    gradient_p99: float | None
    gradient_max: float | None
    observation_min: int | None
    observation_p50: float | None
    observation_p90: float | None
    observation_max: int | None
    eligible_count: int
    candidate_limit: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "gradient_min": self.gradient_min,
            "gradient_p50": self.gradient_p50,
            "gradient_p90": self.gradient_p90,
            "gradient_p95": self.gradient_p95,
            "gradient_p99": self.gradient_p99,
            "gradient_max": self.gradient_max,
            "observation_min": self.observation_min,
            "observation_p50": self.observation_p50,
            "observation_p90": self.observation_p90,
            "observation_max": self.observation_max,
            "eligible_count": self.eligible_count,
            "candidate_limit": self.candidate_limit,
        }


@dataclass(frozen=True)
class RasterizationMetadataDiagnostics:
    """Read-only shape audit for gsplat's non-packed refinement metadata."""

    means2d_shape: tuple[int, ...]
    absgrad_shape: tuple[int, ...]
    radii_shape: tuple[int, ...]
    observation_slots: int

    def as_dict(self) -> dict[str, int | list[int]]:
        return {
            "means2d_shape": list(self.means2d_shape),
            "means2d_absgrad_shape": list(self.absgrad_shape),
            "radii_shape": list(self.radii_shape),
            "observation_slots": self.observation_slots,
        }


class RefinementStatistics:
    """Mean screen-space gradient and visibility count per Gaussian."""

    def __init__(self, count: int, device: str | torch.device) -> None:
        self.gradient_sum = torch.zeros(count, dtype=torch.float32, device=device)
        self.observations = torch.zeros(count, dtype=torch.long, device=device)
        self.last_metadata_diagnostics: RasterizationMetadataDiagnostics | None = None

    @property
    def count(self) -> int:
        return len(self.gradient_sum)

    def mean_gradient(self) -> torch.Tensor:
        return self.gradient_sum / self.observations.clamp_min(1)

    def accumulate(self, meta: dict[str, Any]) -> None:
        """Accumulate ``meta['means2d'].absgrad`` only where metadata reports visibility."""
        self.last_metadata_diagnostics = inspect_rasterization_metadata(meta, self.count)
        means2d = meta["means2d"]
        radii = meta["radii"]
        gradients = means2d.absgrad.norm(dim=-1).reshape(-1, self.count)
        visible = (radii.reshape(-1, self.count) > 0)
        self.gradient_sum += (gradients * visible).sum(dim=0)
        self.observations += visible.sum(dim=0).to(self.observations.dtype)

    def reset(self, count: int | None = None) -> None:
        count = self.count if count is None else count
        self.gradient_sum = torch.zeros(count, dtype=torch.float32, device=self.gradient_sum.device)
        self.observations = torch.zeros(count, dtype=torch.long, device=self.observations.device)


def inspect_rasterization_metadata(
    meta: dict[str, Any], gaussian_count: int
) -> RasterizationMetadataDiagnostics:
    """Validate and describe gsplat non-packed metadata without changing accounting."""
    means2d = meta.get("means2d")
    if means2d is None or getattr(means2d, "absgrad", None) is None:
        raise RuntimeError("gsplat metadata must provide means2d.absgrad with absgrad=True")
    radii = meta.get("radii")
    if radii is None:
        raise RuntimeError("gsplat metadata must provide radii for visibility accounting")
    if means2d.shape[-2:] != (gaussian_count, 2):
        raise ValueError(
            f"expected non-packed means2d shape [..., {gaussian_count}, 2], got {tuple(means2d.shape)}"
        )
    if means2d.absgrad.shape != means2d.shape:
        raise ValueError("means2d.absgrad shape must match means2d")
    if radii.shape != means2d.shape[:-1]:
        raise ValueError("radii shape must equal means2d.shape[:-1] in non-packed mode")
    return RasterizationMetadataDiagnostics(
        means2d_shape=tuple(means2d.shape),
        absgrad_shape=tuple(means2d.absgrad.shape),
        radii_shape=tuple(radii.shape),
        observation_slots=radii.numel() // gaussian_count,
    )


def pruning_mask(
    opacities: torch.Tensor, observations: torch.Tensor, *, opacity_threshold: float, min_observations: int
) -> torch.Tensor:
    return (opacities < opacity_threshold) & (observations >= min_observations)


def compute_refinement_diagnostics(
    gradients: torch.Tensor,
    observations: torch.Tensor,
    *,
    eligible_count: int,
    candidate_limit: int,
) -> RefinementDiagnostics:
    """Summarize selection inputs without modifying the selection decision."""
    if len(gradients) != len(observations):
        raise ValueError("gradient/observation count mismatch")
    if not len(gradients):
        return RefinementDiagnostics(
            gradient_min=None, gradient_p50=None, gradient_p90=None, gradient_p95=None,
            gradient_p99=None, gradient_max=None, observation_min=None, observation_p50=None,
            observation_p90=None, observation_max=None, eligible_count=eligible_count,
            candidate_limit=candidate_limit,
        )
    gradient_values = gradients.detach().float()
    observation_values = observations.detach().float()
    return RefinementDiagnostics(
        gradient_min=float(gradient_values.min()),
        gradient_p50=float(torch.quantile(gradient_values, 0.50)),
        gradient_p90=float(torch.quantile(gradient_values, 0.90)),
        gradient_p95=float(torch.quantile(gradient_values, 0.95)),
        gradient_p99=float(torch.quantile(gradient_values, 0.99)),
        gradient_max=float(gradient_values.max()),
        observation_min=int(observations.min()),
        observation_p50=float(torch.quantile(observation_values, 0.50)),
        observation_p90=float(torch.quantile(observation_values, 0.90)),
        observation_max=int(observations.max()),
        eligible_count=eligible_count,
        candidate_limit=candidate_limit,
    )


def select_candidates(
    gaussians: AdaptiveGaussianParameters,
    statistics: RefinementStatistics,
    settings: dict[str, Any],
) -> CandidateSelection:
    """Select bounded candidates using normalized gradients and conservative pruning."""
    if gaussians.count != statistics.count:
        raise ValueError("Gaussian/statistics count mismatch")
    device = gaussians.means.device
    scores = statistics.mean_gradient()
    observations = statistics.observations
    minimum = int(settings["min_gaussian_count"])
    raw_prune = pruning_mask(
        gaussians.opacities().detach(),
        observations,
        opacity_threshold=float(settings["prune_opacity_threshold"]),
        min_observations=int(settings["min_observations"]),
    )
    prune_order = torch.argsort(gaussians.opacities().detach(), stable=True)
    selected_prune: list[int] = []
    for index in prune_order.tolist():
        if raw_prune[index] and gaussians.count - len(selected_prune) > minimum:
            selected_prune.append(index)
    prune_indices = torch.tensor(selected_prune, dtype=torch.long, device=device)

    eligible = (scores >= float(settings["gradient_threshold"])) & (
        observations >= int(settings["min_observations"])
    )
    eligible[prune_indices] = False
    ranked = torch.argsort(scores, descending=True, stable=True)
    candidate_limit = min(
        int(settings["max_candidates"]),
        max(1, int(gaussians.count * float(settings["max_candidate_fraction"]))),
    )
    diagnostics = compute_refinement_diagnostics(
        scores,
        observations,
        eligible_count=int(eligible.sum()),
        candidate_limit=candidate_limit,
    )
    available_growth = max(0, int(settings["max_gaussian_count"]) - (gaussians.count - len(prune_indices)))
    split: list[int] = []
    duplicate: list[int] = []
    scale_threshold = float(settings["split_scale_threshold"])
    for index in ranked.tolist():
        if not eligible[index] or len(split) + len(duplicate) >= candidate_limit or available_growth <= 0:
            continue
        if gaussians.scales().detach()[index].mean() >= scale_threshold:
            split.append(index)
        else:
            duplicate.append(index)
        available_growth -= 1  # Both operations add one net Gaussian.
    return CandidateSelection(
        split_indices=torch.tensor(split, dtype=torch.long, device=device),
        duplicate_indices=torch.tensor(duplicate, dtype=torch.long, device=device),
        prune_indices=prune_indices,
        diagnostics=diagnostics,
    )


def refine(
    gaussians: AdaptiveGaussianParameters, statistics: RefinementStatistics, settings: dict[str, Any], seed: int
) -> tuple[MutationResult, CandidateSelection]:
    selection = select_candidates(gaussians, statistics, settings)
    result = gaussians.mutate(
        selection.split_indices,
        selection.duplicate_indices,
        selection.prune_indices,
        seed=seed,
        split_scale_factor=float(settings["split_scale_factor"]),
        split_jitter_factor=float(settings["split_jitter_factor"]),
        duplicate_jitter_factor=float(settings["duplicate_jitter_factor"]),
    )
    return result, selection
