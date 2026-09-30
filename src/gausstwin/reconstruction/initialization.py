"""Sparse COLMAP point-cloud initialization for GaussianParameters."""

from __future__ import annotations

from gausstwin.gaussian.parameters import GaussianParameters

from .dataset import ColmapScene


def initialize_gaussians(scene: ColmapScene, config: dict, device: str) -> GaussianParameters:
    reconstruction = config["reconstruction"]
    return GaussianParameters.from_sparse_points(
        scene.points,
        device=device,
        max_points=reconstruction["max_points"],
        initial_scale=reconstruction.get("initial_scale"),
        initial_opacity=reconstruction["initial_opacity"],
    )
