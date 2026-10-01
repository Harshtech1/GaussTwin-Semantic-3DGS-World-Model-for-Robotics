"""gsplat 1.5.3 adaptive rendering boundary with explicit Gaussian indexing."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gausstwin.gaussian.adaptive import AdaptiveGaussianParameters

from .camera import CameraView


@dataclass(frozen=True)
class AdaptiveRenderResult:
    rgb: torch.Tensor
    alpha: torch.Tensor
    meta: dict


def render_adaptive_view(
    gaussians: AdaptiveGaussianParameters,
    view: CameraView,
    device: str | torch.device,
    *,
    absgrad: bool = True,
) -> AdaptiveRenderResult:
    """Render one view with gsplat metadata required by adaptive refinement."""
    try:
        from gsplat import rasterization
    except ImportError as exc:
        raise RuntimeError("gsplat==1.5.3 is required for GaussTwin-002 rendering") from exc
    camera = view.scaled_to(None)
    viewmat = torch.tensor(camera.world_to_camera, dtype=torch.float32, device=device).unsqueeze(0)
    intrinsics = torch.tensor(camera.intrinsics.matrix(), dtype=torch.float32, device=device).unsqueeze(0)
    renders, alphas, meta = rasterization(
        means=gaussians.means,
        quats=gaussians.normalized_quaternions(),
        scales=gaussians.scales(),
        opacities=gaussians.opacities(),
        colors=gaussians.colors(),
        viewmats=viewmat,
        Ks=intrinsics,
        width=camera.intrinsics.width,
        height=camera.intrinsics.height,
        packed=False,
        absgrad=absgrad,
    )
    return AdaptiveRenderResult(rgb=renders[0], alpha=alphas[0], meta=meta)
