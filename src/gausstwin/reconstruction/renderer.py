"""Single-camera gsplat rendering interface."""

from __future__ import annotations

import torch

from gausstwin.gaussian.parameters import GaussianParameters

from .camera import CameraView


def render_view(gaussians: GaussianParameters, view: CameraView, device: str | torch.device):
    """Render RGB and alpha for exactly one camera, leaving batching to future work."""
    try:
        from gsplat import rasterization
    except ImportError as exc:
        raise RuntimeError("gsplat is required for rendering; install Kaggle GPU requirements") from exc
    camera = view.scaled_to(None)
    viewmat = torch.tensor(camera.world_to_camera, dtype=torch.float32, device=device).unsqueeze(0)
    intrinsics = torch.tensor(camera.intrinsics.matrix(), dtype=torch.float32, device=device).unsqueeze(0)
    renders, alphas, _ = rasterization(
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
    )
    return renders[0], alphas[0]
