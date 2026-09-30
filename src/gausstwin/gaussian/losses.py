"""Small, dependency-free photometric losses and baseline metrics."""

from __future__ import annotations

import torch
from torch.nn import functional as functional


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = functional.mse_loss(prediction, target).clamp_min(1e-10)
    return -10.0 * torch.log10(mse)


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """A compact global-window SSIM for baseline monitoring, not an LPIPS substitute."""
    prediction = prediction.permute(2, 0, 1).unsqueeze(0)
    target = target.permute(2, 0, 1).unsqueeze(0)
    mu_x = functional.avg_pool2d(prediction, kernel_size=11, stride=1, padding=5)
    mu_y = functional.avg_pool2d(target, kernel_size=11, stride=1, padding=5)
    sigma_x = functional.avg_pool2d(prediction * prediction, 11, 1, 5) - mu_x.square()
    sigma_y = functional.avg_pool2d(target * target, 11, 1, 5) - mu_y.square()
    sigma_xy = functional.avg_pool2d(prediction * target, 11, 1, 5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean()


def photometric_loss(
    prediction: torch.Tensor, target: torch.Tensor, l1_weight: float, ssim_weight: float
) -> torch.Tensor:
    return l1_weight * functional.l1_loss(prediction, target) + ssim_weight * (1.0 - ssim(prediction, target))
