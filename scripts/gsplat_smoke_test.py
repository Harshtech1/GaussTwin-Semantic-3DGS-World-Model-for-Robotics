#!/usr/bin/env python3
"""Rasterize one tiny synthetic Gaussian scene on one selected CUDA device."""

from __future__ import annotations

import argparse
import sys


def _gib(bytes_value: int) -> float:
    return bytes_value / 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0 or cuda:1")
    args = parser.parse_args()
    try:
        import torch
        from gsplat import rasterization

        device = torch.device(args.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("--device must name one CUDA device, for example cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"{device} is unavailable; visible GPU count is {torch.cuda.device_count()}")

        width = height = 64
        # A single, small Gaussian in front of an identity camera. This is not training.
        means = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=device)
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        scales = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32, device=device)
        opacities = torch.tensor([0.9], dtype=torch.float32, device=device)
        colors = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        viewmats = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
        intrinsics = torch.tensor(
            [[[50.0, 0.0, width / 2], [0.0, 50.0, height / 2], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
            device=device,
        )
        renders, alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=intrinsics,
            width=width,
            height=height,
            packed=False,
        )
        torch.cuda.synchronize(device)
        print(f"render tensor shape: {list(renders.shape)}")
        print(f"alpha tensor shape: {list(alphas.shape)}")
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
        print(f"Allocated VRAM: {_gib(torch.cuda.memory_allocated(device)):.4f} GiB")
        return 0
    except Exception as exc:
        print(f"gsplat smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
