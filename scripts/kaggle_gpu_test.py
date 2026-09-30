#!/usr/bin/env python3
"""Run a tiny, per-device CUDA smoke test suitable for a Kaggle T4 session."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gausstwin.utils.environment import collect_environment, format_environment  # noqa: E402


def main() -> int:
    report = collect_environment()
    print(format_environment(report))
    if not report["cuda_available"]:
        print("CUDA is unavailable; no GPU tensor operation was attempted.")
        return 0

    import torch

    for index in range(report["gpu_count"]):
        device = torch.device(f"cuda:{index}")
        # Four float32 values: intentionally tiny and allocated on one device only.
        result = (torch.arange(4, dtype=torch.float32, device=device) + 1).sum()
        torch.cuda.synchronize(index)
        print(f"GPU {index} smoke test passed: sum={result.item():.1f} on {device}")
    print("Each test uses an independent CUDA device; GPU memory is not combined.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
