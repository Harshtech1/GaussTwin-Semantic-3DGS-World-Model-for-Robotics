#!/usr/bin/env python3
"""Report actual Kaggle CUDA devices and their per-device memory state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gausstwin.config import load_config  # noqa: E402
from gausstwin.utils.environment import collect_environment, format_environment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/kaggle_t4.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    report = collect_environment()
    print(format_environment(report))
    print(f"Configured backend: {config['runtime']['backend']}")
    print(f"Configured device: {config['runtime']['device']}")
    print(f"Expected GPU count: {config['runtime']['expected_gpu_count']}")
    print("VRAM totals are sums for reporting only; devices remain separate CUDA devices.")
    if report["gpu_count"] != config["runtime"]["expected_gpu_count"]:
        print("Note: current host does not match the configured two-GPU Kaggle target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
