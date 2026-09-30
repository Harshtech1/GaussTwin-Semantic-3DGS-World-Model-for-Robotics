#!/usr/bin/env python3
"""Report the installed gsplat and CUDA environment without running rendering."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gausstwin.utils.environment import collect_environment  # noqa: E402


def _gib(bytes_value: int) -> float:
    return bytes_value / 1024**3


def _gsplat_version() -> str:
    try:
        return version("gsplat")
    except PackageNotFoundError:
        return "not installed"


def main() -> int:
    report = collect_environment()
    print(f"Python version: {report['python_version']}")
    print(f"PyTorch version: {report['pytorch_version'] or 'not installed'}")
    print(f"PyTorch CUDA runtime: {report['cuda_runtime_version'] or 'not available'}")
    print(f"gsplat version: {_gsplat_version()}")
    print(f"CUDA available: {report['cuda_available']}")
    print(f"GPU count: {report['gpu_count']}")
    print(f"TOTAL VRAM ACROSS DEVICES: {_gib(report['total_vram_bytes']):.2f} GiB")
    print("PER-DEVICE VRAM:")
    if not report["gpus"]:
        print("  No CUDA devices attached")
    for gpu in report["gpus"]:
        free = "unsupported" if gpu["free_bytes"] is None else f"{_gib(gpu['free_bytes']):.2f} GiB"
        print(
            f"  GPU {gpu['index']}: {gpu['name']} | total {_gib(gpu['memory_bytes']):.2f} GiB | "
            f"allocated {_gib(gpu['allocated_bytes']):.2f} GiB | "
            f"reserved {_gib(gpu['reserved_bytes']):.2f} GiB | free {free}"
        )
    print("The total is a reporting sum only; CUDA memory remains per-device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
