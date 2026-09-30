"""Read-only environment inspection with optional PyTorch support."""

from __future__ import annotations

import platform
from typing import Any


def collect_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "pytorch_version": None,
        "cuda_available": False,
        "cuda_runtime_version": None,
        "gpu_count": 0,
        "gpus": [],
        "total_vram_bytes": 0,
    }
    try:
        import torch
    except ImportError:
        return report

    report["pytorch_version"] = torch.__version__
    report["cuda_runtime_version"] = torch.version.cuda
    report["cuda_available"] = bool(torch.cuda.is_available())
    if not report["cuda_available"]:
        return report

    count = torch.cuda.device_count()
    report["gpu_count"] = count
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        allocated = torch.cuda.memory_allocated(index)
        reserved = torch.cuda.memory_reserved(index)
        free_bytes: int | None = None
        total_bytes = properties.total_memory
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except (AttributeError, RuntimeError):
            # Some CUDA runtimes do not expose allocator-independent free memory.
            pass
        report["gpus"].append(
            {
                "index": index,
                "name": properties.name,
                "memory_bytes": total_bytes,
                "memory_gib": round(total_bytes / 1024**3, 2),
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "free_bytes": free_bytes,
            }
        )
        report["total_vram_bytes"] += total_bytes
    return report


def format_environment(report: dict[str, Any]) -> str:
    lines = [
        f"Python version: {report['python_version']}",
        f"PyTorch version: {report['pytorch_version'] or 'not installed'}",
        f"CUDA available: {report['cuda_available']}",
        f"CUDA runtime version: {report['cuda_runtime_version'] or 'not available'}",
        f"GPU count: {report['gpu_count']}",
        f"OS: {report['os']}",
    ]
    if report["gpus"]:
        lines.append(f"Total VRAM across separate devices: {_gib(report['total_vram_bytes']):.2f} GiB")
        for gpu in report["gpus"]:
            free = "unsupported" if gpu["free_bytes"] is None else f"{_gib(gpu['free_bytes']):.2f} GiB"
            lines.append(
                f"GPU {gpu['index']}: {gpu['name']} | total {_gib(gpu['memory_bytes']):.2f} GiB | "
                f"allocated {_gib(gpu['allocated_bytes']):.2f} GiB | "
                f"reserved {_gib(gpu['reserved_bytes']):.2f} GiB | free {free}"
            )
    else:
        lines.append("GPUs: none attached")
    return "\n".join(lines)


def _gib(bytes_value: int) -> float:
    return bytes_value / 1024**3
