"""Small YAML configuration loader with deterministic inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is missing or invalid."""


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, resolving an optional relative ``defaults`` parent file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration does not exist: {config_path}")
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(parsed, dict):
        raise ConfigError("Configuration root must be a mapping")
    parent = parsed.pop("defaults", None)
    config = load_config(config_path.parent / parent) if parent else {}
    config = _merge(config, parsed)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ConfigError("Missing runtime configuration")
    if runtime.get("device") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("runtime.device must be auto, cpu, or cuda")
    count = runtime.get("expected_gpu_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ConfigError("runtime.expected_gpu_count must be a non-negative integer")
    if not isinstance(runtime.get("backend"), str):
        raise ConfigError("runtime.backend must be a string")
