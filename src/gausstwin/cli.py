"""Shared behavior for planned pipeline entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from gausstwin.config import ConfigError, load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def planned_stage(stage: str, default_config: str) -> int:
    parser = argparse.ArgumentParser(description=f"GaussTwin {stage} stage (planned)")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / default_config)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))
    runtime = config["runtime"]
    print(f"Validated {stage} configuration: {args.config}")
    print(f"Runtime target: {runtime['backend']} / {runtime['device']}")
    print(f"{stage} is planned and no compute was started.")
    return 0
