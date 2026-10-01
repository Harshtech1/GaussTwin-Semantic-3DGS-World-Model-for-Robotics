"""Append-only metric and structural-event persistence for GaussTwin-002."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    def __init__(self, metrics_root: str | Path) -> None:
        self.root = Path(metrics_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "trajectory.jsonl"
        self.entries = self._read_entries()
        self._iterations = {entry["iteration"] for entry in self.entries}

    def _read_entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def append(self, entry: dict[str, Any]) -> None:
        iteration = entry["iteration"]
        if iteration in self._iterations:
            raise FileExistsError(f"Trajectory already contains iteration {iteration}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        self.entries.append(entry)
        self._iterations.add(iteration)

    def write_evaluation(self, iteration: int, evaluation: dict[str, Any]) -> Path:
        path = self.root / f"evaluation_iter_{iteration:06d}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {path}")
        path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        path = self.root / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path
