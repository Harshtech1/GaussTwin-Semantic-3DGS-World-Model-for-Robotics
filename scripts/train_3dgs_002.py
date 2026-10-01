#!/usr/bin/env python3
"""Train the GaussTwin-002 single-GPU adaptive 3DGS baseline."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from gausstwin.config import load_config
from gausstwin.reconstruction.adaptive_trainer import AdaptiveThreeDGSTrainer
from gausstwin.reconstruction.dataset import load_colmap_scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reconstruction_002.yaml")
    parser.add_argument("--scene-root")
    parser.add_argument("--device", default=None, help="One CUDA device; default is cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    reconstruction = config["reconstruction"]
    scene_root = args.scene_root or reconstruction["scene_root"]
    if not scene_root:
        parser.error("set reconstruction.scene_root or provide --scene-root")
    scene = load_colmap_scene(scene_root, reconstruction["sparse_dir"], reconstruction["images_dir"])
    summary = AdaptiveThreeDGSTrainer(scene, config, device=args.device).train()
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
