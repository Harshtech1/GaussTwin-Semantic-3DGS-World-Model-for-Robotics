#!/usr/bin/env python3
"""Evaluate PSNR/SSIM for a GaussTwin-001 checkpoint on one CUDA device."""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from gausstwin.config import load_config
from gausstwin.reconstruction.dataset import load_colmap_scene
from gausstwin.reconstruction.trainer import ThreeDGSTrainer, load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/reconstruction.yaml")
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    reconstruction = config["reconstruction"]
    scene = load_colmap_scene(args.scene_root, reconstruction["sparse_dir"], reconstruction["images_dir"])
    trainer = ThreeDGSTrainer(scene, config, device=args.device)
    load_checkpoint(args.checkpoint, trainer.gaussians, trainer.optimizer)
    metrics = trainer.evaluate()
    path = trainer.paths.metrics / "evaluation.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
