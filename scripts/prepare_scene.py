#!/usr/bin/env python3
"""Validate a COLMAP-style scene without requiring COLMAP to be installed."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from gausstwin.config import load_config
from gausstwin.reconstruction.dataset import load_colmap_scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reconstruction.yaml")
    parser.add_argument("--scene-root", required=False)
    args = parser.parse_args()
    config = load_config(args.config)
    reconstruction = config["reconstruction"]
    scene_root = args.scene_root or reconstruction["scene_root"]
    if not scene_root:
        parser.error("set reconstruction.scene_root or provide --scene-root")
    scene = load_colmap_scene(scene_root, reconstruction["sparse_dir"], reconstruction["images_dir"])
    missing_images = [view.image_name for view in scene.views if not scene.image_path(view).is_file()]
    print(f"Scene root: {scene.root}")
    print(f"Cameras: {len(scene.model.cameras)}")
    print(f"Images: {len(scene.views)}")
    print(f"Sparse points: {len(scene.points)}")
    print(f"Missing image files: {len(missing_images)}")
    if missing_images:
        print("First missing image:", missing_images[0])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
