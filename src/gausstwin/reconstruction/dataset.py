"""COLMAP-style scene access and lazy RGB image loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .camera import CameraView
from .colmap import ColmapModel, SparsePoint, load_colmap_model


@dataclass(frozen=True)
class ColmapScene:
    root: Path
    images_root: Path
    model: ColmapModel

    @property
    def views(self) -> list[CameraView]:
        return self.model.camera_views()

    @property
    def points(self) -> list[SparsePoint]:
        return self.model.points

    def image_path(self, view: CameraView) -> Path:
        return self.images_root / view.image_name

    def load_image(self, view: CameraView, max_dimension: int | None):
        """Return an HWC float32 RGB tensor; Pillow is only required at run time."""
        try:
            from PIL import Image
            import torch
        except ImportError as exc:
            raise RuntimeError("Pillow and PyTorch are required to load training images") from exc
        path = self.image_path(view)
        if not path.is_file():
            raise FileNotFoundError(f"COLMAP image does not exist: {path}")
        with Image.open(path) as image:
            image = image.convert("RGB")
            target = view.intrinsics.scaled_to(max_dimension)
            if image.size != (target.width, target.height):
                image = image.resize((target.width, target.height), Image.Resampling.LANCZOS)
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            return pixels.reshape(target.height, target.width, 3).float().div_(255.0)


def load_colmap_scene(
    scene_root: str | Path, sparse_dir: str = "sparse/0", images_dir: str = "images"
) -> ColmapScene:
    root = Path(scene_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Scene root does not exist: {root}")
    images_root = root / images_dir
    if not images_root.is_dir():
        raise FileNotFoundError(f"COLMAP images directory does not exist: {images_root}")
    return ColmapScene(root=root, images_root=images_root, model=load_colmap_model(root / sparse_dir))
