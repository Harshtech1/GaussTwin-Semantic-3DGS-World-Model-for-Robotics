"""Minimal binary COLMAP sparse-model reader without a COLMAP runtime dependency."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .camera import CameraIntrinsics, CameraView, world_to_camera_matrix

_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    intrinsics: CameraIntrinsics


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    camera_id: int
    name: str
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]


@dataclass(frozen=True)
class SparsePoint:
    point_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float


@dataclass(frozen=True)
class ColmapModel:
    cameras: dict[int, ColmapCamera]
    images: list[ColmapImage]
    points: list[SparsePoint]

    def camera_views(self) -> list[CameraView]:
        views: list[CameraView] = []
        for image in self.images:
            camera = self.cameras.get(image.camera_id)
            if camera is None:
                raise ValueError(f"Image {image.name} references missing camera {image.camera_id}")
            views.append(
                CameraView(
                    image_id=image.image_id,
                    image_name=image.name,
                    intrinsics=camera.intrinsics,
                    world_to_camera=world_to_camera_matrix(image.qvec, image.tvec),
                )
            )
        return views


def _read(handle, format_string: str):
    size = struct.calcsize("<" + format_string)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Unexpected end of COLMAP binary model")
    return struct.unpack("<" + format_string, data)


def _read_c_string(handle) -> str:
    data = bytearray()
    while True:
        byte = handle.read(1)
        if not byte:
            raise ValueError("Unexpected end of COLMAP image name")
        if byte == b"\x00":
            return data.decode("utf-8")
        data.extend(byte)


def _to_intrinsics(model: str, width: int, height: int, params: tuple[float, ...]) -> CameraIntrinsics:
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")
    return CameraIntrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)


def read_cameras_binary(path: str | Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with Path(path).open("rb") as handle:
        (count,) = _read(handle, "Q")
        for _ in range(count):
            camera_id, model_id, width, height = _read(handle, "IiQQ")
            if model_id not in _CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            model, parameter_count = _CAMERA_MODELS[model_id]
            params = _read(handle, "d" * parameter_count)
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model,
                intrinsics=_to_intrinsics(model, width, height, params),
            )
    return cameras


def read_images_binary(path: str | Path) -> list[ColmapImage]:
    images: list[ColmapImage] = []
    with Path(path).open("rb") as handle:
        (count,) = _read(handle, "Q")
        for _ in range(count):
            values = _read(handle, "IdddddddI")
            image_id = values[0]
            qvec = tuple(values[1:5])
            tvec = tuple(values[5:8])
            camera_id = values[8]
            name = _read_c_string(handle)
            (points2d_count,) = _read(handle, "Q")
            handle.seek(points2d_count * struct.calcsize("<ddq"), 1)
            images.append(ColmapImage(image_id, camera_id, name, qvec, tvec))
    return images


def read_points3d_binary(path: str | Path) -> list[SparsePoint]:
    points: list[SparsePoint] = []
    with Path(path).open("rb") as handle:
        (count,) = _read(handle, "Q")
        for _ in range(count):
            point_id, x, y, z, red, green, blue, error = _read(handle, "QdddBBBd")
            (track_length,) = _read(handle, "Q")
            handle.seek(track_length * struct.calcsize("<II"), 1)
            points.append(SparsePoint(point_id, (x, y, z), (red, green, blue), error))
    return points


def load_colmap_model(sparse_root: str | Path) -> ColmapModel:
    sparse_path = Path(sparse_root)
    required = [sparse_path / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing COLMAP sparse files: " + ", ".join(missing))
    return ColmapModel(
        cameras=read_cameras_binary(sparse_path / "cameras.bin"),
        images=read_images_binary(sparse_path / "images.bin"),
        points=read_points3d_binary(sparse_path / "points3D.bin"),
    )
