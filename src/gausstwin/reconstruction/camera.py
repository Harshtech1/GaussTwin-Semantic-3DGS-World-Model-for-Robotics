"""Camera primitives shared by COLMAP loading, rendering, and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixel coordinates."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def scaled_to(self, max_dimension: int | None) -> "CameraIntrinsics":
        """Scale dimensions and focal lengths without enlarging an image."""
        if max_dimension is None or max(self.width, self.height) <= max_dimension:
            return self
        scale = max_dimension / max(self.width, self.height)
        return CameraIntrinsics(
            width=round(self.width * scale),
            height=round(self.height * scale),
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
        )

    def matrix(self) -> tuple[tuple[float, float, float], ...]:
        return ((self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0))


@dataclass(frozen=True)
class CameraView:
    """One calibrated image, with COLMAP's world-to-camera transform."""

    image_id: int
    image_name: str
    intrinsics: CameraIntrinsics
    world_to_camera: tuple[tuple[float, float, float, float], ...]

    def scaled_to(self, max_dimension: int | None) -> "CameraView":
        return CameraView(
            image_id=self.image_id,
            image_name=self.image_name,
            intrinsics=self.intrinsics.scaled_to(max_dimension),
            world_to_camera=self.world_to_camera,
        )


def quaternion_to_rotation(qvec: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """Convert COLMAP's normalized ``(qw, qx, qy, qz)`` quaternion to rotation."""
    qw, qx, qy, qz = qvec
    norm = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    if norm == 0:
        raise ValueError("Camera quaternion must be non-zero")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )


def world_to_camera_matrix(
    qvec: Sequence[float], tvec: Sequence[float]
) -> tuple[tuple[float, float, float, float], ...]:
    rotation = quaternion_to_rotation(qvec)
    return tuple(
        tuple(rotation[row]) + (float(tvec[row]),) for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
