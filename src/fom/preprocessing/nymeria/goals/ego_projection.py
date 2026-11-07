"""Project 3D world points onto Aria head RGB (fisheye) images."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from fom.preprocessing.nymeria.goals.fov import points_in_rgb_frustum, world_to_camera


@dataclass(frozen=True)
class ProjectedGoal:
    u: float
    v: float
    visible: bool


class AriaFisheyeProjector:
    """Fisheye projection using the camera model embedded in ``data.vrs``."""

    def __init__(self, data_vrs: Path):
        from projectaria_tools.core import data_provider

        if not data_vrs.is_file():
            raise FileNotFoundError(f"Missing RGB VRS: {data_vrs}")
        provider = data_provider.create_vrs_data_provider(str(data_vrs))
        if provider is None:
            raise RuntimeError(f"Could not open RGB VRS: {data_vrs}")
        self._cam = provider.get_device_calibration().get_camera_calib("camera-rgb")
        self.full_width, self.full_height = (int(x) for x in self._cam.get_image_size())

    @property
    def image_size(self) -> tuple[int, int]:
        return self.full_width, self.full_height

    def goal_in_frustum(
        self,
        goal_world: np.ndarray,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        *,
        fov_y_rad: float,
        aspect: float,
        margin_rad: float = 0.05,
    ) -> bool:
        point = np.asarray(goal_world, dtype=np.float64).reshape(1, 3)
        return bool(
            points_in_rgb_frustum(
                point,
                camera_position,
                camera_rotation,
                fov_y_rad=fov_y_rad,
                aspect=aspect,
                margin_rad=margin_rad,
            )[0]
        )

    def project_world(
        self,
        goal_world: np.ndarray,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        *,
        fov_y_rad: float,
        aspect: float,
        margin_rad: float = 0.05,
    ) -> ProjectedGoal:
        """Project a world-space goal point to full-resolution RGB pixel coordinates."""
        point = np.asarray(goal_world, dtype=np.float64).reshape(1, 3)
        visible = self.goal_in_frustum(
            point[0],
            camera_position,
            camera_rotation,
            fov_y_rad=fov_y_rad,
            aspect=aspect,
            margin_rad=margin_rad,
        )
        if not visible:
            return ProjectedGoal(u=-1.0, v=-1.0, visible=False)

        p_cam = world_to_camera(point, camera_position, camera_rotation)[0]
        if p_cam[2] <= 1e-3:
            return ProjectedGoal(u=-1.0, v=-1.0, visible=False)

        uv = self._cam.project(p_cam.reshape(3, 1))
        if uv is None:
            return ProjectedGoal(u=-1.0, v=-1.0, visible=False)
        uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1)
        if uv_arr.size < 2:
            return ProjectedGoal(u=-1.0, v=-1.0, visible=False)
        u, v = float(uv_arr[0]), float(uv_arr[1])
        if not (0.0 <= u < self.full_width and 0.0 <= v < self.full_height):
            return ProjectedGoal(u=-1.0, v=-1.0, visible=False)
        return ProjectedGoal(u=u, v=v, visible=True)

    def scale_uv_to_image(
        self,
        u: float,
        v: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        sx = float(image_width) / float(self.full_width)
        sy = float(image_height) / float(self.full_height)
        return u * sx, v * sy


def draw_goal_dot(
    rgb: np.ndarray,
    u: float,
    v: float,
    *,
    radius: int = 11,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Draw a filled goal marker on an RGB image (returns a copy)."""
    out = np.asarray(rgb, dtype=np.uint8).copy()
    if u < 0.0 or v < 0.0:
        return out
    ui, vi = int(round(u)), int(round(v))
    h, w = out.shape[:2]
    r2 = int(radius) ** 2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > r2:
                continue
            x, y = ui + dx, vi + dy
            if 0 <= x < w and 0 <= y < h:
                out[y, x] = color
    return out


def goal_mask_image(
    height: int,
    width: int,
    u: float,
    v: float,
    *,
    radius: int = 11,
) -> np.ndarray:
    """Return a single-channel mask with a white goal point on black."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if u < 0.0 or v < 0.0:
        return mask
    ui, vi = int(round(u)), int(round(v))
    r2 = int(radius) ** 2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > r2:
                continue
            x, y = ui + dx, vi + dy
            if 0 <= x < width and 0 <= y < height:
                mask[y, x] = 255
    return mask


@lru_cache(maxsize=4)
def load_fisheye_projector(data_vrs: str) -> AriaFisheyeProjector:
    return AriaFisheyeProjector(Path(data_vrs))
