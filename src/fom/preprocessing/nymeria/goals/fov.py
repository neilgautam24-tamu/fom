"""Head-camera field-of-view checks for Nymeria Plus objects."""

from __future__ import annotations

import numpy as np

from fom.preprocessing.nymeria.contacts.geometry import closest_point_on_obb
from fom.preprocessing.nymeria.io.objects import box_corners_world


def world_to_camera(
    points_world: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
) -> np.ndarray:
    """Map world points to OpenCV camera coordinates (+Z forward)."""
    points = np.asarray(points_world, dtype=np.float64)
    cam_pos = np.asarray(camera_position, dtype=np.float64)
    rot = np.asarray(camera_rotation, dtype=np.float64)
    return (rot.T @ (points - cam_pos).T).T


def points_in_rgb_frustum(
    points_world: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    *,
    fov_y_rad: float,
    aspect: float,
    margin_rad: float = 0.05,
) -> np.ndarray:
    """Return boolean mask: point projects inside the RGB pinhole frustum."""
    local = world_to_camera(points_world, camera_position, camera_rotation)
    z = local[:, 2]
    visible = z > 1e-3
    half_y = max(float(fov_y_rad) * 0.5 - margin_rad, 0.05)
    half_x = float(np.arctan(np.tan(half_y) * aspect))
    x_over_z = np.zeros_like(z)
    y_over_z = np.zeros_like(z)
    x_over_z[visible] = local[visible, 0] / z[visible]
    y_over_z[visible] = local[visible, 1] / z[visible]
    tan_x = np.tan(half_x)
    tan_y = np.tan(half_y)
    in_frustum = visible & (np.abs(x_over_z) <= tan_x) & (np.abs(y_over_z) <= tan_y)
    return in_frustum


def object_in_head_fov(
    object_center: np.ndarray,
    object_dimensions: np.ndarray,
    object_wxyz: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    *,
    fov_y_rad: float,
    aspect: float,
    margin_rad: float = 0.05,
) -> bool:
    """True if any OBB corner or the center is inside the head RGB frustum."""
    corners = box_corners_world(object_center, object_dimensions, object_wxyz)
    samples = np.vstack([corners, np.asarray(object_center, dtype=np.float64).reshape(1, 3)])
    return bool(points_in_rgb_frustum(
        samples,
        camera_position,
        camera_rotation,
        fov_y_rad=fov_y_rad,
        aspect=aspect,
        margin_rad=margin_rad,
    ).any())


APPROACH_FOV_MARGIN_RAD = 0.12
MIN_APPROACH_VISIBLE_CORNERS = 5


def object_visible_strict(
    obj,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    rgb_calib,
    *,
    fov_margin_rad: float = APPROACH_FOV_MARGIN_RAD,
) -> bool:
    """Stricter than ``object_in_head_fov``: target must be squarely in view, not edge-grazing."""
    cam_pos = np.asarray(camera_position, dtype=np.float64)
    cam_rot = np.asarray(camera_rotation, dtype=np.float64)
    corners = box_corners_world(obj.center_world, obj.dimensions, obj.wxyz)
    visible_corners = int(
        points_in_rgb_frustum(
            corners,
            cam_pos,
            cam_rot,
            fov_y_rad=float(rgb_calib.fov_y_rad),
            aspect=float(rgb_calib.aspect),
            margin_rad=float(fov_margin_rad),
        ).sum()
    )
    center_visible = bool(
        points_in_rgb_frustum(
            np.asarray(obj.center_world, dtype=np.float64).reshape(1, 3),
            cam_pos,
            cam_rot,
            fov_y_rad=float(rgb_calib.fov_y_rad),
            aspect=float(rgb_calib.aspect),
            margin_rad=float(fov_margin_rad),
        )[0]
    )
    closest = closest_point_on_obb(cam_pos, obj.center_world, obj.dimensions, obj.wxyz)
    closest_visible = bool(
        points_in_rgb_frustum(
            closest.reshape(1, 3),
            cam_pos,
            cam_rot,
            fov_y_rad=float(rgb_calib.fov_y_rad),
            aspect=float(rgb_calib.aspect),
            margin_rad=float(fov_margin_rad),
        )[0]
    )
    body_visible = center_visible or visible_corners >= MIN_APPROACH_VISIBLE_CORNERS
    return bool(body_visible and closest_visible)
