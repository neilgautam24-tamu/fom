"""Oriented bounding-box geometry helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def points_to_obb_distance(
    points: np.ndarray,
    center: np.ndarray,
    dimensions: np.ndarray,
    wxyz: np.ndarray,
) -> np.ndarray:
    """Signed distance from each point to an OBB (negative = inside)."""
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    dimensions = np.asarray(dimensions, dtype=np.float64)
    wxyz = np.asarray(wxyz, dtype=np.float64)

    rot = Rotation.from_quat(wxyz, scalar_first=True)
    local = rot.inv().apply(points - center)
    half = dimensions * 0.5
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.maximum(q.max(axis=1), 0.0), 0.0)
    return outside + inside


def closest_point_on_obb(
    point: np.ndarray,
    center: np.ndarray,
    dimensions: np.ndarray,
    wxyz: np.ndarray,
) -> np.ndarray:
    """Closest point on the OBB surface (or interior) to ``point``."""
    point = np.asarray(point, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    dimensions = np.asarray(dimensions, dtype=np.float64)
    wxyz = np.asarray(wxyz, dtype=np.float64)

    rot = Rotation.from_quat(wxyz, scalar_first=True)
    local = rot.inv().apply(point - center)
    half = dimensions * 0.5
    clamped = np.clip(local, -half, half)
    return rot.apply(clamped) + center
