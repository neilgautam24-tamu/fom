"""Detect walking-towards-object motion from SMPL-X pelvis trajectories."""

from __future__ import annotations

import numpy as np


def pelvis_velocity(
    transl: np.ndarray,
    timestamps_us: np.ndarray,
    frame_idx: int,
    *,
    window: int = 3,
) -> np.ndarray:
    """Central-difference pelvis velocity (m/s) at ``frame_idx``."""
    n = transl.shape[0]
    i0 = max(0, frame_idx - window)
    i1 = min(n - 1, frame_idx + window)
    if i1 <= i0:
        return np.zeros(3, dtype=np.float64)
    dt = (float(timestamps_us[i1]) - float(timestamps_us[i0])) * 1e-6
    if dt <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    return (transl[i1] - transl[i0]).astype(np.float64) / dt


def direction_to_point(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float64)
    return delta / norm


def approach_alignment(
    pelvis_pos: np.ndarray,
    object_center: np.ndarray,
    velocity: np.ndarray,
) -> tuple[float, float]:
    """Return ``(speed_m_s, cos_alignment)`` towards ``object_center``."""
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-6:
        return 0.0, 0.0
    to_obj = direction_to_point(pelvis_pos, object_center)
    if float(np.linalg.norm(to_obj)) < 1e-6:
        return speed, 0.0
    cos_align = float(np.dot(velocity / speed, to_obj))
    return speed, cos_align


def validate_approach_segment(
    transl: np.ndarray,
    timestamps_us: np.ndarray,
    object_center: np.ndarray,
    start_frame: int,
    end_frame: int,
    *,
    min_speed_m_s: float = 0.08,
    min_alignment: float = 0.35,
    min_valid_fraction: float = 0.45,
    vel_window: int = 3,
) -> tuple[bool, float, float]:
    """Check that pelvis motion between frames walks towards the object."""
    if end_frame <= start_frame:
        return False, 0.0, 0.0

    speeds: list[float] = []
    aligns: list[float] = []
    valid = 0
    total = end_frame - start_frame + 1
    for frame_idx in range(start_frame, end_frame + 1):
        vel = pelvis_velocity(transl, timestamps_us, frame_idx, window=vel_window)
        speed, cos_align = approach_alignment(transl[frame_idx], object_center, vel)
        speeds.append(speed)
        aligns.append(cos_align)
        if speed >= min_speed_m_s and cos_align >= min_alignment:
            valid += 1

    fraction = valid / max(total, 1)
    ok = fraction >= min_valid_fraction and np.median(speeds) >= min_speed_m_s * 0.5
    return ok, float(np.median(speeds)), float(np.median(aligns))
