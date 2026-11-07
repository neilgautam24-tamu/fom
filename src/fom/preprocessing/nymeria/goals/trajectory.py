"""Precompute Aria head trajectory samples aligned to body motion frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from fom.preprocessing.nymeria.io.aria_slam import (
    ClosedLoopTrajectory,
    RgbCameraCalibration,
    body_timestamp_to_device_us,
    camera_pose_world,
)


@dataclass
class HeadTrajectorySamples:
    body_timestamp_us: np.ndarray
    device_timestamp_us: np.ndarray
    device_position: np.ndarray
    device_quat_wxyz: np.ndarray
    device_rotation: np.ndarray
    camera_position: np.ndarray
    camera_quat_wxyz: np.ndarray
    camera_rotation: np.ndarray
    t_device_camera: np.ndarray


def precompute_head_trajectory(
    body_timestamps_us: np.ndarray,
    trajectory: ClosedLoopTrajectory,
    rgb_calib: RgbCameraCalibration,
    device_offset_us: int,
) -> HeadTrajectorySamples:
    """Sample device + RGB camera pose at each body motion timestamp."""
    ts = np.asarray(body_timestamps_us, dtype=np.int64)
    n = ts.shape[0]
    device_us = np.array(
        [body_timestamp_to_device_us(int(t), device_offset_us) for t in ts],
        dtype=np.int64,
    )

    device_pos = np.empty((n, 3), dtype=np.float64)
    device_quat = np.empty((n, 4), dtype=np.float64)
    device_rot = np.empty((n, 3, 3), dtype=np.float64)
    camera_pos = np.empty((n, 3), dtype=np.float64)
    camera_quat = np.empty((n, 4), dtype=np.float64)
    camera_rot = np.empty((n, 3, 3), dtype=np.float64)

    for i, dev_t in enumerate(device_us):
        dpos, drot = trajectory.pose_at_us(int(dev_t))
        device_pos[i] = dpos
        device_quat[i] = Rotation.from_matrix(drot).as_quat(scalar_first=True)
        device_rot[i] = drot
        cpos, crot = camera_pose_world(
            trajectory, rgb_calib.t_device_camera, int(dev_t)
        )
        camera_pos[i] = cpos
        camera_quat[i] = Rotation.from_matrix(crot).as_quat(scalar_first=True)
        camera_rot[i] = crot

    return HeadTrajectorySamples(
        body_timestamp_us=ts,
        device_timestamp_us=device_us,
        device_position=device_pos.astype(np.float32),
        device_quat_wxyz=device_quat.astype(np.float32),
        device_rotation=device_rot.astype(np.float32),
        camera_position=camera_pos.astype(np.float32),
        camera_quat_wxyz=camera_quat.astype(np.float32),
        camera_rotation=camera_rot.astype(np.float32),
        t_device_camera=rgb_calib.t_device_camera.astype(np.float32),
    )


def slice_trajectory(traj: HeadTrajectorySamples, start: int, end: int) -> dict[str, np.ndarray]:
    start = int(start)
    end = int(end)
    sl = slice(start, end + 1)
    return {
        "body_timestamp_us": traj.body_timestamp_us[sl].astype(np.int64),
        "device_timestamp_us": traj.device_timestamp_us[sl].astype(np.int64),
        "device_position": traj.device_position[sl].astype(np.float32),
        "device_quat_wxyz": traj.device_quat_wxyz[sl].astype(np.float32),
        "device_rotation": traj.device_rotation[sl].astype(np.float32),
        "camera_position": traj.camera_position[sl].astype(np.float32),
        "camera_quat_wxyz": traj.camera_quat_wxyz[sl].astype(np.float32),
        "camera_rotation": traj.camera_rotation[sl].astype(np.float32),
        "t_device_camera": traj.t_device_camera.astype(np.float32),
        "rgb_fov_y_rad": np.array([0.0], dtype=np.float32),  # filled by exporter
        "rgb_aspect": np.array([0.0], dtype=np.float32),
    }


def head_trajectory_cache_path(episode_dir: Path, stride: int) -> Path:
    return episode_dir / ".cache" / f"head_trajectory_stride{stride}.npz"


def save_head_trajectory_cache(path: Path, traj: HeadTrajectorySamples) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        body_timestamp_us=traj.body_timestamp_us,
        device_timestamp_us=traj.device_timestamp_us,
        device_position=traj.device_position,
        device_quat_wxyz=traj.device_quat_wxyz,
        device_rotation=traj.device_rotation,
        camera_position=traj.camera_position,
        camera_quat_wxyz=traj.camera_quat_wxyz,
        camera_rotation=traj.camera_rotation,
        t_device_camera=traj.t_device_camera,
    )


def load_head_trajectory_cache(path: Path) -> HeadTrajectorySamples | None:
    if not path.is_file():
        return None
    data = np.load(path)
    return HeadTrajectorySamples(
        body_timestamp_us=np.asarray(data["body_timestamp_us"], dtype=np.int64),
        device_timestamp_us=np.asarray(data["device_timestamp_us"], dtype=np.int64),
        device_position=np.asarray(data["device_position"], dtype=np.float32),
        device_quat_wxyz=np.asarray(data["device_quat_wxyz"], dtype=np.float32),
        device_rotation=np.asarray(data["device_rotation"], dtype=np.float32),
        camera_position=np.asarray(data["camera_position"], dtype=np.float32),
        camera_quat_wxyz=np.asarray(data["camera_quat_wxyz"], dtype=np.float32),
        camera_rotation=np.asarray(data["camera_rotation"], dtype=np.float32),
        t_device_camera=np.asarray(data["t_device_camera"], dtype=np.float32),
    )


def load_or_precompute_head_trajectory(
    cache_path: Path,
    body_timestamps_us: np.ndarray,
    trajectory: ClosedLoopTrajectory,
    rgb_calib: RgbCameraCalibration,
    device_offset_us: int,
    *,
    refresh: bool = False,
) -> HeadTrajectorySamples:
    if not refresh:
        cached = load_head_trajectory_cache(cache_path)
        if cached is not None and cached.body_timestamp_us.shape[0] == len(body_timestamps_us):
            return cached
    traj = precompute_head_trajectory(
        body_timestamps_us, trajectory, rgb_calib, device_offset_us
    )
    save_head_trajectory_cache(cache_path, traj)
    return traj
