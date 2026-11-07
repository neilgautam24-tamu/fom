"""Project contact goals onto exported ego RGB frames."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from fom.preprocessing.nymeria.contacts.detection import ContactEvent
from fom.preprocessing.nymeria.goals.ego_projection import (
    AriaFisheyeProjector,
    draw_goal_dot,
    goal_mask_image,
)
from fom.preprocessing.nymeria.goals.visible_goal_sequences import first_contact_goal
from fom.preprocessing.nymeria.io.aria_slam import RgbCameraCalibration


def rotate_image_uv_cw90(u: float, v: float, width: int, height: int) -> tuple[float, float]:
    """Map pixel coords from a pre-rotation image to the CW-rotated image."""
    if u < 0.0 or v < 0.0:
        return -1.0, -1.0
    return float(height - 1.0 - v), float(u)


def load_contact_events_csv(path: Path) -> list[ContactEvent]:
    events: list[ContactEvent] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            events.append(
                ContactEvent(
                    frame_idx=int(row["frame_idx"]),
                    timestamp_ns=int(row["timestamp_ns"]),
                    time_s=float(row["time_s"]),
                    hand=str(row["hand"]),
                    object_uid=int(row["object_uid"]),
                    category=str(row["category"]),
                    instance_name=str(row["instance_name"]),
                    distance_m=float(row["distance_m"]),
                    hand_point_x=float(row["hand_point_x"]),
                    hand_point_y=float(row["hand_point_y"]),
                    hand_point_z=float(row["hand_point_z"]),
                    object_point_x=float(row["object_point_x"]),
                    object_point_y=float(row["object_point_y"]),
                    object_point_z=float(row["object_point_z"]),
                    sample_idx=int(row["sample_idx"]),
                )
            )
    return events


def contact_goal_world(
    event_index: dict[tuple[int, int], list[ContactEvent]],
    *,
    object_uid: int,
    contact_start_frame: int,
    contact_end_frame: int | None = None,
) -> np.ndarray | None:
    """Return the first-contact surface point for one object episode."""
    end = int(contact_end_frame) if contact_end_frame is not None else int(contact_start_frame)
    for frame_idx in range(int(contact_start_frame), end + 1):
        info = first_contact_goal(
            event_index,
            object_uid=int(object_uid),
            contact_start_frame=int(frame_idx),
        )
        if info is not None:
            return np.asarray(info[0], dtype=np.float64)
    return None


def annotate_rgb_sequence(
    rgb: np.ndarray,
    frame_indices: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    goal_world: np.ndarray,
    projector: AriaFisheyeProjector,
    rgb_calib: RgbCameraCalibration,
    *,
    fov_margin_rad: float = 0.05,
    dot_radius: int = 11,
    mask_radius: int = 11,
) -> dict[str, np.ndarray]:
    """Overlay a fixed world-space contact goal on rotated bundle RGB frames."""
    n = int(rgb.shape[0])
    h_rot, w_rot = rgb.shape[1:3]
    w_pre, h_pre = int(w_rot), int(h_rot)

    goal_uv = np.full((n, 2), -1.0, dtype=np.float32)
    goal_visible = np.zeros(n, dtype=np.int8)
    rgb_goal_frames: list[np.ndarray] = []
    mask_frames: list[np.ndarray] = []

    goal = np.asarray(goal_world, dtype=np.float64)
    for slot, local_idx in enumerate(frame_indices.astype(np.int64)):
        frame = np.asarray(rgb[slot], dtype=np.uint8)
        proj = projector.project_world(
            goal,
            camera_position[int(local_idx)],
            camera_rotation[int(local_idx)],
            fov_y_rad=rgb_calib.fov_y_rad,
            aspect=rgb_calib.aspect,
            margin_rad=fov_margin_rad,
        )
        if proj.visible:
            su, sv = projector.scale_uv_to_image(proj.u, proj.v, w_pre, h_pre)
            su, sv = rotate_image_uv_cw90(su, sv, w_pre, h_pre)
            if 0.0 <= su < w_rot and 0.0 <= sv < h_rot:
                goal_uv[slot] = (su, sv)
                goal_visible[slot] = 1
                rgb_goal_frames.append(draw_goal_dot(frame, su, sv, radius=dot_radius))
                mask_frames.append(goal_mask_image(h_rot, w_rot, su, sv, radius=mask_radius))
            else:
                rgb_goal_frames.append(frame.copy())
                mask_frames.append(np.zeros((h_rot, w_rot), dtype=np.uint8))
        else:
            rgb_goal_frames.append(frame.copy())
            mask_frames.append(np.zeros((h_rot, w_rot), dtype=np.uint8))

    return {
        "rgb_goal": np.stack(rgb_goal_frames, axis=0),
        "goal_mask": np.stack(mask_frames, axis=0),
        "goal_uv": goal_uv,
        "goal_visible": goal_visible,
    }


def build_goal_overlay_from_bundle(
    bundle_dir: Path,
    event_index: dict[tuple[int, int], list[ContactEvent]],
    projector: AriaFisheyeProjector,
    rgb_calib: RgbCameraCalibration,
    *,
    fov_margin_rad: float = 0.05,
    dot_radius: int = 11,
    mask_radius: int = 11,
) -> dict[str, np.ndarray] | None:
    """Build goal overlays for one bundle using stored ego RGB + trajectory."""
    rgb_path = bundle_dir / "ego_rgb.npz"
    traj_path = bundle_dir / "trajectory.npz"
    meta_path = bundle_dir / "metadata.json"
    if not rgb_path.is_file() or not traj_path.is_file() or not meta_path.is_file():
        return None

    import json

    meta = json.loads(meta_path.read_text())
    goal = None
    obj_path = bundle_dir / "object.npz"
    if obj_path.is_file():
        obj_npz = np.load(obj_path)
        if "goal_world" in obj_npz.files:
            goal = np.asarray(obj_npz["goal_world"], dtype=np.float64).reshape(3)
    if goal is None:
        goal = contact_goal_world(
            event_index,
            object_uid=int(meta["object_uid"]),
            contact_start_frame=int(meta["contact_start_frame_idx"]),
            contact_end_frame=int(meta.get("contact_end_frame_idx", meta["contact_start_frame_idx"])),
        )
    if goal is None:
        return None

    rgb_npz = np.load(rgb_path)
    order = np.argsort(rgb_npz["frame_indices"].astype(np.int64))
    rgb = np.asarray(rgb_npz["rgb"][order], dtype=np.uint8)
    frame_indices = np.asarray(rgb_npz["frame_indices"][order], dtype=np.int32)

    traj = np.load(traj_path)
    return annotate_rgb_sequence(
        rgb,
        frame_indices,
        traj["camera_position"],
        traj["camera_rotation"],
        goal,
        projector,
        rgb_calib,
        fov_margin_rad=fov_margin_rad,
        dot_radius=dot_radius,
        mask_radius=mask_radius,
    )
