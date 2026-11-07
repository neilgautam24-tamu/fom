"""Export per-sequence bundles: ego RGB, SMPL-X motion, trajectory, timing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fom.preprocessing.nymeria.body.smpl_motion import save_smpl_motion_npz, slice_smpl_motion
from fom.preprocessing.nymeria.body.smplx_motion import save_motion_npz, slice_motion
from fom.preprocessing.nymeria.contacts.detection import ContactEvent
from fom.preprocessing.nymeria.goals.approach import approach_alignment, pelvis_velocity
from fom.preprocessing.nymeria.goals.approach_sequences import ApproachContactSequence
from fom.preprocessing.nymeria.goals.ego_projection import AriaFisheyeProjector
from fom.preprocessing.nymeria.goals.goal_rgb_overlay import annotate_rgb_sequence, contact_goal_world
from fom.preprocessing.nymeria.goals.trajectory import HeadTrajectorySamples, slice_trajectory
from fom.preprocessing.nymeria.io.aria_slam import AriaRgbReader, RgbCameraCalibration
from fom.preprocessing.nymeria.io.objects import ObjectBBox3D


def rotate_rgb_cw90(rgb: np.ndarray) -> np.ndarray:
    """Rotate an RGB frame 90 deg clockwise before bundle export."""
    return np.rot90(np.asarray(rgb, dtype=np.uint8), k=-1).copy()


def rotate_image_uv_cw90(u: float, v: float, width: int, height: int) -> tuple[float, float]:
    """Map pixel coords from a pre-rotation image to the CW-rotated image."""
    if u < 0.0 or v < 0.0:
        return -1.0, -1.0
    return float(height - 1.0 - v), float(u)


def _sequence_dir_name(seq: ApproachContactSequence) -> str:
    return f"seq_{seq.sequence_id:06d}_obj{seq.object_uid}"


def build_timing_arrays(
    seq: ApproachContactSequence,
    motion_slice: dict[str, np.ndarray],
    traj_slice: dict[str, np.ndarray],
    object_center: np.ndarray,
) -> dict[str, np.ndarray]:
    n = int(motion_slice["num_frames"])
    rel_t = (motion_slice["timestamps"].astype(np.float64) - motion_slice["timestamps"][0]) * 1e-6
    phase = np.zeros(n, dtype=np.int8)
    contact_local = seq.contact_start_frame_idx - seq.start_frame_idx
    phase[contact_local:] = 1

    pelvis_vel = np.zeros((n, 3), dtype=np.float32)
    speed = np.zeros(n, dtype=np.float32)
    align = np.zeros(n, dtype=np.float32)
    ts = motion_slice["timestamps"]
    transl = motion_slice["transl"]
    obj_c = np.asarray(object_center, dtype=np.float32)
    for i in range(n):
        global_i = seq.start_frame_idx + i
        vel = pelvis_velocity(transl, ts, i, window=3)
        sp, al = approach_alignment(transl[i], obj_c, vel)
        pelvis_vel[i] = vel.astype(np.float32)
        speed[i] = sp
        align[i] = al

    return {
        "rel_time_s": rel_t.astype(np.float32),
        "phase": phase,
        "phase_label": np.array(["approach", "contact"], dtype="<U8"),
        "contact_start_local_idx": np.array([contact_local], dtype=np.int32),
        "contact_end_local_idx": np.array(
            [seq.contact_end_frame_idx - seq.start_frame_idx], dtype=np.int32
        ),
        "pelvis_velocity": pelvis_vel,
        "approach_speed_m_s": speed,
        "approach_alignment": align,
        "body_timestamp_us": traj_slice["body_timestamp_us"],
        "device_timestamp_us": traj_slice["device_timestamp_us"],
    }


def load_rgb_frames(
    rgb_reader: AriaRgbReader,
    device_timestamps_us: np.ndarray,
    *,
    max_edge: int = 512,
    rgb_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load egocentric RGB frames; return ``(frames, frame_indices)``."""
    idxs = np.arange(0, len(device_timestamps_us), max(1, rgb_stride), dtype=np.int32)
    frames: list[np.ndarray] = []
    for i in idxs:
        rgb = rgb_reader.rgb_at_device_time_us(int(device_timestamps_us[i]), max_edge=max_edge)
        frames.append(rotate_rgb_cw90(rgb))
    if not frames:
        return np.empty((0, 0, 0, 3), dtype=np.uint8), idxs
    return np.stack(frames, axis=0), idxs


def export_sequence_bundle(
    seq: ApproachContactSequence,
    out_root: Path,
    motion: dict[str, np.ndarray],
    head_traj: HeadTrajectorySamples,
    rgb_calib: RgbCameraCalibration,
    object_bbox: ObjectBBox3D,
    rgb_reader: AriaRgbReader | None,
    *,
    max_rgb_edge: int = 512,
    rgb_stride: int = 1,
    contact_event_index: dict[tuple[int, int], list[ContactEvent]] | None = None,
    projector: AriaFisheyeProjector | None = None,
    fov_margin_rad: float = 0.05,
    dot_radius: int = 11,
    mask_radius: int = 11,
    smpl_motion: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write one approach->contact sequence folder with all modalities."""
    seq_dir = out_root / _sequence_dir_name(seq)
    seq_dir.mkdir(parents=True, exist_ok=True)

    start = seq.start_frame_idx
    end = seq.end_frame_idx
    motion_slice = slice_motion(motion, start, end)
    traj_slice = slice_trajectory(head_traj, start, end)
    traj_slice["rgb_fov_y_rad"] = np.array([rgb_calib.fov_y_rad], dtype=np.float32)
    traj_slice["rgb_aspect"] = np.array([rgb_calib.aspect], dtype=np.float32)

    timing = build_timing_arrays(seq, motion_slice, traj_slice, object_bbox.center_world)

    goal_world = None
    contact_hand = None
    if contact_event_index is not None:
        goal_info = contact_goal_world(
            contact_event_index,
            object_uid=seq.object_uid,
            contact_start_frame=seq.contact_start_frame_idx,
            contact_end_frame=seq.contact_end_frame_idx,
        )
        if goal_info is not None:
            goal_world = goal_info

    save_motion_npz(motion_slice, seq_dir / "smplx_motion.npz")
    if smpl_motion is not None:
        save_smpl_motion_npz(slice_smpl_motion(smpl_motion, start, end), seq_dir / "smpl_motion.npz")
    np.savez_compressed(seq_dir / "trajectory.npz", **traj_slice)
    np.savez_compressed(seq_dir / "timing.npz", **timing)

    object_payload: dict[str, np.ndarray] = {
        "object_uid": np.array([seq.object_uid], dtype=np.int32),
        "category": np.array([seq.category]),
        "instance_name": np.array([seq.instance_name]),
        "center_world": object_bbox.center_world.astype(np.float32),
        "dimensions": object_bbox.dimensions.astype(np.float32),
        "wxyz": object_bbox.wxyz.astype(np.float32),
    }
    if goal_world is not None:
        object_payload["goal_world"] = goal_world.astype(np.float32)
        if contact_event_index is not None:
            frame_events = contact_event_index.get(
                (seq.object_uid, seq.contact_start_frame_idx), []
            )
            if frame_events:
                contact_hand = str(min(frame_events, key=lambda e: e.distance_m).hand)
                object_payload["contact_hand"] = np.array([contact_hand])
    np.savez_compressed(seq_dir / "object.npz", **object_payload)

    has_goal_rgb = False
    if rgb_reader is not None:
        rgb_frames, rgb_idxs = load_rgb_frames(
            rgb_reader,
            traj_slice["device_timestamp_us"],
            max_edge=max_rgb_edge,
            rgb_stride=rgb_stride,
        )
        np.savez_compressed(
            seq_dir / "ego_rgb.npz",
            rgb=rgb_frames,
            frame_indices=rgb_idxs,
            rgb_stride=np.array([rgb_stride], dtype=np.int32),
            max_edge=np.array([max_rgb_edge], dtype=np.int32),
            rgb_rotated_cw90=np.array([1], dtype=np.int8),
            device_timestamp_us=traj_slice["device_timestamp_us"][rgb_idxs],
        )
        if goal_world is not None and projector is not None and rgb_frames.size > 0:
            goal_pack = annotate_rgb_sequence(
                rgb_frames,
                rgb_idxs,
                traj_slice["camera_position"],
                traj_slice["camera_rotation"],
                goal_world,
                projector,
                rgb_calib,
                fov_margin_rad=fov_margin_rad,
                dot_radius=dot_radius,
                mask_radius=mask_radius,
            )
            np.savez_compressed(
                seq_dir / "ego_rgb_goal.npz",
                rgb=goal_pack["rgb_goal"],
                frame_indices=rgb_idxs,
                goal_uv=goal_pack["goal_uv"],
                goal_visible=goal_pack["goal_visible"],
                rgb_stride=np.array([rgb_stride], dtype=np.int32),
                max_edge=np.array([max_rgb_edge], dtype=np.int32),
                rgb_rotated_cw90=np.array([1], dtype=np.int8),
                device_timestamp_us=traj_slice["device_timestamp_us"][rgb_idxs],
            )
            np.savez_compressed(
                seq_dir / "goal_mask.npz",
                mask=goal_pack["goal_mask"],
                frame_indices=rgb_idxs,
                goal_uv=goal_pack["goal_uv"],
                goal_visible=goal_pack["goal_visible"],
                rgb_stride=np.array([rgb_stride], dtype=np.int32),
                max_edge=np.array([max_rgb_edge], dtype=np.int32),
                rgb_rotated_cw90=np.array([1], dtype=np.int8),
            )
            has_goal_rgb = True

    metadata = {
        **seq.to_dict(),
        "num_local_frames": int(motion_slice["num_frames"]),
        "global_start_frame_idx": start,
        "global_end_frame_idx": end,
        "files": {
            "smpl_motion": "smpl_motion.npz" if smpl_motion is not None else None,
            "smplx_motion": "smplx_motion.npz",
            "trajectory": "trajectory.npz",
            "timing": "timing.npz",
            "object": "object.npz",
            "ego_rgb": "ego_rgb.npz" if rgb_reader is not None else None,
            "ego_rgb_goal": "ego_rgb_goal.npz" if has_goal_rgb else None,
            "goal_mask": "goal_mask.npz" if has_goal_rgb else None,
        },
        "rgb_stride": int(rgb_stride),
        "max_rgb_edge": int(max_rgb_edge),
        "rgb_rotated_cw90": True,
    }
    if goal_world is not None:
        metadata["goal_world"] = [float(x) for x in goal_world]
    if contact_hand is not None:
        metadata["contact_hand"] = contact_hand
    (seq_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return seq_dir


def export_all_sequences(
    sequences: list[ApproachContactSequence],
    out_root: Path,
    motion: dict[str, np.ndarray],
    head_traj: HeadTrajectorySamples,
    rgb_calib: RgbCameraCalibration,
    objects_by_uid: dict[int, ObjectBBox3D],
    rgb_reader: AriaRgbReader | None,
    *,
    smpl_motion: dict[str, np.ndarray] | None = None,
    max_rgb_edge: int = 512,
    rgb_stride: int = 1,
    contact_event_index: dict[tuple[int, int], list[ContactEvent]] | None = None,
    projector: AriaFisheyeProjector | None = None,
    fov_margin_rad: float = 0.05,
    progress_every: int = 10,
) -> list[Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for i, seq in enumerate(sequences):
        obj = objects_by_uid.get(seq.object_uid)
        if obj is None:
            continue
        exported.append(
            export_sequence_bundle(
                seq,
                out_root,
                motion,
                head_traj,
                rgb_calib,
                obj,
                rgb_reader,
                max_rgb_edge=max_rgb_edge,
                rgb_stride=rgb_stride,
                contact_event_index=contact_event_index,
                projector=projector,
                fov_margin_rad=fov_margin_rad,
                smpl_motion=smpl_motion,
            )
        )
        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"[export]   {i + 1}/{len(sequences)}", flush=True)
    return exported
