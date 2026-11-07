"""Export multi-object visible-goal bundles with per-frame goal switching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fom.preprocessing.nymeria.body.smpl_motion import save_smpl_motion_npz, slice_smpl_motion
from fom.preprocessing.nymeria.body.smplx_motion import save_motion_npz, slice_motion
from fom.preprocessing.nymeria.contacts.detection import ContactInterval
from fom.preprocessing.nymeria.goals.ego_projection import (
    AriaFisheyeProjector,
    draw_goal_dot,
    goal_mask_image,
)
from fom.preprocessing.nymeria.goals.multi_object_sequences import (
    MultiObjectVisibleGoalSequence,
    active_chain_index_at_frame,
    goal_world_at_frame,
)
from fom.preprocessing.nymeria.goals.sequence_export import rotate_image_uv_cw90, rotate_rgb_cw90
from fom.preprocessing.nymeria.goals.trajectory import HeadTrajectorySamples, slice_trajectory
from fom.preprocessing.nymeria.goals.visible_goal_sequences import goal_visible_at_frame
from fom.preprocessing.nymeria.io.aria_slam import AriaRgbReader, RgbCameraCalibration
from fom.preprocessing.nymeria.io.objects import ObjectBBox3D


def _sequence_dir_name(seq: MultiObjectVisibleGoalSequence) -> str:
    uids = "_".join(str(u) for u in seq.chain_object_uids)
    return f"mvgoal_seq_{seq.sequence_id:06d}_chain{len(seq.chain_object_uids)}_{uids}"


def export_multi_goal_frames(
    rgb_reader: AriaRgbReader,
    projector: AriaFisheyeProjector,
    device_timestamps_us: np.ndarray,
    camera_position: np.ndarray,
    camera_quat_or_rot,
    seq: MultiObjectVisibleGoalSequence,
    global_start: int,
    rgb_calib: RgbCameraCalibration,
    intervals: list[ContactInterval] | None,
    *,
    max_edge: int = 512,
    rgb_stride: int = 1,
    fov_margin_rad: float = 0.05,
    dot_radius: int = 11,
    mask_radius: int = 11,
) -> dict[str, np.ndarray]:
    """Load ego RGB; goal point switches when each chain contact completes."""
    idxs = np.arange(0, len(device_timestamps_us), max(1, rgb_stride), dtype=np.int32)
    rgb_frames: list[np.ndarray] = []
    rgb_goal_frames: list[np.ndarray] = []
    mask_frames: list[np.ndarray] = []
    goal_uv = np.full((len(idxs), 2), -1.0, dtype=np.float32)
    goal_visible = np.zeros(len(idxs), dtype=np.int8)
    active_uid = np.full(len(idxs), -1, dtype=np.int32)
    active_chain_idx = np.full(len(idxs), -1, dtype=np.int16)
    camera_rotation = np.asarray(camera_quat_or_rot, dtype=np.float64)

    for slot, local_idx in enumerate(idxs):
        global_idx = int(global_start) + int(local_idx)
        goal_world = goal_world_at_frame(seq, global_idx, intervals)
        chain_i = active_chain_index_at_frame(seq, global_idx, intervals)
        active_uid[slot] = int(seq.chain_object_uids[chain_i])
        active_chain_idx[slot] = int(chain_i)

        rgb = rgb_reader.rgb_at_device_time_us(
            int(device_timestamps_us[local_idx]),
            max_edge=max_edge,
        )
        rgb = np.asarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]

        proj = projector.project_world(
            goal_world,
            camera_position[local_idx],
            camera_rotation[local_idx],
            fov_y_rad=rgb_calib.fov_y_rad,
            aspect=rgb_calib.aspect,
            margin_rad=fov_margin_rad,
        )
        su, sv = -1.0, -1.0
        if proj.visible:
            su, sv = projector.scale_uv_to_image(proj.u, proj.v, w, h)

        rgb = rotate_rgb_cw90(rgb)
        h_rot, w_rot = rgb.shape[:2]
        if proj.visible:
            su, sv = rotate_image_uv_cw90(su, sv, w, h)
            goal_uv[slot] = (su, sv)
            goal_visible[slot] = 1
            rgb_goal = draw_goal_dot(rgb, su, sv, radius=dot_radius)
            mask = goal_mask_image(h_rot, w_rot, su, sv, radius=mask_radius)
        else:
            rgb_goal = rgb.copy()
            mask = np.zeros((h_rot, w_rot), dtype=np.uint8)

        rgb_frames.append(rgb)
        rgb_goal_frames.append(rgb_goal)
        mask_frames.append(mask)

    if not rgb_frames:
        empty_rgb = np.empty((0, 0, 0, 3), dtype=np.uint8)
        return {
            "rgb": empty_rgb,
            "rgb_goal": empty_rgb,
            "goal_mask": np.empty((0, 0, 0), dtype=np.uint8),
            "frame_indices": idxs,
            "goal_uv": goal_uv,
            "goal_visible": goal_visible,
            "active_object_uid": active_uid,
            "active_chain_idx": active_chain_idx,
        }

    return {
        "rgb": np.stack(rgb_frames, axis=0),
        "rgb_goal": np.stack(rgb_goal_frames, axis=0),
        "goal_mask": np.stack(mask_frames, axis=0),
        "frame_indices": idxs,
        "goal_uv": goal_uv,
        "goal_visible": goal_visible,
        "active_object_uid": active_uid,
        "active_chain_idx": active_chain_idx,
    }


def build_multi_timing_arrays(
    seq: MultiObjectVisibleGoalSequence,
    motion_slice: dict[str, np.ndarray],
    traj_slice: dict[str, np.ndarray],
    intervals: list[ContactInterval] | None,
    rgb_calib: RgbCameraCalibration,
    *,
    fov_margin_rad: float,
) -> dict[str, np.ndarray]:
    n = int(motion_slice["num_frames"])
    rel_t = (motion_slice["timestamps"].astype(np.float64) - motion_slice["timestamps"][0]) * 1e-6
    active_uid = np.zeros(n, dtype=np.int32)
    active_chain = np.zeros(n, dtype=np.int16)
    goal_visible = np.zeros(n, dtype=np.int8)
    in_contact = np.zeros(n, dtype=np.int8)

    for i in range(n):
        gf = int(seq.start_frame_idx) + i
        ci = active_chain_index_at_frame(seq, gf, intervals)
        active_chain[i] = ci
        active_uid[i] = int(seq.chain_object_uids[ci])
        c0 = int(seq.chain_contact_starts[ci])
        c1 = int(seq.chain_contact_ends[ci])
        in_contact[i] = int(c0 <= gf <= c1)
        gw = goal_world_at_frame(seq, gf, intervals)
        goal_visible[i] = int(
            goal_visible_at_frame(
                gw,
                i,
                traj_slice["camera_position"],
                traj_slice["camera_rotation"],
                rgb_calib,
                fov_margin_rad=fov_margin_rad,
            )
        )

    return {
        "rel_time_s": rel_t.astype(np.float32),
        "active_object_uid": active_uid,
        "active_chain_idx": active_chain,
        "goal_visible": goal_visible,
        "in_contact": in_contact,
        "chain_contact_starts": np.array(seq.chain_contact_starts, dtype=np.int32),
        "chain_contact_ends": np.array(seq.chain_contact_ends, dtype=np.int32),
        "body_timestamp_us": traj_slice["body_timestamp_us"],
        "device_timestamp_us": traj_slice["device_timestamp_us"],
    }


def export_multi_object_bundle(
    seq: MultiObjectVisibleGoalSequence,
    out_root: Path,
    motion: dict[str, np.ndarray],
    head_traj: HeadTrajectorySamples,
    rgb_calib: RgbCameraCalibration,
    objects_by_uid: dict[int, ObjectBBox3D],
    intervals: list[ContactInterval],
    rgb_reader: AriaRgbReader | None,
    projector: AriaFisheyeProjector | None,
    *,
    smpl_motion: dict[str, np.ndarray] | None = None,
    max_rgb_edge: int = 512,
    rgb_stride: int = 1,
    fov_margin_rad: float = 0.05,
    dot_radius: int = 11,
    mask_radius: int = 11,
) -> Path:
    seq_dir = out_root / _sequence_dir_name(seq)
    seq_dir.mkdir(parents=True, exist_ok=True)

    start = int(seq.start_frame_idx)
    end = int(seq.end_frame_idx)
    motion_slice = slice_motion(motion, start, end)
    traj_slice = slice_trajectory(head_traj, start, end)
    traj_slice["rgb_fov_y_rad"] = np.array([rgb_calib.fov_y_rad], dtype=np.float32)
    traj_slice["rgb_aspect"] = np.array([rgb_calib.aspect], dtype=np.float32)

    timing = build_multi_timing_arrays(
        seq,
        motion_slice,
        traj_slice,
        intervals,
        rgb_calib,
        fov_margin_rad=fov_margin_rad,
    )

    save_motion_npz(motion_slice, seq_dir / "smplx_motion.npz")
    if smpl_motion is not None:
        save_smpl_motion_npz(slice_smpl_motion(smpl_motion, start, end), seq_dir / "smpl_motion.npz")
    np.savez_compressed(seq_dir / "trajectory.npz", **traj_slice)
    np.savez_compressed(seq_dir / "timing.npz", **timing)

    uids = np.array(seq.chain_object_uids, dtype=np.int32)
    centers = np.stack(
        [objects_by_uid[int(u)].center_world.astype(np.float32) for u in seq.chain_object_uids],
        axis=0,
    )
    dims = np.stack(
        [objects_by_uid[int(u)].dimensions.astype(np.float32) for u in seq.chain_object_uids],
        axis=0,
    )
    wxyz = np.stack(
        [objects_by_uid[int(u)].wxyz.astype(np.float32) for u in seq.chain_object_uids],
        axis=0,
    )
    goals = np.array(seq.chain_goal_world, dtype=np.float32)
    np.savez_compressed(
        seq_dir / "objects.npz",
        chain_object_uids=uids,
        chain_categories=np.array(seq.chain_categories),
        chain_instance_names=np.array(seq.chain_instance_names),
        chain_hands=np.array(seq.chain_hands),
        chain_contact_starts=np.array(seq.chain_contact_starts, dtype=np.int32),
        chain_contact_ends=np.array(seq.chain_contact_ends, dtype=np.int32),
        chain_goal_world=goals,
        center_world=centers,
        dimensions=dims,
        wxyz=wxyz,
    )

    has_rgb = rgb_reader is not None and projector is not None
    if has_rgb:
        rgb_pack = export_multi_goal_frames(
            rgb_reader,
            projector,
            traj_slice["device_timestamp_us"],
            traj_slice["camera_position"],
            traj_slice["camera_rotation"],
            seq,
            start,
            rgb_calib,
            intervals,
            max_edge=max_rgb_edge,
            rgb_stride=rgb_stride,
            fov_margin_rad=fov_margin_rad,
            dot_radius=dot_radius,
            mask_radius=mask_radius,
        )
        np.savez_compressed(
            seq_dir / "ego_rgb.npz",
            rgb=rgb_pack["rgb"],
            frame_indices=rgb_pack["frame_indices"],
            rgb_stride=np.array([rgb_stride], dtype=np.int32),
            max_edge=np.array([max_rgb_edge], dtype=np.int32),
            rgb_rotated_cw90=np.array([1], dtype=np.int8),
            device_timestamp_us=traj_slice["device_timestamp_us"][rgb_pack["frame_indices"]],
        )
        np.savez_compressed(
            seq_dir / "ego_rgb_goal.npz",
            rgb=rgb_pack["rgb_goal"],
            frame_indices=rgb_pack["frame_indices"],
            goal_uv=rgb_pack["goal_uv"],
            goal_visible=rgb_pack["goal_visible"],
            active_object_uid=rgb_pack["active_object_uid"],
            active_chain_idx=rgb_pack["active_chain_idx"],
            rgb_stride=np.array([rgb_stride], dtype=np.int32),
            max_edge=np.array([max_rgb_edge], dtype=np.int32),
            rgb_rotated_cw90=np.array([1], dtype=np.int8),
            device_timestamp_us=traj_slice["device_timestamp_us"][rgb_pack["frame_indices"]],
        )
        np.savez_compressed(
            seq_dir / "goal_mask.npz",
            mask=rgb_pack["goal_mask"],
            frame_indices=rgb_pack["frame_indices"],
            goal_uv=rgb_pack["goal_uv"],
            goal_visible=rgb_pack["goal_visible"],
            active_object_uid=rgb_pack["active_object_uid"],
            active_chain_idx=rgb_pack["active_chain_idx"],
            rgb_stride=np.array([rgb_stride], dtype=np.int32),
            max_edge=np.array([max_rgb_edge], dtype=np.int32),
            rgb_rotated_cw90=np.array([1], dtype=np.int8),
        )

    metadata = {
        **seq.to_dict(),
        "num_local_frames": int(motion_slice["num_frames"]),
        "global_start_frame_idx": start,
        "global_end_frame_idx": end,
        "sequence_type": "multi_object_visible_goal",
        "files": {
            "smpl_motion": "smpl_motion.npz" if smpl_motion is not None else None,
            "smplx_motion": "smplx_motion.npz",
            "trajectory": "trajectory.npz",
            "timing": "timing.npz",
            "objects": "objects.npz",
            "ego_rgb": "ego_rgb.npz" if has_rgb else None,
            "ego_rgb_goal": "ego_rgb_goal.npz" if has_rgb else None,
            "goal_mask": "goal_mask.npz" if has_rgb else None,
        },
        "rgb_stride": int(rgb_stride),
        "max_rgb_edge": int(max_rgb_edge),
        "rgb_rotated_cw90": True,
    }
    (seq_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return seq_dir


def export_all_multi_object_sequences(
    sequences: list[MultiObjectVisibleGoalSequence],
    out_root: Path,
    motion: dict[str, np.ndarray],
    head_traj: HeadTrajectorySamples,
    rgb_calib: RgbCameraCalibration,
    objects_by_uid: dict[int, ObjectBBox3D],
    intervals: list[ContactInterval],
    rgb_reader: AriaRgbReader | None,
    projector: AriaFisheyeProjector | None,
    *,
    smpl_motion: dict[str, np.ndarray] | None = None,
    max_rgb_edge: int = 512,
    rgb_stride: int = 1,
    fov_margin_rad: float = 0.05,
    progress_every: int = 5,
) -> list[Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for i, seq in enumerate(sequences):
        exported.append(
            export_multi_object_bundle(
                seq,
                out_root,
                motion,
                head_traj,
                rgb_calib,
                objects_by_uid,
                intervals,
                rgb_reader,
                projector,
                smpl_motion=smpl_motion,
                max_rgb_edge=max_rgb_edge,
                rgb_stride=rgb_stride,
                fov_margin_rad=fov_margin_rad,
            )
        )
        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"[export]   {i + 1}/{len(sequences)}", flush=True)
    return exported
