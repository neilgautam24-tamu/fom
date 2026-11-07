"""Export visible-goal approach sequences with annotated ego RGB and goal masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fom.preprocessing.nymeria.body.smpl_motion import save_smpl_motion_npz, slice_smpl_motion
from fom.preprocessing.nymeria.body.smplx_motion import save_motion_npz, slice_motion
from fom.preprocessing.nymeria.goals.approach_sequences import ApproachContactSequence
from fom.preprocessing.nymeria.goals.ego_projection import (
    AriaFisheyeProjector,
    draw_goal_dot,
    goal_mask_image,
)
from fom.preprocessing.nymeria.goals.sequence_export import (
    build_timing_arrays,
    rotate_image_uv_cw90,
    rotate_rgb_cw90,
)
from fom.preprocessing.nymeria.goals.trajectory import HeadTrajectorySamples, slice_trajectory
from fom.preprocessing.nymeria.goals.visible_goal_sequences import (
    VisibleGoalSequence,
    compute_motion_goal_visibility,
)
from fom.preprocessing.nymeria.io.aria_slam import AriaRgbReader, RgbCameraCalibration
from fom.preprocessing.nymeria.io.objects import ObjectBBox3D


def _sequence_dir_name(seq: ApproachContactSequence) -> str:
    return f"vgoal_seq_{seq.sequence_id:06d}_obj{seq.object_uid}"


def export_goal_frames(
    rgb_reader: AriaRgbReader,
    projector: AriaFisheyeProjector,
    device_timestamps_us: np.ndarray,
    camera_position: np.ndarray,
    camera_quat_or_rot,
    goal_world: np.ndarray,
    rgb_calib: RgbCameraCalibration,
    *,
    max_edge: int = 512,
    rgb_stride: int = 1,
    fov_margin_rad: float = 0.05,
    dot_radius: int = 11,
    mask_radius: int = 11,
) -> dict[str, np.ndarray]:
    """Load ego RGB and build annotated RGB + goal-mask sequences."""
    idxs = np.arange(0, len(device_timestamps_us), max(1, rgb_stride), dtype=np.int32)
    rgb_frames: list[np.ndarray] = []
    rgb_goal_frames: list[np.ndarray] = []
    mask_frames: list[np.ndarray] = []
    goal_uv = np.full((len(idxs), 2), -1.0, dtype=np.float32)
    goal_visible = np.zeros(len(idxs), dtype=np.int8)
    camera_rotation = np.asarray(camera_quat_or_rot, dtype=np.float64)

    for slot, local_idx in enumerate(idxs):
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
        empty_mask = np.empty((0, 0, 0), dtype=np.uint8)
        return {
            "rgb": empty_rgb,
            "rgb_goal": empty_rgb,
            "goal_mask": empty_mask,
            "frame_indices": idxs,
            "goal_uv": goal_uv,
            "goal_visible": goal_visible,
        }

    return {
        "rgb": np.stack(rgb_frames, axis=0),
        "rgb_goal": np.stack(rgb_goal_frames, axis=0),
        "goal_mask": np.stack(mask_frames, axis=0),
        "frame_indices": idxs,
        "goal_uv": goal_uv,
        "goal_visible": goal_visible,
    }


def export_visible_goal_bundle(
    seq: VisibleGoalSequence,
    out_root: Path,
    motion: dict[str, np.ndarray],
    head_traj: HeadTrajectorySamples,
    rgb_calib: RgbCameraCalibration,
    object_bbox: ObjectBBox3D,
    rgb_reader: AriaRgbReader | None,
    projector: AriaFisheyeProjector | None,
    *,
    max_rgb_edge: int = 512,
    rgb_stride: int = 1,
    fov_margin_rad: float = 0.05,
    dot_radius: int = 11,
    mask_radius: int = 11,
    smpl_motion: dict[str, np.ndarray] | None = None,
) -> Path:
    seq_dir = out_root / _sequence_dir_name(seq)
    seq_dir.mkdir(parents=True, exist_ok=True)

    start = seq.start_frame_idx
    end = seq.end_frame_idx
    motion_slice = slice_motion(motion, start, end)
    traj_slice = slice_trajectory(head_traj, start, end)
    traj_slice["rgb_fov_y_rad"] = np.array([rgb_calib.fov_y_rad], dtype=np.float32)
    traj_slice["rgb_aspect"] = np.array([rgb_calib.aspect], dtype=np.float32)

    goal_world = np.array(
        [seq.goal_world_x, seq.goal_world_y, seq.goal_world_z],
        dtype=np.float32,
    )
    timing = build_timing_arrays(seq, motion_slice, traj_slice, object_bbox.center_world)
    goal_visible_motion = compute_motion_goal_visibility(
        seq,
        head_traj.camera_position,
        head_traj.camera_rotation,
        rgb_calib,
        fov_margin_rad=fov_margin_rad,
    )
    timing["goal_visible"] = goal_visible_motion.astype(np.int8)

    save_motion_npz(motion_slice, seq_dir / "smplx_motion.npz")
    if smpl_motion is not None:
        save_smpl_motion_npz(slice_smpl_motion(smpl_motion, start, end), seq_dir / "smpl_motion.npz")
    np.savez_compressed(seq_dir / "trajectory.npz", **traj_slice)
    np.savez_compressed(seq_dir / "timing.npz", **timing)
    np.savez_compressed(
        seq_dir / "object.npz",
        object_uid=np.array([seq.object_uid], dtype=np.int32),
        category=np.array([seq.category]),
        instance_name=np.array([seq.instance_name]),
        center_world=object_bbox.center_world.astype(np.float32),
        dimensions=object_bbox.dimensions.astype(np.float32),
        wxyz=object_bbox.wxyz.astype(np.float32),
        goal_world=goal_world.astype(np.float32),
        contact_hand=np.array([seq.contact_hand]),
    )

    has_rgb = rgb_reader is not None and projector is not None
    if has_rgb:
        rgb_pack = export_goal_frames(
            rgb_reader,
            projector,
            traj_slice["device_timestamp_us"],
            traj_slice["camera_position"],
            traj_slice["camera_rotation"],
            goal_world.astype(np.float64),
            rgb_calib,
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
            rgb_stride=np.array([rgb_stride], dtype=np.int32),
            max_edge=np.array([max_rgb_edge], dtype=np.int32),
            rgb_rotated_cw90=np.array([1], dtype=np.int8),
        )

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
            "ego_rgb": "ego_rgb.npz" if has_rgb else None,
            "ego_rgb_goal": "ego_rgb_goal.npz" if has_rgb else None,
            "goal_mask": "goal_mask.npz" if has_rgb else None,
        },
        "rgb_stride": int(rgb_stride),
        "max_rgb_edge": int(max_rgb_edge),
        "rgb_rotated_cw90": True,
        "goal_world": [float(x) for x in goal_world],
        "contact_hand": seq.contact_hand,
        "goal_visible_fraction": seq.goal_visible_fraction,
        "goal_approach_visible_fraction": seq.goal_approach_visible_fraction,
        "goal_lost_during_approach": seq.goal_lost_during_approach,
        "object_fov_start_frame_idx": seq.object_fov_start_frame_idx,
        "goal_crop_start_frame_idx": seq.start_frame_idx,
    }
    (seq_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return seq_dir


def export_all_visible_goal_sequences(
    sequences: list[VisibleGoalSequence],
    out_root: Path,
    motion: dict[str, np.ndarray],
    head_traj: HeadTrajectorySamples,
    rgb_calib: RgbCameraCalibration,
    objects_by_uid: dict[int, ObjectBBox3D],
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
        obj = objects_by_uid.get(seq.object_uid)
        if obj is None:
            continue
        exported.append(
            export_visible_goal_bundle(
                seq,
                out_root,
                motion,
                head_traj,
                rgb_calib,
                obj,
                rgb_reader,
                projector,
                max_rgb_edge=max_rgb_edge,
                rgb_stride=rgb_stride,
                fov_margin_rad=fov_margin_rad,
                smpl_motion=smpl_motion,
            )
        )
        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"[export]   {i + 1}/{len(sequences)}", flush=True)
    return exported
