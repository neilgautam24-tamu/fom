"""Approach->contact sequences cropped to when the first-contact goal enters head FOV."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from fom.preprocessing.nymeria.contacts.detection import ContactEvent, ContactInterval
from fom.preprocessing.nymeria.goals.approach import approach_alignment, pelvis_velocity, validate_approach_segment
from fom.preprocessing.nymeria.goals.approach_sequences import (
    _find_fov_onset_frame,
    merge_object_contact_episodes,
)
from fom.preprocessing.nymeria.goals.fov import points_in_rgb_frustum
from fom.preprocessing.nymeria.io.aria_slam import RgbCameraCalibration
from fom.preprocessing.nymeria.io.objects import ObjectBBox3D


@dataclass(frozen=True)
class VisibleGoalSequence:
    """Sequence with t=0 at first head-FOV sighting of the first-contact goal point."""

    sequence_id: int
    object_uid: int
    category: str
    instance_name: str
    start_frame_idx: int
    contact_start_frame_idx: int
    contact_end_frame_idx: int
    end_frame_idx: int
    start_timestamp_us: int
    contact_start_timestamp_us: int
    contact_end_timestamp_us: int
    end_timestamp_us: int
    start_time_s: float
    contact_start_time_s: float
    contact_end_time_s: float
    end_time_s: float
    approach_duration_s: float
    contact_duration_s: float
    total_duration_s: float
    median_approach_speed_m_s: float
    median_approach_alignment: float
    approach_valid_fraction: float
    contact_hand: str
    goal_world_x: float
    goal_world_y: float
    goal_world_z: float
    goal_visible_at_start: bool
    goal_visible_fraction: float
    goal_approach_visible_fraction: float
    goal_lost_during_approach: bool
    object_fov_start_frame_idx: int

    def to_dict(self) -> dict:
        return asdict(self)


def _frame_time_s(timestamps_us: np.ndarray, frame_idx: int) -> float:
    t0 = int(timestamps_us[0])
    return (int(timestamps_us[frame_idx]) - t0) * 1e-6


def index_contact_events(
    events: list[ContactEvent],
) -> dict[tuple[int, int], list[ContactEvent]]:
    out: dict[tuple[int, int], list[ContactEvent]] = defaultdict(list)
    for event in events:
        out[(event.object_uid, event.frame_idx)].append(event)
    return out


def first_contact_goal(
    event_index: dict[tuple[int, int], list[ContactEvent]],
    *,
    object_uid: int,
    contact_start_frame: int,
) -> tuple[np.ndarray, str] | None:
    """Return ``(goal_world, hand)`` from the first contact frame for one object."""
    frame_events = event_index.get((object_uid, contact_start_frame), [])
    if not frame_events:
        return None
    best = min(frame_events, key=lambda e: e.distance_m)
    goal = np.array(
        [best.object_point_x, best.object_point_y, best.object_point_z],
        dtype=np.float64,
    )
    return goal, str(best.hand)


def goal_visible_at_frame(
    goal_world: np.ndarray,
    frame_idx: int,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    rgb_calib: RgbCameraCalibration,
    *,
    fov_margin_rad: float,
) -> bool:
    point = np.asarray(goal_world, dtype=np.float64).reshape(1, 3)
    return bool(
        points_in_rgb_frustum(
            point,
            camera_position[int(frame_idx)],
            camera_rotation[int(frame_idx)],
            fov_y_rad=rgb_calib.fov_y_rad,
            aspect=rgb_calib.aspect,
            margin_rad=fov_margin_rad,
        )[0]
    )


def goal_visibility_track(
    goal_world: np.ndarray,
    start_frame: int,
    end_frame: int,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    rgb_calib: RgbCameraCalibration,
    *,
    fov_margin_rad: float,
) -> np.ndarray:
    """Per-frame goal-in-FOV flags for ``[start_frame, end_frame]`` inclusive."""
    n = int(end_frame - start_frame + 1)
    flags = np.zeros(n, dtype=np.int8)
    for offset, frame_idx in enumerate(range(int(start_frame), int(end_frame) + 1)):
        flags[offset] = int(
            goal_visible_at_frame(
                goal_world,
                frame_idx,
                camera_position,
                camera_rotation,
                rgb_calib,
                fov_margin_rad=fov_margin_rad,
            )
        )
    return flags


def _find_goal_fov_onset_frame(
    goal_world: np.ndarray,
    contact_start_frame: int,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    rgb_calib: RgbCameraCalibration,
    *,
    max_lookback_frames: int,
    fov_margin_rad: float,
) -> int | None:
    """Walk backward from contact to when the goal point first enters head FOV."""
    earliest = max(0, int(contact_start_frame) - int(max_lookback_frames))
    fov_onset: int | None = None

    for frame_idx in range(int(contact_start_frame), earliest - 1, -1):
        if goal_visible_at_frame(
            goal_world,
            frame_idx,
            camera_position,
            camera_rotation,
            rgb_calib,
            fov_margin_rad=fov_margin_rad,
        ):
            fov_onset = frame_idx
        elif fov_onset is not None:
            break

    return fov_onset


def _goal_visibility_stats(
    visibility: np.ndarray,
    *,
    approach_frames: int,
) -> tuple[float, float, bool]:
    """Return ``(full_fraction, approach_fraction, lost_during_approach)``."""
    if visibility.size == 0:
        return 0.0, 0.0, False

    full_fraction = float(visibility.mean())
    approach_vis = visibility[: max(approach_frames, 0)]
    approach_fraction = float(approach_vis.mean()) if approach_vis.size else 0.0

    lost = False
    if approach_vis.size > 1 and approach_vis[0] == 1:
        saw_visible = True
        for flag in approach_vis[1:]:
            if flag == 0:
                lost = True
                break

    return full_fraction, approach_fraction, lost


def extract_visible_goal_sequences(
    intervals: list[ContactInterval],
    contact_events: list[ContactEvent],
    motion_timestamps_us: np.ndarray,
    transl: np.ndarray,
    objects_by_uid: dict[int, ObjectBBox3D],
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    rgb_calib: RgbCameraCalibration,
    *,
    max_gap_frames: int = 2,
    max_goal_lookback_s: float = 25.0,
    dt_s: float = 1.0 / 60.0,
    min_approach_s: float = 0.25,
    min_speed_m_s: float = 0.08,
    min_alignment: float = 0.35,
    min_valid_fraction: float = 0.45,
    fov_margin_rad: float = 0.05,
    object_fov_lookback_s: float = 12.0,
) -> list[VisibleGoalSequence]:
    """Build sequences cropped so local t=0 is the first goal-in-FOV frame.

    Unlike filtering pre-cut approach sequences, this walks back from contact
    using a (typically longer) goal lookback window. Sequences where the goal
    only becomes visible after the object bbox entered FOV are included by
    cropping to the goal-onset frame.
    """
    episodes = merge_object_contact_episodes(intervals, max_gap_frames=max_gap_frames)
    event_index = index_contact_events(contact_events)
    max_goal_lookback_frames = max(1, int(max_goal_lookback_s / max(dt_s, 1e-6)))
    min_approach_frames = max(1, int(min_approach_s / max(dt_s, 1e-6)))
    object_lookback_frames = max(1, int(object_fov_lookback_s / max(dt_s, 1e-6)))

    sequences: list[VisibleGoalSequence] = []
    seq_id = 0
    t_us = motion_timestamps_us

    for episode in episodes:
        obj = objects_by_uid.get(episode.object_uid)
        if obj is None:
            continue

        contact_start = int(episode.start_frame_idx)
        contact_end = int(episode.end_frame_idx)

        goal_info = first_contact_goal(
            event_index,
            object_uid=episode.object_uid,
            contact_start_frame=contact_start,
        )
        if goal_info is None:
            continue
        goal_world, hand = goal_info

        goal_start = _find_goal_fov_onset_frame(
            goal_world,
            contact_start,
            camera_position,
            camera_rotation,
            rgb_calib,
            max_lookback_frames=max_goal_lookback_frames,
            fov_margin_rad=fov_margin_rad,
        )
        if goal_start is None:
            continue

        if contact_start - goal_start < min_approach_frames:
            continue

        approach_end = max(goal_start, contact_start - 1)
        ok, med_speed, med_align = validate_approach_segment(
            transl,
            t_us,
            obj.center_world,
            goal_start,
            approach_end if approach_end >= goal_start else contact_start,
            min_speed_m_s=min_speed_m_s,
            min_alignment=min_alignment,
            min_valid_fraction=min_valid_fraction,
        )
        if not ok:
            continue

        valid = 0
        total = max(contact_start - goal_start, 1)
        for frame_idx in range(goal_start, contact_start):
            vel = pelvis_velocity(transl, t_us, frame_idx)
            speed, cos_align = approach_alignment(transl[frame_idx], obj.center_world, vel)
            if speed >= min_speed_m_s and cos_align >= min_alignment:
                valid += 1
        valid_fraction = valid / total

        visibility = goal_visibility_track(
            goal_world,
            goal_start,
            contact_end,
            camera_position,
            camera_rotation,
            rgb_calib,
            fov_margin_rad=fov_margin_rad,
        )
        approach_frames = int(contact_start - goal_start)
        full_frac, approach_frac, lost = _goal_visibility_stats(
            visibility,
            approach_frames=approach_frames,
        )

        object_fov_start = _find_fov_onset_frame(
            contact_start_frame=contact_start,
            object_bbox=obj,
            camera_position=camera_position,
            camera_rotation=camera_rotation,
            rgb_calib=rgb_calib,
            max_lookback_frames=object_lookback_frames,
            fov_margin_rad=fov_margin_rad,
        )
        object_fov_frame = int(object_fov_start) if object_fov_start is not None else -1

        start_s = _frame_time_s(t_us, goal_start)
        contact_start_s = _frame_time_s(t_us, contact_start)
        contact_end_s = _frame_time_s(t_us, contact_end)

        sequences.append(
            VisibleGoalSequence(
                sequence_id=seq_id,
                object_uid=episode.object_uid,
                category=episode.category,
                instance_name=episode.instance_name,
                start_frame_idx=goal_start,
                contact_start_frame_idx=contact_start,
                contact_end_frame_idx=contact_end,
                end_frame_idx=contact_end,
                start_timestamp_us=int(t_us[goal_start]),
                contact_start_timestamp_us=int(t_us[contact_start]),
                contact_end_timestamp_us=int(t_us[contact_end]),
                end_timestamp_us=int(t_us[contact_end]),
                start_time_s=start_s,
                contact_start_time_s=contact_start_s,
                contact_end_time_s=contact_end_s,
                end_time_s=contact_end_s,
                approach_duration_s=contact_start_s - start_s,
                contact_duration_s=contact_end_s - contact_start_s,
                total_duration_s=contact_end_s - start_s,
                median_approach_speed_m_s=med_speed,
                median_approach_alignment=med_align,
                approach_valid_fraction=valid_fraction,
                contact_hand=hand,
                goal_world_x=float(goal_world[0]),
                goal_world_y=float(goal_world[1]),
                goal_world_z=float(goal_world[2]),
                goal_visible_at_start=True,
                goal_visible_fraction=full_frac,
                goal_approach_visible_fraction=approach_frac,
                goal_lost_during_approach=lost,
                object_fov_start_frame_idx=object_fov_frame,
            )
        )
        seq_id += 1

    return sequences


def compute_motion_goal_visibility(
    seq: VisibleGoalSequence,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    rgb_calib: RgbCameraCalibration,
    *,
    fov_margin_rad: float,
) -> np.ndarray:
    """Goal visibility flags aligned to exported motion frames (local indices)."""
    goal = np.array(
        [seq.goal_world_x, seq.goal_world_y, seq.goal_world_z],
        dtype=np.float64,
    )
    return goal_visibility_track(
        goal,
        seq.start_frame_idx,
        seq.end_frame_idx,
        camera_position,
        camera_rotation,
        rgb_calib,
        fov_margin_rad=fov_margin_rad,
    )


# Defaults for optional fields of the exported sequence JSON.
VISIBLE_GOAL_DEFAULTS: dict[str, object] = {
    "goal_visible_fraction": 1.0,
    "goal_approach_visible_fraction": 1.0,
    "goal_lost_during_approach": False,
    "object_fov_start_frame_idx": -1,
}


def load_visible_goal_sequence_row(row: dict) -> VisibleGoalSequence:
    payload = {**VISIBLE_GOAL_DEFAULTS, **row}
    return VisibleGoalSequence(**payload)
