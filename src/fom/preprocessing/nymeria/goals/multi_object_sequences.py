"""Multi-object visible-goal sequences: one clip, goals switch after each contact."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from fom.preprocessing.nymeria.contacts.detection import ContactEvent, ContactInterval
from fom.preprocessing.nymeria.goals.interaction_categories import CategoryClip, extract_cat4_multi_object_chain
from fom.preprocessing.nymeria.goals.visible_goal_sequences import (
    _find_goal_fov_onset_frame,
    _frame_time_s,
    first_contact_goal,
    index_contact_events,
)
from fom.preprocessing.nymeria.io.aria_slam import RgbCameraCalibration
from fom.preprocessing.nymeria.io.objects import ObjectBBox3D


@dataclass(frozen=True)
class MultiObjectVisibleGoalSequence:
    """Continuous chain clip; active goal switches when each object contact ends."""

    sequence_id: int
    start_frame_idx: int
    end_frame_idx: int
    start_timestamp_us: int
    end_timestamp_us: int
    start_time_s: float
    end_time_s: float
    total_duration_s: float
    chain_object_uids: tuple[int, ...]
    chain_categories: tuple[str, ...]
    chain_instance_names: tuple[str, ...]
    chain_hands: tuple[str, ...]
    chain_contact_starts: tuple[int, ...]
    chain_contact_ends: tuple[int, ...]
    chain_goal_world: tuple[tuple[float, float, float], ...]
    first_goal_visible_at_start: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["chain_object_uids"] = list(self.chain_object_uids)
        d["chain_categories"] = list(self.chain_categories)
        d["chain_instance_names"] = list(self.chain_instance_names)
        d["chain_hands"] = list(self.chain_hands)
        d["chain_contact_starts"] = list(self.chain_contact_starts)
        d["chain_contact_ends"] = list(self.chain_contact_ends)
        d["chain_goal_world"] = [list(g) for g in self.chain_goal_world]
        d["num_chain_objects"] = len(self.chain_object_uids)
        return d


def _clip_to_category_clip(seq: MultiObjectVisibleGoalSequence) -> CategoryClip:
    """Adapter for shared chain-segment helpers in ``category_gif``."""
    uid0 = int(seq.chain_object_uids[0])
    return CategoryClip(
        clip_id=int(seq.sequence_id),
        category="cat4_multi_object_chain",
        start_frame_idx=int(seq.start_frame_idx),
        contact_start_frame_idx=int(seq.chain_contact_starts[0]),
        contact_end_frame_idx=int(seq.chain_contact_ends[-1]),
        end_frame_idx=int(seq.end_frame_idx),
        object_uid=uid0,
        category_name=str(seq.chain_categories[0]),
        instance_name=str(seq.chain_instance_names[0]),
        contact_hand=str(seq.chain_hands[0]),
        start_timestamp_us=int(seq.start_timestamp_us),
        contact_start_timestamp_us=int(seq.start_timestamp_us),
        contact_end_timestamp_us=int(seq.end_timestamp_us),
        end_timestamp_us=int(seq.end_timestamp_us),
        start_time_s=float(seq.start_time_s),
        contact_start_time_s=float(seq.start_time_s),
        contact_end_time_s=float(seq.end_time_s),
        end_time_s=float(seq.end_time_s),
        total_duration_s=float(seq.total_duration_s),
        chain_object_uids=tuple(int(u) for u in seq.chain_object_uids),
        chain_hands=tuple(str(h) for h in seq.chain_hands),
        chain_contact_starts=tuple(int(s) for s in seq.chain_contact_starts),
    )


def active_chain_index_at_frame(
    seq: MultiObjectVisibleGoalSequence,
    global_frame: int,
    intervals: list[ContactInterval] | None = None,
) -> int:
    """Index into the chain whose goal should be active on ``global_frame``."""
    from fom.preprocessing.nymeria.goals.category_gif import chain_segment_at_frame

    clip = _clip_to_category_clip(seq)
    seg = chain_segment_at_frame(clip, int(global_frame), intervals)
    if seg is None:
        return 0
    uids = [int(u) for u in seq.chain_object_uids]
    try:
        return uids.index(int(seg.object_uid))
    except ValueError:
        return 0


def goal_world_at_frame(
    seq: MultiObjectVisibleGoalSequence,
    global_frame: int,
    intervals: list[ContactInterval] | None = None,
) -> np.ndarray:
    idx = active_chain_index_at_frame(seq, global_frame, intervals)
    g = seq.chain_goal_world[idx]
    return np.array(g, dtype=np.float64)


def extract_multi_object_visible_goal_sequences(
    intervals: list[ContactInterval],
    contact_events: list[ContactEvent],
    motion_timestamps_us: np.ndarray,
    objects_by_uid: dict[int, ObjectBBox3D],
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    rgb_calib: RgbCameraCalibration,
    *,
    max_gap_frames: int = 2,
    max_chain_span_s: float = 120.0,
    max_first_contact_s: float = 6.0,
    max_switch_gap_s: float = 30.0,
    min_switch_gap_s: float = 0.15,
    max_goal_lookback_s: float = 25.0,
    fov_margin_rad: float = 0.05,
    dt_s: float = 1.0 / 60.0,
    min_objects: int = 2,
    max_objects: int = 4,
) -> list[MultiObjectVisibleGoalSequence]:
    """Build multi-object clips cropped to first-object goal entering FOV."""
    event_index = index_contact_events(contact_events)
    goal_lb = max(1, int(max_goal_lookback_s / max(dt_s, 1e-6)))

    chains = extract_cat4_multi_object_chain(
        intervals,
        contact_events,
        motion_timestamps_us,
        objects_by_uid,
        camera_position,
        camera_rotation,
        rgb_calib,
        max_gap_frames=max_gap_frames,
        max_chain_span_s=max_chain_span_s,
        max_first_contact_s=max_first_contact_s,
        max_switch_gap_s=max_switch_gap_s,
        min_switch_gap_s=min_switch_gap_s,
        fov_margin_rad=fov_margin_rad,
        dt_s=dt_s,
        min_objects=min_objects,
        max_objects=max_objects,
    )

    # Deduplicate: cat4 emits one clip per prefix; keep only full chains.
    by_key: dict[tuple[int, ...], CategoryClip] = {}
    for clip in chains:
        key = tuple(int(u) for u in clip.chain_object_uids)
        if len(key) < min_objects:
            continue
        prev = by_key.get(key)
        if prev is None or len(key) > len(prev.chain_object_uids):
            by_key[key] = clip

    sequences: list[MultiObjectVisibleGoalSequence] = []
    seq_id = 0
    interval_by_uid_start = {
        (int(iv.object_uid), int(iv.start_frame_idx)): iv for iv in intervals
    }

    for clip in sorted(by_key.values(), key=lambda c: (c.start_frame_idx, c.chain_object_uids)):
        uids = tuple(int(u) for u in clip.chain_object_uids)
        if not (min_objects <= len(uids) <= max_objects):
            continue

        goals: list[tuple[float, float, float]] = []
        categories: list[str] = []
        names: list[str] = []
        hands: list[str] = []
        c_starts: list[int] = []
        c_ends: list[int] = []

        ok = True
        for i, uid in enumerate(uids):
            c_start = int(clip.chain_contact_starts[i]) if i < len(clip.chain_contact_starts) else int(clip.contact_start_frame_idx)
            ginfo = first_contact_goal(event_index, object_uid=uid, contact_start_frame=c_start)
            if ginfo is None:
                ok = False
                break
            goal, hand = ginfo
            goals.append((float(goal[0]), float(goal[1]), float(goal[2])))
            hands.append(str(hand))
            c_starts.append(c_start)

            iv = interval_by_uid_start.get((uid, c_start))
            if iv is None:
                matches = [x for x in intervals if int(x.object_uid) == uid and int(x.start_frame_idx) <= c_start <= int(x.end_frame_idx)]
                iv = matches[0] if matches else None
            if iv is None:
                ok = False
                break
            categories.append(str(iv.category))
            names.append(str(iv.instance_name))
            c_ends.append(int(iv.end_frame_idx))

        if not ok:
            continue

        first_goal = np.array(goals[0], dtype=np.float64)
        goal_start = _find_goal_fov_onset_frame(
            first_goal,
            c_starts[0],
            camera_position,
            camera_rotation,
            rgb_calib,
            max_lookback_frames=goal_lb,
            fov_margin_rad=fov_margin_rad,
        )
        if goal_start is None:
            continue

        end_f = int(clip.end_frame_idx)
        if end_f <= goal_start:
            continue

        t_us = motion_timestamps_us
        start_s = _frame_time_s(t_us, goal_start)
        end_s = _frame_time_s(t_us, end_f)
        first_vis = bool(
            goal_start == c_starts[0]
            or _find_goal_fov_onset_frame(
                first_goal,
                c_starts[0],
                camera_position,
                camera_rotation,
                rgb_calib,
                max_lookback_frames=0,
                fov_margin_rad=fov_margin_rad,
            )
            == goal_start
        )

        sequences.append(
            MultiObjectVisibleGoalSequence(
                sequence_id=seq_id,
                start_frame_idx=int(goal_start),
                end_frame_idx=end_f,
                start_timestamp_us=int(t_us[goal_start]),
                end_timestamp_us=int(t_us[end_f]),
                start_time_s=start_s,
                end_time_s=end_s,
                total_duration_s=end_s - start_s,
                chain_object_uids=uids,
                chain_categories=tuple(categories),
                chain_instance_names=tuple(names),
                chain_hands=tuple(hands),
                chain_contact_starts=tuple(c_starts),
                chain_contact_ends=tuple(c_ends),
                chain_goal_world=tuple(goals),
                first_goal_visible_at_start=first_vis,
            )
        )
        seq_id += 1

    return sequences
