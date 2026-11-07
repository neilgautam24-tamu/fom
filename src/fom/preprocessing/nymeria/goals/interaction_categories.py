"""Categorize Nymeria contact / interaction sequences into training buckets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from fom.preprocessing.nymeria.contacts.detection import ContactEvent, ContactInterval
from fom.preprocessing.nymeria.goals.approach import validate_approach_segment
from fom.preprocessing.nymeria.goals.approach_sequences import (
    _find_fov_onset_frame,
    merge_object_contact_episodes,
)
from fom.preprocessing.nymeria.goals.visible_goal_sequences import index_contact_events
from fom.preprocessing.nymeria.io.aria_slam import RgbCameraCalibration
from fom.preprocessing.nymeria.io.objects import ObjectBBox3D

CategoryName = Literal[
    "cat1_fov_navigate",
    "cat2_navigation",
    "cat3_sustained_interaction",
    "cat4_multi_object_chain",
]

HAND_COLORS = {
    "left": (30, 120, 255),   # blue
    "right": (255, 60, 60),   # red
    "both": (200, 80, 200),
    "unknown": (180, 180, 180),
}


@dataclass(frozen=True)
class CategoryClip:
    """One exported clip with metadata shared across category types."""

    clip_id: int
    category: CategoryName
    start_frame_idx: int
    contact_start_frame_idx: int
    contact_end_frame_idx: int
    end_frame_idx: int
    object_uid: int
    category_name: str
    instance_name: str
    contact_hand: str
    start_timestamp_us: int
    contact_start_timestamp_us: int
    contact_end_timestamp_us: int
    end_timestamp_us: int
    start_time_s: float
    contact_start_time_s: float
    contact_end_time_s: float
    end_time_s: float
    total_duration_s: float
    # Optional chain metadata (cat4)
    chain_object_uids: tuple[int, ...] = ()
    chain_hands: tuple[str, ...] = ()
    chain_contact_starts: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["chain_object_uids"] = list(self.chain_object_uids)
        d["chain_hands"] = list(self.chain_hands)
        d["chain_contact_starts"] = list(self.chain_contact_starts)
        return d


def _frame_time_s(timestamps_us: np.ndarray, frame_idx: int) -> float:
    t0 = int(timestamps_us[0])
    return (int(timestamps_us[frame_idx]) - t0) * 1e-6


def _clip_from_interval(
    *,
    clip_id: int,
    category: CategoryName,
    start_f: int,
    contact_start: int,
    contact_end: int,
    end_f: int,
    interval: ContactInterval,
    timestamps_us: np.ndarray,
    hand: str | None = None,
    chain_object_uids: tuple[int, ...] = (),
    chain_hands: tuple[str, ...] = (),
    chain_contact_starts: tuple[int, ...] = (),
) -> CategoryClip:
    hand_str = hand or interval.hand
    t_us = timestamps_us
    start_s = _frame_time_s(t_us, start_f)
    c0_s = _frame_time_s(t_us, contact_start)
    c1_s = _frame_time_s(t_us, contact_end)
    end_s = _frame_time_s(t_us, end_f)
    return CategoryClip(
        clip_id=clip_id,
        category=category,
        start_frame_idx=int(start_f),
        contact_start_frame_idx=int(contact_start),
        contact_end_frame_idx=int(contact_end),
        end_frame_idx=int(end_f),
        object_uid=int(interval.object_uid),
        category_name=str(interval.category),
        instance_name=str(interval.instance_name),
        contact_hand=str(hand_str),
        start_timestamp_us=int(t_us[start_f]),
        contact_start_timestamp_us=int(t_us[contact_start]),
        contact_end_timestamp_us=int(t_us[contact_end]),
        end_timestamp_us=int(t_us[end_f]),
        start_time_s=start_s,
        contact_start_time_s=c0_s,
        contact_end_time_s=c1_s,
        end_time_s=end_s,
        total_duration_s=end_s - start_s,
        chain_object_uids=chain_object_uids,
        chain_hands=chain_hands,
        chain_contact_starts=chain_contact_starts,
    )


def _dominant_hand(events: list[ContactEvent]) -> str:
    if not events:
        return "unknown"
    hands = [e.hand for e in events]
    left = sum(1 for h in hands if h == "left")
    right = sum(1 for h in hands if h == "right")
    if left > 0 and right > 0:
        return "both"
    if left >= right:
        return "left"
    return "right"


def compute_navigation_stats(
    transl: np.ndarray,
    timestamps_us: np.ndarray,
    *,
    window_s: float = 3.0,
) -> tuple[float, list[tuple[int, int, float]]]:
    """Return (median displacement per window, list of (start, end, displacement))."""
    n = transl.shape[0]
    if n < 2:
        return 0.0, []

    ts = timestamps_us.astype(np.float64) * 1e-6
    windows: list[tuple[int, int, float]] = []
    i = 0
    while i < n:
        t0 = ts[i]
        j = i
        while j + 1 < n and (ts[j + 1] - t0) < window_s:
            j += 1
        if j > i:
            path = float(np.sum(np.linalg.norm(np.diff(transl[i : j + 1], axis=0), axis=1)))
            windows.append((i, j, path))
        i = max(i + 1, j)

    if not windows:
        return 0.0, []
    median = float(np.median([w[2] for w in windows]))
    return median, windows


def extract_cat1_fov_navigate(
    intervals: list[ContactInterval],
    motion_timestamps_us: np.ndarray,
    transl: np.ndarray,
    objects_by_uid: dict[int, ObjectBBox3D],
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    rgb_calib: RgbCameraCalibration,
    *,
    max_gap_frames: int = 2,
    max_lookback_s: float = 12.0,
    dt_s: float = 1.0 / 60.0,
    min_approach_s: float = 0.25,
    min_speed_m_s: float = 0.08,
    min_alignment: float = 0.35,
    min_valid_fraction: float = 0.45,
    fov_margin_rad: float = 0.05,
    contact_events: list[ContactEvent] | None = None,
) -> list[CategoryClip]:
    """Object visible in FOV, then navigate toward it until contact."""
    episodes = merge_object_contact_episodes(intervals, max_gap_frames=max_gap_frames)
    max_lb = max(1, int(max_lookback_s / max(dt_s, 1e-6)))
    min_app = max(1, int(min_approach_s / max(dt_s, 1e-6)))
    event_index = index_contact_events(contact_events or [])
    clips: list[CategoryClip] = []
    clip_id = 0

    for episode in episodes:
        obj = objects_by_uid.get(episode.object_uid)
        if obj is None:
            continue
        c0 = int(episode.start_frame_idx)
        c1 = int(episode.end_frame_idx)

        fov_start = _find_fov_onset_frame(
            contact_start_frame=c0,
            object_bbox=obj,
            camera_position=camera_position,
            camera_rotation=camera_rotation,
            rgb_calib=rgb_calib,
            max_lookback_frames=max_lb,
            fov_margin_rad=fov_margin_rad,
        )
        if fov_start is None or c0 - fov_start < min_app:
            continue

        ok, _, _ = validate_approach_segment(
            transl,
            motion_timestamps_us,
            obj.center_world,
            fov_start,
            max(fov_start, c0 - 1),
            min_speed_m_s=min_speed_m_s,
            min_alignment=min_alignment,
            min_valid_fraction=min_valid_fraction,
        )
        if not ok:
            continue

        frame_events = event_index.get((episode.object_uid, c0), [])
        hand = _dominant_hand(frame_events) if frame_events else "both"

        clips.append(
            _clip_from_interval(
                clip_id=clip_id,
                category="cat1_fov_navigate",
                start_f=fov_start,
                contact_start=c0,
                contact_end=c1,
                end_f=c1,
                interval=episode,
                timestamps_us=motion_timestamps_us,
                hand=hand,
            )
        )
        clip_id += 1
    return clips


def extract_cat2_navigation(
    motion_timestamps_us: np.ndarray,
    transl: np.ndarray,
    *,
    window_s: float = 3.0,
    nav_multiplier: float = 1.5,
    min_window_s: float = 2.0,
    merge_gap_frames: int = 30,
    dt_s: float = 1.0 / 60.0,
) -> list[CategoryClip]:
    """Pelvis travels much farther than episode median in a time window."""
    median, windows = compute_navigation_stats(
        transl, motion_timestamps_us, window_s=window_s
    )
    if median <= 1e-6:
        return []

    threshold = nav_multiplier * median
    hot = [(s, e, d) for s, e, d in windows if d >= threshold]
    if not hot:
        return []

    hot.sort(key=lambda x: x[0])
    merged: list[tuple[int, int, float]] = []
    for s, e, d in hot:
        if not merged or s > merged[-1][1] + merge_gap_frames:
            merged.append((s, e, d))
        else:
            ps, pe, pd = merged[-1]
            merged[-1] = (ps, max(pe, e), max(pd, d))

    clips: list[CategoryClip] = []
    min_frames = max(1, int(min_window_s / max(dt_s, 1e-6)))
    dummy = ContactInterval(
        hand="unknown",
        object_uid=-1,
        category="navigation",
        instance_name="navigation",
        start_frame_idx=0,
        end_frame_idx=0,
        start_timestamp_ns=0,
        end_timestamp_ns=0,
        start_time_s=0.0,
        end_time_s=0.0,
        num_frames=0,
    )

    for clip_id, (s, e, dist) in enumerate(merged):
        if e - s + 1 < min_frames:
            continue
        clips.append(
            _clip_from_interval(
                clip_id=clip_id,
                category="cat2_navigation",
                start_f=s,
                contact_start=s,
                contact_end=e,
                end_f=e,
                interval=dummy,
                timestamps_us=motion_timestamps_us,
                hand="unknown",
            )
        )
    return clips


def extract_cat3_sustained_interaction(
    intervals: list[ContactInterval],
    motion_timestamps_us: np.ndarray,
    *,
    min_interaction_s: float = 1.0,
    max_gap_frames: int = 2,
    dt_s: float = 1.0 / 60.0,
) -> list[CategoryClip]:
    """Object-level contact episode held for at least ``min_interaction_s``."""
    min_frames = max(1, int(min_interaction_s / max(dt_s, 1e-6)))
    episodes = merge_object_contact_episodes(intervals, max_gap_frames=max_gap_frames)
    clips: list[CategoryClip] = []
    clip_id = 0

    for ep in episodes:
        if ep.end_frame_idx - ep.start_frame_idx + 1 < min_frames:
            continue
        clips.append(
            _clip_from_interval(
                clip_id=clip_id,
                category="cat3_sustained_interaction",
                start_f=ep.start_frame_idx,
                contact_start=ep.start_frame_idx,
                contact_end=ep.end_frame_idx,
                end_f=ep.end_frame_idx,
                interval=ep,
                timestamps_us=motion_timestamps_us,
                hand=ep.hand,
            )
        )
        clip_id += 1
    return clips


def extract_cat4_multi_object_chain(
    intervals: list[ContactInterval],
    contact_events: list[ContactEvent],
    motion_timestamps_us: np.ndarray,
    objects_by_uid: dict[int, ObjectBBox3D],
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    rgb_calib: RgbCameraCalibration,
    *,
    max_gap_frames: int = 2,
    max_chain_span_s: float = 90.0,
    max_first_contact_s: float = 4.0,
    max_switch_gap_s: float = 25.0,
    min_switch_gap_s: float = 0.15,
    object_fov_lookback_s: float = 12.0,
    fov_margin_rad: float = 0.05,
    dt_s: float = 1.0 / 60.0,
    min_objects: int = 2,
    max_objects: int = 3,
) -> list[CategoryClip]:
    """Object1 in FOV -> brief contact -> switch to object2 (-> optional object3)."""
    episodes = merge_object_contact_episodes(intervals, max_gap_frames=max_gap_frames)
    episodes.sort(key=lambda e: e.start_frame_idx)
    event_index = index_contact_events(contact_events)
    obj_lb = max(1, int(object_fov_lookback_s / max(dt_s, 1e-6)))
    max_first_f = max(1, int(max_first_contact_s / max(dt_s, 1e-6)))
    max_switch_f = max(1, int(max_switch_gap_s / max(dt_s, 1e-6)))
    min_switch_f = max(1, int(min_switch_gap_s / max(dt_s, 1e-6)))
    max_span_f = max(1, int(max_chain_span_s / max(dt_s, 1e-6)))

    clips: list[CategoryClip] = []
    clip_id = 0
    n = len(episodes)

    for i in range(n):
        chain: list[ContactInterval] = [episodes[i]]
        for j in range(i + 1, min(n, i + max_objects)):
            prev, cur = chain[-1], episodes[j]
            if cur.object_uid == prev.object_uid:
                continue
            gap = cur.start_frame_idx - prev.end_frame_idx
            if gap < min_switch_f or gap > max_switch_f:
                break
            span = cur.end_frame_idx - chain[0].start_frame_idx
            if span > max_span_f:
                break
            chain.append(cur)
            if len(chain) >= min_objects:
                first = chain[0]
                obj1 = objects_by_uid.get(first.object_uid)
                if obj1 is None:
                    continue
                if first.end_frame_idx - first.start_frame_idx + 1 > max_first_f:
                    continue
                fov_start = _find_fov_onset_frame(
                    contact_start_frame=first.start_frame_idx,
                    object_bbox=obj1,
                    camera_position=camera_position,
                    camera_rotation=camera_rotation,
                    rgb_calib=rgb_calib,
                    max_lookback_frames=obj_lb,
                    fov_margin_rad=fov_margin_rad,
                )
                if fov_start is None:
                    continue

                uids = tuple(int(e.object_uid) for e in chain)
                hands: list[str] = []
                starts: list[int] = []
                for e in chain:
                    ev = event_index.get((e.object_uid, e.start_frame_idx), [])
                    hands.append(_dominant_hand(ev) if ev else e.hand)
                    starts.append(int(e.start_frame_idx))

                last = chain[-1]
                clips.append(
                    _clip_from_interval(
                        clip_id=clip_id,
                        category="cat4_multi_object_chain",
                        start_f=fov_start,
                        contact_start=first.start_frame_idx,
                        contact_end=last.end_frame_idx,
                        end_f=last.end_frame_idx,
                        interval=first,
                        timestamps_us=motion_timestamps_us,
                        hand=hands[0],
                        chain_object_uids=uids,
                        chain_hands=tuple(hands),
                        chain_contact_starts=tuple(starts),
                    )
                )
                clip_id += 1
            if len(chain) >= max_objects:
                break
    return clips


def per_frame_contact_hands(
    intervals: list[ContactInterval],
    start_frame: int,
    end_frame: int,
) -> list[str]:
    """For each global frame in [start, end], which hand(s) contact (or '')."""
    n = end_frame - start_frame + 1
    hands: list[str] = [""] * n
    for iv in intervals:
        s = max(iv.start_frame_idx, start_frame)
        e = min(iv.end_frame_idx, end_frame)
        for f in range(s, e + 1):
            local = f - start_frame
            if iv.hand == "left":
                hands[local] = "left" if hands[local] == "" else "both"
            elif iv.hand == "right":
                hands[local] = "right" if hands[local] == "" else "both"
    return hands
