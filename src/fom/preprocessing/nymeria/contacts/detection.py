"""Detect hand-object contacts from SMPL-X hand points and Boxy OBBs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fom.preprocessing.nymeria.body.smplx_hands import NUM_HAND_JOINT_SAMPLES
from fom.preprocessing.nymeria.contacts.geometry import closest_point_on_obb, points_to_obb_distance


@dataclass(frozen=True)
class ContactEvent:
    frame_idx: int
    timestamp_ns: int
    time_s: float
    hand: str
    object_uid: int
    category: str
    instance_name: str
    distance_m: float
    hand_point_x: float
    hand_point_y: float
    hand_point_z: float
    object_point_x: float
    object_point_y: float
    object_point_z: float
    sample_idx: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ContactInterval:
    hand: str
    object_uid: int
    category: str
    instance_name: str
    start_frame_idx: int
    end_frame_idx: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    start_time_s: float
    end_time_s: float
    num_frames: int

    def to_dict(self) -> dict:
        return asdict(self)


def detect_hand_object_contacts(
    hand_points: dict[str, np.ndarray],
    timestamps_ns: np.ndarray,
    objects: Iterable,
    *,
    contact_threshold_m: float = 0.03,
    bbox_shrink: float = 0.0,
    joints_only: bool = True,
) -> list[ContactEvent]:
    """Find all frame/hand/object contacts below ``contact_threshold_m``.

    Parameters
    ----------
    hand_points
        ``left_points`` / ``right_points`` arrays with shape ``(N, P, 3)``.
    objects
        Iterable of ``ObjectBBox3D`` from ``fom.preprocessing.nymeria.io.objects``.
    bbox_shrink
        Uniform shrink applied to each box dimension (m) before testing contact.
        Positive values tighten loose Boxy annotations.
    """
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    t0 = int(ts[0]) if ts.size else 0
    rel_s = (ts - t0) * 1e-6

    obj_list = list(objects)
    events: list[ContactEvent] = []

    for hand_name, points in hand_points.items():
        points = np.asarray(points, dtype=np.float32)
        if joints_only:
            points = points[:, :NUM_HAND_JOINT_SAMPLES, :]
        n_frames = points.shape[0]
        for frame_idx in range(n_frames):
            pts = points[frame_idx]
            for obj in obj_list:
                dims = np.asarray(obj.dimensions, dtype=np.float64)
                if bbox_shrink > 0:
                    dims = np.maximum(dims - 2.0 * bbox_shrink, 0.05)

                dists = points_to_obb_distance(
                    pts,
                    obj.center_world,
                    dims,
                    obj.wxyz,
                )
                best_idx = int(np.argmin(dists))
                best_dist = float(dists[best_idx])
                if best_dist > contact_threshold_m:
                    continue

                hand_pt = pts[best_idx].astype(np.float64)
                obj_pt = closest_point_on_obb(
                    hand_pt,
                    obj.center_world,
                    dims,
                    obj.wxyz,
                )
                events.append(
                    ContactEvent(
                        frame_idx=frame_idx,
                        timestamp_ns=int(ts[frame_idx]),
                        time_s=float(rel_s[frame_idx]),
                        hand=hand_name,
                        object_uid=int(obj.object_uid),
                        category=str(obj.category),
                        instance_name=str(obj.instance_name),
                        distance_m=best_dist,
                        hand_point_x=float(hand_pt[0]),
                        hand_point_y=float(hand_pt[1]),
                        hand_point_z=float(hand_pt[2]),
                        object_point_x=float(obj_pt[0]),
                        object_point_y=float(obj_pt[1]),
                        object_point_z=float(obj_pt[2]),
                        sample_idx=best_idx,
                    )
                )
    events.sort(key=lambda e: (e.timestamp_ns, e.hand, e.object_uid))
    return events


def group_contact_intervals(
    events: list[ContactEvent],
    *,
    max_gap_frames: int = 2,
) -> list[ContactInterval]:
    """Merge per-frame contacts into continuous hand-object intervals."""
    if not events:
        return []

    by_key: dict[tuple[str, int], list[ContactEvent]] = {}
    for event in events:
        by_key.setdefault((event.hand, event.object_uid), []).append(event)

    intervals: list[ContactInterval] = []
    for group in by_key.values():
        group.sort(key=lambda e: e.frame_idx)
        run_start = group[0]
        run_end = group[0]
        meta = group[0]
        for event in group[1:]:
            if event.frame_idx - run_end.frame_idx <= max_gap_frames + 1:
                run_end = event
                continue
            intervals.append(_make_interval(run_start, run_end, meta))
            run_start = event
            run_end = event
            meta = event
        intervals.append(_make_interval(run_start, run_end, meta))

    intervals.sort(key=lambda x: (x.start_timestamp_ns, x.hand, x.object_uid))
    return intervals


def _make_interval(start: ContactEvent, end: ContactEvent, meta: ContactEvent) -> ContactInterval:
    return ContactInterval(
        hand=start.hand,
        object_uid=start.object_uid,
        category=meta.category,
        instance_name=meta.instance_name,
        start_frame_idx=start.frame_idx,
        end_frame_idx=end.frame_idx,
        start_timestamp_ns=start.timestamp_ns,
        end_timestamp_ns=end.timestamp_ns,
        start_time_s=start.time_s,
        end_time_s=end.time_s,
        num_frames=end.frame_idx - start.frame_idx + 1,
    )


def contact_timestamps(events: list[ContactEvent]) -> np.ndarray:
    """Unique body timestamps (ns) where any hand contacts any object."""
    if not events:
        return np.array([], dtype=np.int64)
    return np.unique(np.array([e.timestamp_ns for e in events], dtype=np.int64))


def count_human_contact_episodes(
    events: list[ContactEvent],
    *,
    max_gap_frames: int = 2,
) -> int:
    """Count episode-level contacts: any hand touching any object."""
    if not events:
        return 0
    frames = sorted({e.frame_idx for e in events})
    episodes = 1
    for prev, cur in zip(frames, frames[1:]):
        if cur - prev > max_gap_frames + 1:
            episodes += 1
    return episodes


def count_object_contact_episodes(
    intervals: list[ContactInterval],
    *,
    max_gap_frames: int = 2,
) -> dict[int, dict]:
    """Count touch episodes per object, merging left/right hand intervals."""
    by_uid: dict[int, list[tuple[int, int, ContactInterval]]] = {}
    for interval in intervals:
        by_uid.setdefault(interval.object_uid, []).append(
            (interval.start_frame_idx, interval.end_frame_idx, interval)
        )

    out: dict[int, dict] = {}
    for uid, segments in by_uid.items():
        segments.sort(key=lambda x: x[0])
        merged: list[tuple[int, int, ContactInterval]] = []
        for start, end, meta in segments:
            if not merged or start > merged[-1][1] + max_gap_frames + 1:
                merged.append((start, end, meta))
            else:
                prev_start, prev_end, prev_meta = merged[-1]
                merged[-1] = (prev_start, max(prev_end, end), prev_meta)

        meta = merged[0][2]
        out[uid] = {
            "object_uid": uid,
            "category": meta.category,
            "instance_name": meta.instance_name,
            "num_touch_episodes": len(merged),
            "num_hand_object_intervals": len(segments),
        }
    return out


def summarize_contact_counts(
    events: list[ContactEvent],
    intervals: list[ContactInterval],
    *,
    max_gap_frames: int = 2,
) -> dict:
    """Aggregate contact statistics for a full episode."""
    per_object = count_object_contact_episodes(intervals, max_gap_frames=max_gap_frames)
    by_object = sorted(per_object.values(), key=lambda x: (-x["num_touch_episodes"], x["object_uid"]))
    total_object_episodes = sum(x["num_touch_episodes"] for x in by_object)

    return {
        "num_contact_frames": len({e.frame_idx for e in events}),
        "num_unique_contact_timestamps": int(contact_timestamps(events).size),
        "num_hand_object_intervals": len(intervals),
        "num_human_contact_episodes": count_human_contact_episodes(
            events, max_gap_frames=max_gap_frames
        ),
        "num_object_touch_episodes_total": total_object_episodes,
        "num_unique_objects_contacted": len(per_object),
        "max_gap_frames": int(max_gap_frames),
        "per_object": by_object,
    }


def save_contact_counts_json(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(summary, indent=2))


def save_contact_counts_csv(per_object: list[dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not per_object:
        path.write_text("")
        return
    fieldnames = [
        "object_uid",
        "category",
        "instance_name",
        "num_touch_episodes",
        "num_hand_object_intervals",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_object:
            writer.writerow({k: row[k] for k in fieldnames})


def save_contacts_csv(events: list[ContactEvent], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not events:
        path.write_text("")
        return
    fieldnames = list(events[0].to_dict().keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(event.to_dict())


def save_intervals_csv(intervals: list[ContactInterval], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not intervals:
        path.write_text("")
        return
    fieldnames = list(intervals[0].to_dict().keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for interval in intervals:
            writer.writerow(interval.to_dict())
