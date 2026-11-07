"""Extract approach->contact sequences (object in head FOV, walk towards, until contact ends)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fom.preprocessing.nymeria.contacts.detection import ContactInterval
from fom.preprocessing.nymeria.goals.approach import validate_approach_segment
from fom.preprocessing.nymeria.goals.fov import object_in_head_fov
from fom.preprocessing.nymeria.io.aria_slam import RgbCameraCalibration
from fom.preprocessing.nymeria.io.objects import ObjectBBox3D


@dataclass(frozen=True)
class ApproachContactSequence:
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

    def to_dict(self) -> dict:
        return asdict(self)


def merge_object_contact_episodes(
    intervals: list[ContactInterval],
    *,
    max_gap_frames: int = 2,
) -> list[ContactInterval]:
    """Merge per-hand intervals into object-level contact episodes."""
    by_uid: dict[int, list[ContactInterval]] = {}
    for interval in intervals:
        by_uid.setdefault(interval.object_uid, []).append(interval)

    merged: list[ContactInterval] = []
    for uid, segments in by_uid.items():
        segments.sort(key=lambda s: s.start_frame_idx)
        runs: list[tuple[int, int, ContactInterval]] = []
        for seg in segments:
            if not runs or seg.start_frame_idx > runs[-1][1] + max_gap_frames + 1:
                runs.append((seg.start_frame_idx, seg.end_frame_idx, seg))
            else:
                s0, e0, meta = runs[-1]
                runs[-1] = (s0, max(e0, seg.end_frame_idx), meta)

        for start_f, end_f, meta in runs:
            merged.append(
                ContactInterval(
                    hand="both",
                    object_uid=uid,
                    category=meta.category,
                    instance_name=meta.instance_name,
                    start_frame_idx=start_f,
                    end_frame_idx=end_f,
                    start_timestamp_ns=meta.start_timestamp_ns,
                    end_timestamp_ns=meta.end_timestamp_ns,
                    start_time_s=meta.start_time_s,
                    end_time_s=meta.end_time_s,
                    num_frames=end_f - start_f + 1,
                )
            )
    merged.sort(key=lambda s: (s.start_frame_idx, s.object_uid))
    return merged


def _frame_time_s(timestamps_us: np.ndarray, frame_idx: int) -> float:
    t0 = int(timestamps_us[0])
    return (int(timestamps_us[frame_idx]) - t0) * 1e-6


def _find_fov_onset_frame(
    *,
    contact_start_frame: int,
    object_bbox: ObjectBBox3D,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    rgb_calib: RgbCameraCalibration,
    max_lookback_frames: int,
    fov_margin_rad: float,
) -> int | None:
    """Walk backward from contact to when the object first enters head FOV."""
    earliest = max(0, contact_start_frame - max_lookback_frames)
    fov_onset: int | None = None

    for frame_idx in range(contact_start_frame, earliest - 1, -1):
        in_fov = object_in_head_fov(
            object_bbox.center_world,
            object_bbox.dimensions,
            object_bbox.wxyz,
            camera_position[frame_idx],
            camera_rotation[frame_idx],
            fov_y_rad=rgb_calib.fov_y_rad,
            aspect=rgb_calib.aspect,
            margin_rad=fov_margin_rad,
        )
        if in_fov:
            fov_onset = frame_idx
        elif fov_onset is not None:
            break

    return fov_onset


def extract_approach_contact_sequences(
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
) -> list[ApproachContactSequence]:
    """Build sequences: FOV onset -> walk towards object -> contact -> contact ends."""
    episodes = merge_object_contact_episodes(intervals, max_gap_frames=max_gap_frames)
    max_lookback_frames = max(1, int(max_lookback_s / max(dt_s, 1e-6)))
    min_approach_frames = max(1, int(min_approach_s / max(dt_s, 1e-6)))

    sequences: list[ApproachContactSequence] = []
    seq_id = 0

    for episode in episodes:
        obj = objects_by_uid.get(episode.object_uid)
        if obj is None:
            continue

        contact_start = episode.start_frame_idx
        contact_end = episode.end_frame_idx

        fov_start = _find_fov_onset_frame(
            contact_start_frame=contact_start,
            object_bbox=obj,
            camera_position=camera_position,
            camera_rotation=camera_rotation,
            rgb_calib=rgb_calib,
            max_lookback_frames=max_lookback_frames,
            fov_margin_rad=fov_margin_rad,
        )
        if fov_start is None:
            continue

        approach_end = max(fov_start, contact_start - 1)
        if contact_start - fov_start < min_approach_frames:
            continue

        ok, med_speed, med_align = validate_approach_segment(
            transl,
            motion_timestamps_us,
            obj.center_world,
            fov_start,
            approach_end if approach_end >= fov_start else contact_start,
            min_speed_m_s=min_speed_m_s,
            min_alignment=min_alignment,
            min_valid_fraction=min_valid_fraction,
        )
        if not ok:
            continue

        # Recompute valid fraction for metadata
        valid = 0
        total = max(contact_start - fov_start, 1)
        for frame_idx in range(fov_start, contact_start):
            from fom.preprocessing.nymeria.goals.approach import approach_alignment, pelvis_velocity

            vel = pelvis_velocity(transl, motion_timestamps_us, frame_idx)
            speed, cos_align = approach_alignment(transl[frame_idx], obj.center_world, vel)
            if speed >= min_speed_m_s and cos_align >= min_alignment:
                valid += 1
        valid_fraction = valid / total

        t_us = motion_timestamps_us
        start_s = _frame_time_s(t_us, fov_start)
        contact_start_s = _frame_time_s(t_us, contact_start)
        contact_end_s = _frame_time_s(t_us, contact_end)

        sequences.append(
            ApproachContactSequence(
                sequence_id=seq_id,
                object_uid=episode.object_uid,
                category=episode.category,
                instance_name=episode.instance_name,
                start_frame_idx=fov_start,
                contact_start_frame_idx=contact_start,
                contact_end_frame_idx=contact_end,
                end_frame_idx=contact_end,
                start_timestamp_us=int(t_us[fov_start]),
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
            )
        )
        seq_id += 1

    return sequences


def save_sequences_csv(sequences: list[ApproachContactSequence], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not sequences:
        path.write_text("")
        return
    fieldnames = list(sequences[0].to_dict().keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seq in sequences:
            writer.writerow(seq.to_dict())
