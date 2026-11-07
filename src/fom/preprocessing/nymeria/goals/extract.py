#!/usr/bin/env python3
"""Extract approach->contact sequences cropped to first goal-in-FOV frame (t=0).

Each sequence starts when the first hand-object contact goal point enters the
head RGB field of view (not when the object bbox first appears). Uses a longer
goal lookback window to capture extended approach phases. Goal visibility is
tracked per frame; the goal may leave FOV during navigation.

Exports bundles with ``ego_rgb.npz``, ``ego_rgb_goal.npz``, and ``goal_mask.npz``.
RGB frames are rotated 90 deg clockwise before saving.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from fom.preprocessing.nymeria.body.smpl_motion import load_smpl_motion
from fom.preprocessing.nymeria.body.smplx_motion import load_smplx_motion
from fom.preprocessing.nymeria.contacts.detection import ContactEvent, ContactInterval
from fom.preprocessing.nymeria.goals.approach_sequences import save_sequences_csv
from fom.preprocessing.nymeria.goals.ego_projection import AriaFisheyeProjector
from fom.preprocessing.nymeria.goals.goal_sequence_export import export_all_visible_goal_sequences
from fom.preprocessing.nymeria.goals.trajectory import precompute_head_trajectory
from fom.preprocessing.nymeria.goals.visible_goal_sequences import (
    VisibleGoalSequence,
    extract_visible_goal_sequences,
    load_visible_goal_sequence_row,
)
from fom.preprocessing.nymeria.io.aria_slam import (
    AriaRgbReader,
    ClosedLoopTrajectory,
    compute_body_to_device_offset_us,
    load_rgb_camera_calibration,
    recording_paths,
)
from fom.preprocessing.nymeria.io.objects import load_object_bboxes
from fom.preprocessing.nymeria.io.playback import motion_timing_from_timestamps


def _load_intervals(path: Path) -> list[ContactInterval]:
    intervals: list[ContactInterval] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            intervals.append(
                ContactInterval(
                    hand=str(row["hand"]),
                    object_uid=int(row["object_uid"]),
                    category=str(row["category"]),
                    instance_name=str(row["instance_name"]),
                    start_frame_idx=int(row["start_frame_idx"]),
                    end_frame_idx=int(row["end_frame_idx"]),
                    start_timestamp_ns=int(row["start_timestamp_ns"]),
                    end_timestamp_ns=int(row["end_timestamp_ns"]),
                    start_time_s=float(row["start_time_s"]),
                    end_time_s=float(row["end_time_s"]),
                    num_frames=int(row["num_frames"]),
                )
            )
    return intervals


def _load_contact_events(path: Path) -> list[ContactEvent]:
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


def _load_visible_goal_json(path: Path) -> list[VisibleGoalSequence]:
    data = json.loads(path.read_text())
    return [load_visible_goal_sequence_row(row) for row in data["sequences"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True, help="<episode>/body/xdata_smplx_neutral.npz")
    parser.add_argument("--sequence-dir", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, required=True, help="Dir with contact CSVs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device", default="head", choices=["head"])
    parser.add_argument("--max-gap-frames", type=int, default=2)
    parser.add_argument(
        "--max-goal-lookback-s",
        type=float,
        default=25.0,
        help="Walk back this far from contact to find when the goal first enters FOV",
    )
    parser.add_argument(
        "--object-fov-lookback-s",
        type=float,
        default=12.0,
        help="Lookback used only to record when the object bbox first entered FOV",
    )
    parser.add_argument("--min-approach-s", type=float, default=0.25)
    parser.add_argument("--min-speed", type=float, default=0.08)
    parser.add_argument("--min-alignment", type=float, default=0.35)
    parser.add_argument("--min-valid-fraction", type=float, default=0.45)
    parser.add_argument("--fov-margin-deg", type=float, default=3.0)
    parser.add_argument("--skip-categories", nargs="*", default=["Stairs"])
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse existing visible_goal_sequences.json in the output dir",
    )
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--rgb-stride", type=int, default=4)
    parser.add_argument("--max-rgb-edge", type=int, default=512)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--dot-radius", type=int, default=11)
    parser.add_argument("--mask-radius", type=int, default=11)
    args = parser.parse_args()

    seq_dir = args.sequence_dir or args.npz.resolve().parent.parent
    in_dir = args.input_dir
    out_dir = args.output_dir or in_dir
    bundle_root = out_dir / "visible_goal_sequence_bundles"
    out_dir.mkdir(parents=True, exist_ok=True)

    intervals_path = in_dir / "hand_object_contact_intervals.csv"
    contacts_path = in_dir / "hand_object_contacts.csv"
    if not intervals_path.is_file():
        raise FileNotFoundError(f"Missing {intervals_path}")
    if not contacts_path.is_file():
        raise FileNotFoundError(f"Missing {contacts_path}")

    print(f"[motion] loading SMPL-X stride={args.stride}")
    motion = load_smplx_motion(args.npz, args.stride)
    smpl_npz = seq_dir / "body" / "xdata_smpl_neutral.npz"
    smpl_motion = None
    if smpl_npz.is_file():
        print(f"[motion] loading SMPL stride={args.stride}")
        smpl_motion = load_smpl_motion(smpl_npz, args.stride)
    timing = motion_timing_from_timestamps(motion["timestamps"])

    paths = recording_paths(seq_dir, args.device)
    trajectory = ClosedLoopTrajectory.from_csv(paths["closed_loop_csv"])
    rgb_calib = load_rgb_camera_calibration(paths["calibration_jsonl"])
    offset_us = compute_body_to_device_offset_us(
        paths["motion_vrs"], int(motion["timestamps"][0])
    )

    json_path = out_dir / "visible_goal_sequences.json"
    head_traj = None
    objects_by_uid: dict | None = None

    if args.skip_extract and json_path.is_file():
        print(f"[sequences] loading existing {json_path.name}")
        sequences = _load_visible_goal_json(json_path)
    else:
        print(f"[intervals] loading {intervals_path.name}")
        intervals = _load_intervals(intervals_path)
        print(f"[contacts] loading {contacts_path.name}")
        contact_events = _load_contact_events(contacts_path)

        objects = load_object_bboxes(seq_dir, skip_floor_wall=True)
        if args.skip_categories:
            skip = {c.lower() for c in args.skip_categories}
            objects = [o for o in objects if o.category.lower() not in skip]
        objects_by_uid = {o.object_uid: o for o in objects}

        print("[trajectory] precomputing head poses ...")
        t0 = time.time()
        head_traj = precompute_head_trajectory(
            motion["timestamps"], trajectory, rgb_calib, offset_us
        )
        print(f"[trajectory] done in {time.time() - t0:.1f}s")

        print("[sequences] extracting goal-visible approach->contact sequences ...")
        fov_margin_rad = float(np.deg2rad(args.fov_margin_deg))
        sequences = extract_visible_goal_sequences(
            intervals,
            contact_events,
            motion["timestamps"],
            motion["transl"],
            objects_by_uid,
            head_traj.camera_position,
            head_traj.camera_rotation,
            rgb_calib,
            max_gap_frames=int(args.max_gap_frames),
            max_goal_lookback_s=float(args.max_goal_lookback_s),
            dt_s=timing.dt_s,
            min_approach_s=float(args.min_approach_s),
            min_speed_m_s=float(args.min_speed),
            min_alignment=float(args.min_alignment),
            min_valid_fraction=float(args.min_valid_fraction),
            fov_margin_rad=fov_margin_rad,
            object_fov_lookback_s=float(args.object_fov_lookback_s),
        )
        cropped_later = sum(
            1
            for s in sequences
            if s.object_fov_start_frame_idx >= 0
            and s.start_frame_idx > s.object_fov_start_frame_idx
        )
        lost_during = sum(1 for s in sequences if s.goal_lost_during_approach)
        durations = [s.total_duration_s for s in sequences]
        print(
            f"[sequences] {len(sequences)} sequences | "
            f"{cropped_later} cropped after object-FOV onset | "
            f"{lost_during} with goal leaving FOV during approach"
        )
        if durations:
            print(
                f"[sequences] duration range {min(durations):.1f}s - {max(durations):.1f}s "
                f"(median {float(np.median(durations)):.1f}s)"
            )

        save_sequences_csv(sequences, out_dir / "visible_goal_sequences.csv")
        summary = {
            "num_visible_goal_sequences": len(sequences),
            "max_goal_lookback_s": float(args.max_goal_lookback_s),
            "sequences": [s.to_dict() for s in sequences],
        }
        json_path.write_text(json.dumps(summary, indent=2))
        print(f"[sequences] saved -> {json_path}")

    if args.max_sequences > 0:
        sequences = sequences[: int(args.max_sequences)]

    if objects_by_uid is None:
        objects = load_object_bboxes(seq_dir, skip_floor_wall=True)
        objects_by_uid = {o.object_uid: o for o in objects}

    if head_traj is None:
        print("[trajectory] precomputing head poses ...")
        t0 = time.time()
        head_traj = precompute_head_trajectory(
            motion["timestamps"], trajectory, rgb_calib, offset_us
        )
        print(f"[trajectory] done in {time.time() - t0:.1f}s")

    rgb_reader = None
    projector = None
    if not args.no_rgb:
        print(f"[rgb] opening {paths['data_vrs'].name}")
        rgb_reader = AriaRgbReader(paths["data_vrs"])
        projector = AriaFisheyeProjector(paths["data_vrs"])

    print(f"[export] writing {len(sequences)} bundles -> {bundle_root}")
    t0 = time.time()
    exported = export_all_visible_goal_sequences(
        sequences,
        bundle_root,
        motion,
        head_traj,
        rgb_calib,
        objects_by_uid,
        rgb_reader,
        projector,
        smpl_motion=smpl_motion,
        max_rgb_edge=int(args.max_rgb_edge),
        rgb_stride=int(args.rgb_stride),
        fov_margin_rad=float(np.deg2rad(args.fov_margin_deg)),
    )
    manifest = {
        "num_sequences": len(exported),
        "bundle_root": str(bundle_root),
        "stride": int(args.stride),
        "rgb_stride": int(args.rgb_stride),
        "max_rgb_edge": int(args.max_rgb_edge),
        "sequences": [p.name for p in exported],
    }
    (bundle_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[export] done {len(exported)} bundles in {time.time() - t0:.1f}s")
    print(f"[export] manifest -> {bundle_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
