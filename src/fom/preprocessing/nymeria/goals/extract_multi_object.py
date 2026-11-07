#!/usr/bin/env python3
"""Extract multi-object visible-goal chains (goal switches after each contact).

Example
-------
    python -m fom.preprocessing.nymeria.goals.extract_multi_object \\
        --url-json <urls.json> --episode-name <episode> \\
        --sequence-dir <raw>/<episode> --input-dir <contacts>/<episode> --output-dir <out>/<episode>
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
from fom.preprocessing.nymeria.goals.ego_projection import AriaFisheyeProjector
from fom.preprocessing.nymeria.goals.multi_object_export import export_all_multi_object_sequences
from fom.preprocessing.nymeria.goals.multi_object_sequences import (
    extract_multi_object_visible_goal_sequences,
)
from fom.preprocessing.nymeria.goals.trajectory import precompute_head_trajectory
from fom.preprocessing.nymeria.io.aria_slam import (
    AriaRgbReader,
    ClosedLoopTrajectory,
    compute_body_to_device_offset_us,
    load_rgb_camera_calibration,
    recording_paths,
)
from fom.preprocessing.nymeria.io.objects import load_object_bboxes
from fom.preprocessing.nymeria.io.paths import default_smplx_npz
from fom.preprocessing.nymeria.io.playback import motion_timing_from_timestamps


def load_episodes_from_url_json(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    return sorted(data["sequences"].keys())


def _load_intervals(path: Path) -> list[ContactInterval]:
    out: list[ContactInterval] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out.append(
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
    return out


def _load_contact_events(path: Path) -> list[ContactEvent]:
    out: list[ContactEvent] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out.append(
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
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-json", type=Path, required=True)
    parser.add_argument("--episode-name", type=str, default=None)
    parser.add_argument("--sequence-dir", type=Path, required=True, help="Raw Nymeria-Plus episode dir")
    parser.add_argument("--input-dir", type=Path, required=True, help="Dir with contact CSVs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--min-objects", type=int, default=2)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--max-goal-lookback-s", type=float, default=25.0)
    parser.add_argument("--fov-margin-deg", type=float, default=3.0)
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--rgb-stride", type=int, default=4)
    parser.add_argument("--max-rgb-edge", type=int, default=512)
    parser.add_argument("--max-sequences", type=int, default=0)
    args = parser.parse_args()

    episodes = load_episodes_from_url_json(args.url_json.resolve())
    if args.episode_name:
        episode = args.episode_name
        if episode not in episodes:
            raise SystemExit(f"Episode {episode!r} not in {args.url_json}")
    else:
        episode = episodes[0]

    seq_dir = args.sequence_dir
    in_dir = args.input_dir
    out_dir = args.output_dir
    bundle_root = out_dir / "multi_object_visible_goal_bundles"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = args.npz or default_smplx_npz(seq_dir / "body")
    if not npz.is_file():
        npz = seq_dir / "body" / "xdata_smplx_neutral.npz"

    intervals_path = in_dir / "hand_object_contact_intervals.csv"
    contacts_path = in_dir / "hand_object_contacts.csv"
    if not intervals_path.is_file():
        raise FileNotFoundError(f"Missing {intervals_path}")

    print(f"[episode] {episode}")
    print(f"[motion] loading SMPL-X stride={args.stride}")
    motion = load_smplx_motion(npz, args.stride)
    smpl_npz = seq_dir / "body" / "xdata_smpl_neutral.npz"
    smpl_motion = load_smpl_motion(smpl_npz, args.stride) if smpl_npz.is_file() else None
    timing = motion_timing_from_timestamps(motion["timestamps"])

    paths = recording_paths(seq_dir, "head")
    trajectory = ClosedLoopTrajectory.from_csv(paths["closed_loop_csv"])
    rgb_calib = load_rgb_camera_calibration(paths["calibration_jsonl"])
    offset_us = compute_body_to_device_offset_us(paths["motion_vrs"], int(motion["timestamps"][0]))

    print("[trajectory] precomputing head poses ...")
    t0 = time.time()
    head_traj = precompute_head_trajectory(motion["timestamps"], trajectory, rgb_calib, offset_us)
    print(f"[trajectory] done in {time.time() - t0:.1f}s")

    intervals = _load_intervals(intervals_path)
    contact_events = _load_contact_events(contacts_path) if contacts_path.is_file() else []
    objects = load_object_bboxes(seq_dir, skip_floor_wall=True)
    objects_by_uid = {o.object_uid: o for o in objects}

    fov_margin_rad = float(np.deg2rad(args.fov_margin_deg))
    print(
        f"[sequences] extracting multi-object chains "
        f"(objects {args.min_objects}-{args.max_objects}) ..."
    )
    sequences = extract_multi_object_visible_goal_sequences(
        intervals,
        contact_events,
        motion["timestamps"],
        objects_by_uid,
        head_traj.camera_position,
        head_traj.camera_rotation,
        rgb_calib,
        max_goal_lookback_s=float(args.max_goal_lookback_s),
        fov_margin_rad=fov_margin_rad,
        dt_s=timing.dt_s,
        min_objects=int(args.min_objects),
        max_objects=int(args.max_objects),
    )

    by_len: dict[int, int] = {}
    for s in sequences:
        by_len[len(s.chain_object_uids)] = by_len.get(len(s.chain_object_uids), 0) + 1
    print(f"[sequences] found {len(sequences)} | by chain length: {dict(sorted(by_len.items()))}")

    if args.max_sequences > 0:
        sequences = sequences[: int(args.max_sequences)]

    summary = {
        "episode": episode,
        "num_multi_object_sequences": len(sequences),
        "min_objects": int(args.min_objects),
        "max_objects": int(args.max_objects),
        "sequences": [s.to_dict() for s in sequences],
    }
    (out_dir / "multi_object_visible_goal_sequences.json").write_text(json.dumps(summary, indent=2))

    rgb_reader = None
    projector = None
    if not args.no_rgb:
        print(f"[rgb] opening {paths['data_vrs'].name}")
        rgb_reader = AriaRgbReader(paths["data_vrs"])
        projector = AriaFisheyeProjector(paths["data_vrs"])

    print(f"[export] writing {len(sequences)} bundles -> {bundle_root}")
    t0 = time.time()
    exported = export_all_multi_object_sequences(
        sequences,
        bundle_root,
        motion,
        head_traj,
        rgb_calib,
        objects_by_uid,
        intervals,
        rgb_reader,
        projector,
        smpl_motion=smpl_motion,
        max_rgb_edge=int(args.max_rgb_edge),
        rgb_stride=int(args.rgb_stride),
        fov_margin_rad=fov_margin_rad,
    )
    manifest = {
        "episode": episode,
        "num_sequences": len(exported),
        "bundle_root": str(bundle_root),
        "stride": int(args.stride),
        "sequences": [p.name for p in exported],
    }
    (bundle_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[export] done {len(exported)} bundles in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
