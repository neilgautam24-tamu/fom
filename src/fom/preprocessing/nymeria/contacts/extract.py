#!/usr/bin/env python3
"""Extract hand-object contact timestamps from Nymeria Plus SMPL-X motion.

Uses SMPL-X as the human motion source (``xdata_smplx_neutral.npz``) and Nymeria
Plus Boxy 3D object bounding boxes in the shared world frame. Contact is
detected when any SMPL-X hand sample point lies within ``--threshold`` meters of
an object OBB (negative distance = inside the box).

Example
-------
    python -m fom.preprocessing.nymeria.contacts.extract \\
        --npz <episode>/body/xdata_smplx_neutral.npz --output-dir <work>/<episode> \\
        --stride 4 --threshold 0.03
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fom.preprocessing.nymeria.body.smplx_hands import precompute_hand_points
from fom.preprocessing.nymeria.body.smplx_motion import load_smplx_motion
from fom.preprocessing.nymeria.contacts.detection import (
    contact_timestamps,
    group_contact_intervals,
    save_contact_counts_csv,
    save_contact_counts_json,
    save_contacts_csv,
    save_intervals_csv,
    summarize_contact_counts,
)
from fom.preprocessing.nymeria.contacts.query import build_object_array, detect_contacts_fast
from fom.preprocessing.nymeria.io.objects import load_object_bboxes
from fom.preprocessing.nymeria.io.paths import smplx_model_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True, help="<episode>/body/xdata_smplx_neutral.npz")
    parser.add_argument("--sequence-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--device-torch", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--stride", type=int, default=1, help="Temporal subsampling of SMPL-X motion")
    parser.add_argument("--gpu-batch", type=int, default=256, help="SMPL-X GPU batch size")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="Contact distance threshold in meters (<=0 counts as touching)",
    )
    parser.add_argument(
        "--bbox-shrink",
        type=float,
        default=0.0,
        help="Uniform shrink applied to object box dimensions before testing (m)",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=2,
        help="Max frame gap when merging contact intervals",
    )
    parser.add_argument(
        "--no-hand-vertices",
        action="store_true",
        help="Use wrist/finger joints only (16 pts/hand; faster contact pass)",
    )
    parser.add_argument(
        "--show-floor-wall",
        action="store_true",
        help="Include Floor/Wall boxes (excluded by default)",
    )
    parser.add_argument(
        "--skip-categories",
        nargs="*",
        default=["Stairs"],
        help="Object categories to ignore",
    )
    args = parser.parse_args()

    if not args.npz.is_file():
        raise FileNotFoundError(
            f"SMPL-X motion not found: {args.npz}\n"
            "Run python -m fom.preprocessing.nymeria.body.smpl_to_smplx first."
        )

    seq_dir = args.sequence_dir or args.npz.resolve().parent.parent
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import smplx
    import torch

    device = "cuda" if (args.device_torch == "cuda" and torch.cuda.is_available()) else "cpu"
    torch_device = torch.device(device)

    print(f"[motion] loading SMPL-X {args.npz}")
    motion = load_smplx_motion(args.npz, args.stride)
    n = motion["num_frames"]
    print(f"[motion] {n:,} frames (stride={args.stride})")

    print(f"[objects] loading Boxy OBBs from {seq_dir / 'objects/boxy'}")
    objects = load_object_bboxes(
        seq_dir,
        skip_floor_wall=not args.show_floor_wall,
    )
    if args.skip_categories:
        skip = {c.lower() for c in args.skip_categories}
        objects = [o for o in objects if o.category.lower() not in skip]
    cats = sorted({o.category for o in objects})
    print(f"[objects] {len(objects)} boxes | categories: {', '.join(cats)}")

    gpu_batch = max(1, int(args.gpu_batch))
    print(f"[smplx] loading model on {device} (batch={gpu_batch})")
    model = smplx.create(
        str(smplx_model_file(args.gender)),
        model_type="smplx",
        gender=args.gender,
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=gpu_batch,
    ).to(torch_device)
    model.eval()

    t0 = time.time()
    hand_points = precompute_hand_points(
        model,
        motion,
        torch_device,
        batch_size=gpu_batch,
        use_vertices=not args.no_hand_vertices,
    )
    del model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"[smplx] hand points left={hand_points['left'].shape} "
        f"right={hand_points['right'].shape} computed in {time.time() - t0:.1f}s"
    )

    print(f"[contact] detecting contacts (threshold={args.threshold} m)")
    obb_array = build_object_array(objects, bbox_shrink=float(args.bbox_shrink))
    events = detect_contacts_fast(
        hand_points,
        motion["timestamps"],
        obb_array,
        contact_threshold_m=float(args.threshold),
    )
    intervals = group_contact_intervals(events, max_gap_frames=int(args.max_gap_frames))
    timestamps = contact_timestamps(events)

    contacts_csv = out_dir / "hand_object_contacts.csv"
    intervals_csv = out_dir / "hand_object_contact_intervals.csv"
    timestamps_npy = out_dir / "contact_timestamps_ns.npy"
    counts_json = out_dir / "contact_episode_counts.json"
    counts_csv = out_dir / "contact_episode_counts_by_object.csv"
    summary_json = out_dir / "contact_summary.json"

    save_contacts_csv(events, contacts_csv)
    save_intervals_csv(intervals, intervals_csv)
    np.save(timestamps_npy, timestamps)

    counts = summarize_contact_counts(
        events,
        intervals,
        max_gap_frames=int(args.max_gap_frames),
    )
    save_contact_counts_json(counts, counts_json)
    save_contact_counts_csv(counts["per_object"], counts_csv)

    hands_per_ts: dict[int, set[str]] = {}
    objs_per_ts: dict[int, set[int]] = {}
    for event in events:
        hands_per_ts.setdefault(event.timestamp_ns, set()).add(event.hand)
        objs_per_ts.setdefault(event.timestamp_ns, set()).add(event.object_uid)

    summary = {
        "sequence_dir": str(seq_dir),
        "smplx_npz": str(args.npz.resolve()),
        "stride": int(args.stride),
        "num_motion_frames": int(n),
        "contact_threshold_m": float(args.threshold),
        "bbox_shrink_m": float(args.bbox_shrink),
        "num_contact_events": len(events),
        "num_contact_intervals": len(intervals),
        "num_unique_contact_timestamps": int(timestamps.size),
        "num_human_contact_episodes": counts["num_human_contact_episodes"],
        "num_object_touch_episodes_total": counts["num_object_touch_episodes_total"],
        "num_unique_objects_contacted": counts["num_unique_objects_contacted"],
        "objects_tested": len(objects),
        "use_hand_vertices": not args.no_hand_vertices,
        "outputs": {
            "contacts_csv": str(contacts_csv),
            "intervals_csv": str(intervals_csv),
            "contact_timestamps_ns": str(timestamps_npy),
            "contact_episode_counts_json": str(counts_json),
            "contact_episode_counts_by_object_csv": str(counts_csv),
        },
        "sample_timestamps": [
            {
                "timestamp_ns": int(ts),
                "hands": sorted(hands_per_ts.get(ts, set())),
                "object_uids": sorted(objs_per_ts.get(ts, set())),
            }
            for ts in timestamps[:20]
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print(
        f"[contact] {len(events):,} events | {len(intervals):,} hand-object intervals | "
        f"{timestamps.size:,} unique timestamps"
    )
    print(
        f"[contact] {counts['num_human_contact_episodes']:,} human contact episodes | "
        f"{counts['num_object_touch_episodes_total']:,} object touch episodes | "
        f"{counts['num_unique_objects_contacted']} objects touched"
    )
    print(f"[contact] wrote {contacts_csv}")
    print(f"[contact] wrote {intervals_csv}")
    print(f"[contact] wrote {timestamps_npy}")
    print(f"[contact] wrote {counts_json}")
    print(f"[contact] wrote {counts_csv}")
    print(f"[contact] wrote {summary_json}")


if __name__ == "__main__":
    main()
