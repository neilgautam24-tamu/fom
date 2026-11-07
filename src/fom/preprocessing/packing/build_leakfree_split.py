#!/usr/bin/env python3
"""Assemble the leak-free (subject- and location-disjoint) Nymeria train/val split used in the paper.

Train = every previously packed sequence (random-split train + val), recentered
        (see :mod:`fom.preprocessing.packing.recenter`).
Val   = newly processed held-out episodes under ``<work-dir>/processed_val`` listed in
        ``<work-dir>/val_episode_selection.json``.

Packed outputs (``ee_*.pt``, merged feature files, statistics) are written to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from fom.preprocessing.packing.build_split import (
    convert_episode,
)
from fom.preprocessing.packing.stats import compute_statistics

TRAIN_FEATURE_PAIRS = [
    ("egoview_dinov2_train.pt", "egoview_dinov2_val.pt", "egoview_dinov2_train.pt"),
    ("goal_cond_perframe_train.pt", "goal_cond_perframe_val.pt", "goal_cond_perframe_train.pt"),
    ("depth_train.pt", "depth_val.pt", "depth_train.pt"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_maps(a: dict, b: dict, *, label: str) -> tuple[dict, list[str]]:
    overlap = sorted(set(a) & set(b))
    if overlap:
        print(f"[merge] {label}: dropping {len(overlap)} overlapping keys already in train")
        for k in overlap[:8]:
            print(f"         {k}")
    merged = dict(a)
    for k, v in b.items():
        if k not in merged:
            merged[k] = v
    return merged, overlap


def merge_pt_dicts(src_a: Path, src_b: Path, dst: Path) -> dict:
    print(f"[merge] {src_a.name} + {src_b.name} -> {dst}")
    a = torch.load(src_a, weights_only=False)
    b = torch.load(src_b, weights_only=False)
    wrapped = isinstance(a, dict) and isinstance(b, dict) and "feats" in a and "feats" in b
    if wrapped:
        feats, overlap = _merge_maps(a["feats"], b["feats"], label=src_a.name)
        out = {"feats": feats}
        n = len(feats)
    else:
        out, overlap = _merge_maps(a, b, label=src_a.name)
        n = len(out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    del a, b, out
    return {
        "sequences": n,
        "overlap_dropped": len(overlap),
        "src_a": str(src_a),
        "src_b": str(src_b),
        "dst": str(dst),
        "wrapped_feats": wrapped,
    }


def successful_val_episodes(processed_val: Path, names: list[str]) -> list[dict]:
    episodes = []
    for name in names:
        out_dir = processed_val / name
        status_path = out_dir / "episode_status.json"
        bundles = out_dir / "visible_goal_sequence_bundles"
        if not status_path.is_file() or not bundles.is_dir():
            continue
        status = json.loads(status_path.read_text())
        if status.get("status") != "success":
            continue
        episodes.append(
            {
                "episode_name": name,
                "source": "single_object",
                "bundle_root": str(bundles),
                "prefixes": ("vgoal_seq_",),
            }
        )
    return episodes


def compute_stats(ee_train_path: Path, stats_path: Path) -> None:
    compute_statistics(ee_train_path, stats_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="Holds val_episode_selection.json and processed_val/")
    parser.add_argument("--out-dir", type=Path, required=True, help="Packed output directory")
    parser.add_argument("--recentered-dir", type=Path, required=True,
                        help="Recentered packed random split (becomes the new train split)")
    parser.add_argument("--original-dir", type=Path, required=True,
                        help="Un-recentered packed split; fallback source for feature files")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--min-duration-s", type=float, default=2.0)
    parser.add_argument("--skip-train-merge", action="store_true")
    parser.add_argument("--skip-val-convert", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--skip-feature-merge", action="store_true")
    parser.add_argument("--force-val", action="store_true")
    args = parser.parse_args()

    motion_dir = args.work_dir.resolve()
    v2_dir = args.recentered_dir.resolve()
    v1_dir = args.original_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads((motion_dir / "val_episode_selection.json").read_text())
    val_names = [ep["name"] for ep in selection["episodes"]]

    summary: dict = {
        "generated_at": utc_now(),
        "script": "fom.preprocessing.packing.build_leakfree_split",
        "split_strategy": "all_existing_as_train__new_episodes_val_subject_and_location_disjoint",
        "train_subjects_blocked": selection["train_subjects"],
        "train_locations_blocked": selection["train_locations"],
        "selected_val_episodes": val_names,
    }

    ee_dst = out_dir / "ee_train.pt"
    if not args.skip_train_merge:
        if ee_dst.is_file() and ee_dst.stat().st_size > 1e9:
            print(f"[merge] reuse existing {ee_dst}")
            summary["train_ee"] = {"dst": str(ee_dst), "reused": True}
        else:
            summary["train_ee"] = merge_pt_dicts(
                v2_dir / "ee_train.pt",
                v2_dir / "ee_val.pt",
                ee_dst,
            )

    if not args.skip_feature_merge:
        feat_summary = {}
        for train_name, val_name, out_name in TRAIN_FEATURE_PAIRS:
            dst = out_dir / out_name
            if dst.is_file() and dst.stat().st_size > 1e8:
                print(f"[merge] reuse existing {dst.name}")
                feat_summary[out_name] = {"dst": str(dst), "reused": True}
                continue
            train_src = v2_dir / train_name
            if not train_src.is_file():
                train_src = v1_dir / train_name
            val_src = v2_dir / val_name
            if not val_src.is_file():
                val_src = v1_dir / val_name
            if not train_src.is_file() or not val_src.is_file():
                print(f"[merge] skip missing feature pair {train_name}/{val_name}")
                continue
            feat_summary[out_name] = merge_pt_dicts(train_src, val_src, dst)
        summary["train_features"] = feat_summary

    processed_val = motion_dir / "processed_val"
    val_episodes = successful_val_episodes(processed_val, val_names)
    failed_val = [n for n in val_names if n not in {e["episode_name"] for e in val_episodes}]
    summary["val_episodes_success"] = [e["episode_name"] for e in val_episodes]
    summary["val_episodes_failed"] = failed_val
    print(f"[val] successful processed episodes: {len(val_episodes)} / {len(val_names)}")
    if failed_val:
        print("[val] missing/failed:", failed_val)

    if not args.skip_val_convert:
        if not val_episodes:
            raise SystemExit("No successful val episodes to convert.")
        episode_stats = []
        val_data: dict = {}
        for episode in tqdm(val_episodes, desc="convert val"):
            meta = convert_episode(
                episode,
                output_dir=motion_dir,
                device=args.device,
                min_frames=int(args.min_frames),
                min_duration_s=float(args.min_duration_s),
                force=bool(args.force_val),
                recenter=True,
            )
            episode_stats.append(meta)
            shard = Path(meta["shard_path"])
            if shard.is_file():
                val_data.update(torch.load(shard, weights_only=False))
        torch.save(val_data, out_dir / "ee_val.pt")
        summary["val_ee"] = {
            "sequences": len(val_data),
            "episodes": len(val_episodes),
            "episode_stats": episode_stats,
            "dst": str(out_dir / "ee_val.pt"),
        }
        tmp = motion_dir / "_tmp"
        if tmp.is_dir():
            shutil.rmtree(tmp, ignore_errors=True)

    if not args.skip_stats:
        compute_stats(out_dir / "ee_train.pt", out_dir / "v4_beta_ee_train_stats.pt")
        summary["stats"] = str(out_dir / "v4_beta_ee_train_stats.pt")

    train_n = 0
    val_n = 0
    if (out_dir / "ee_train.pt").is_file():
        train_n = len(torch.load(out_dir / "ee_train.pt", weights_only=False))
    if (out_dir / "ee_val.pt").is_file():
        val_n = len(torch.load(out_dir / "ee_val.pt", weights_only=False))

    split_log = {
        "generated_at": utc_now(),
        "script": "fom.preprocessing.packing.build_leakfree_split",
        "split_strategy": summary["split_strategy"],
        "train_is_all_previous_data": True,
        "val_is_new_download": True,
        "subject_disjoint": True,
        "location_disjoint": True,
        "train_subjects": selection["train_subjects"],
        "train_locations": selection["train_locations"],
        "val_subjects": [ep["subject"] for ep in selection["episodes"] if ep["name"] not in failed_val],
        "val_locations": [ep["location"] for ep in selection["episodes"] if ep["name"] not in failed_val],
        "val_episodes": [e["episode_name"] for e in val_episodes],
        "val_episodes_failed": failed_val,
        "train_sequences": train_n,
        "val_sequences": val_n,
        "assemble_summary": summary,
    }
    (motion_dir / "split_log.json").write_text(json.dumps(split_log, indent=2) + "\n")

    manifest = {
        "generated_at": utc_now(),
        "split_strategy": summary["split_strategy"],
        "train_sequences": train_n,
        "val_sequences": val_n,
        "val_episodes": len(val_episodes),
        "val_episodes_failed": failed_val,
        "subject_disjoint": True,
        "location_disjoint": True,
        "ee_train": str(out_dir / "ee_train.pt"),
        "ee_val": str(out_dir / "ee_val.pt"),
        "processed_val": str(processed_val),
    }
    (motion_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
