#!/usr/bin/env python3
"""Pack all processed Nymeria episodes into one train/val split (random by episode).

Combines single-object (``processed_data``) and multi-object
(``processed_data_multiobject``) episodes into one ``ee_train.pt`` / ``ee_val.pt``
split by episode (default 15% validation episodes).
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from fom.preprocessing.packing.pack import (
    NymeriaSequencePacker,
    find_sequences,
)


def discover_episodes(
    single_root: Path,
    multi_root: Path,
    *,
    require_success: bool = True,
) -> list[dict]:
    episodes: list[dict] = []

    for ep_dir in sorted(single_root.iterdir()):
        if not ep_dir.is_dir() or ep_dir.name == "logs":
            continue
        bundle_root = ep_dir / "visible_goal_sequence_bundles"
        if not bundle_root.is_dir():
            continue
        if require_success:
            status_path = ep_dir / "episode_status.json"
            if status_path.is_file():
                if json.loads(status_path.read_text()).get("status") != "success":
                    continue
        episodes.append(
            {
                "episode_name": ep_dir.name,
                "source": "single_object",
                "bundle_root": str(bundle_root),
                "prefixes": ("vgoal_seq_",),
            }
        )

    for ep_dir in sorted(multi_root.iterdir()):
        if not ep_dir.is_dir() or ep_dir.name == "logs":
            continue
        bundle_root = ep_dir / "multi_object_visible_goal_bundles"
        if not bundle_root.is_dir():
            continue
        if require_success:
            status_path = ep_dir / "episode_status.json"
            if not status_path.is_file():
                continue
            if json.loads(status_path.read_text()).get("status") != "success":
                continue
        episodes.append(
            {
                "episode_name": ep_dir.name,
                "source": "multi_object",
                "bundle_root": str(bundle_root),
                "prefixes": ("mvgoal_seq_",),
            }
        )

    return episodes


def build_episode_split(
    episodes: list[dict],
    *,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign episodes to train/val with stratification by source (single vs multi)."""
    from collections import defaultdict

    by_source: dict[str, list[str]] = defaultdict(list)
    for ep in episodes:
        by_source[ep["source"]].append(ep["episode_name"])

    rng = random.Random(seed)
    split: dict[str, str] = {}
    for source in sorted(by_source):
        names = by_source[source]
        shuffled = names[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n == 0:
            continue
        if n == 1:
            split[shuffled[0]] = "train"
            continue
        n_val = max(1, round(n * val_ratio))
        n_val = min(n_val, n - 1)
        val_set = set(shuffled[:n_val])
        for name in names:
            split[name] = "val" if name in val_set else "train"
    return split


def summarize_split(
    episodes: list[dict],
    episode_split: dict[str, str],
) -> dict:
    from collections import Counter

    episode_counts: Counter[tuple[str, str]] = Counter()
    for ep in episodes:
        source = ep["source"]
        split = episode_split[ep["episode_name"]]
        episode_counts[(source, split)] += 1

    summary: dict[str, dict] = {}
    for source in sorted({ep["source"] for ep in episodes}):
        train_n = episode_counts[(source, "train")]
        val_n = episode_counts[(source, "val")]
        total = train_n + val_n
        summary[source] = {
            "train_episodes": train_n,
            "val_episodes": val_n,
            "total_episodes": total,
            "val_fraction": round(val_n / total, 4) if total else 0.0,
        }
    return summary


def shard_path(shards_dir: Path, source: str, episode_name: str) -> Path:
    return shards_dir / source / f"{episode_name}.pt"


def convert_episode(
    episode: dict,
    *,
    output_dir: Path,
    device: str,
    min_frames: int,
    min_duration_s: float,
    force: bool,
    recenter: bool = True,
    target_fps: int = 15,
) -> dict:
    source = episode["source"]
    episode_name = episode["episode_name"]
    shards_dir = output_dir / "shards"
    out_shard = shard_path(shards_dir, source, episode_name)
    meta_path = out_shard.with_suffix(".json")

    if not force and out_shard.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        return meta

    converter = NymeriaSequencePacker(
        bundle_root=Path(episode["bundle_root"]),
        episode_name=episode_name,
        output_dir=output_dir / "_tmp" / source / episode_name,
        device=device,
        min_frames=min_frames,
        train_ratio=1.0,
        recenter=recenter,
        target_fps=target_fps,
    )

    seq_infos = find_sequences(
        Path(episode["bundle_root"]),
        episode_name,
        prefixes=episode["prefixes"],
        min_duration_s=min_duration_s,
    )

    converted: dict[str, dict] = {}
    failed: list[str] = []
    for seq_info in tqdm(
        seq_infos,
        desc=f"{source[:6]}:{episode_name[:28]}",
        leave=False,
    ):
        seq_info = dict(seq_info)
        seq_info["source"] = source
        result = converter.process_sequence(seq_info)
        if result is None:
            continue
        seq_name, seq_data = result
        seq_data["episode_name"] = episode_name
        seq_data["source"] = source
        seq_data["bundle_seq_name"] = seq_info["seq_name"]
        converted[seq_name] = seq_data

    out_shard.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, out_shard)
    meta = {
        "episode_name": episode_name,
        "source": source,
        "num_sequences": len(converted),
        "failed_sequences": converter.failed_sequences,
        "frame_mismatch_count": converter.frame_mismatch_count,
        "interpolated_count": converter.interpolated_count[0],
        "stored_fps": int(target_fps),
        "shard_path": str(out_shard),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def merge_shards(
    episodes: list[dict],
    episode_split: dict[str, str],
    *,
    output_dir: Path,
) -> tuple[dict, dict]:
    train: dict = {}
    val: dict = {}
    shards_dir = output_dir / "shards"

    for episode in episodes:
        episode_name = episode["episode_name"]
        shard = shard_path(shards_dir, episode["source"], episode_name)
        if not shard.is_file():
            continue
        data = torch.load(shard, weights_only=False)
        if episode_split[episode_name] == "val":
            val.update(data)
        else:
            train.update(data)

    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-root", type=Path, required=True, help="Single-object processed episodes")
    parser.add_argument("--multi-root", type=Path, required=True, help="Multi-object processed episodes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=2.0,
        help="Skip bundles shorter than this (from metadata total_duration_s)",
    )
    parser.add_argument("--include-incomplete-episodes", action="store_true")
    parser.add_argument(
        "--recenter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recenter each sequence to the world XY origin (default). "
        "Use --no-recenter to keep world coordinates.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild episode shards")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing shards")
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = discover_episodes(
        args.single_root.resolve(),
        args.multi_root.resolve(),
        require_success=not args.include_incomplete_episodes,
    )
    if args.max_episodes > 0:
        episodes = episodes[: int(args.max_episodes)]

    if not episodes:
        raise SystemExit("No episodes found to convert.")

    episode_split = build_episode_split(
        episodes,
        val_ratio=float(args.val_ratio),
        seed=int(args.split_seed),
    )

    val_episodes = [name for name, split in episode_split.items() if split == "val"]
    train_episodes = [name for name, split in episode_split.items() if split == "train"]
    split_summary = summarize_split(episodes, episode_split)

    split_log = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "fom.preprocessing.packing.build_split",
        "single_root": str(args.single_root.resolve()),
        "multi_root": str(args.multi_root.resolve()),
        "output_dir": str(output_dir),
        "val_ratio": float(args.val_ratio),
        "split_seed": int(args.split_seed),
        "split_strategy": "stratified_by_source",
        "min_frames": int(args.min_frames),
        "target_fps": int(args.target_fps),
        "min_duration_s": float(args.min_duration_s),
        "num_episodes": len(episodes),
        "num_train_episodes": len(train_episodes),
        "num_val_episodes": len(val_episodes),
        "split_summary_by_source": split_summary,
        "episode_split": episode_split,
        "episodes": episodes,
        "val_episodes": sorted(val_episodes),
        "train_episodes": sorted(train_episodes),
    }
    (output_dir / "split_log.json").write_text(json.dumps(split_log, indent=2) + "\n")

    if not args.merge_only:
        episode_stats = []
        for episode in episodes:
            meta = convert_episode(
                episode,
                output_dir=output_dir,
                device=args.device,
                min_frames=int(args.min_frames),
                min_duration_s=float(args.min_duration_s),
                force=bool(args.force),
                recenter=bool(args.recenter),
                target_fps=int(args.target_fps),
            )
            episode_stats.append(meta)
        (output_dir / "episode_conversion_stats.json").write_text(
            json.dumps(episode_stats, indent=2) + "\n"
        )

    train_data, val_data = merge_shards(episodes, episode_split, output_dir=output_dir)
    train_path = output_dir / "ee_train.pt"
    val_path = output_dir / "ee_val.pt"
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)

    by_source = {"single_object": {"train": 0, "val": 0}, "multi_object": {"train": 0, "val": 0}}
    for episode in episodes:
        shard = shard_path(output_dir / "shards", episode["source"], episode["episode_name"])
        if not shard.is_file():
            continue
        n = len(torch.load(shard, weights_only=False))
        split = episode_split[episode["episode_name"]]
        by_source[episode["source"]][split] += n

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_strategy": "stratified_by_source",
        "split_summary_by_source": split_summary,
        "train_sequences": len(train_data),
        "val_sequences": len(val_data),
        "total_sequences": len(train_data) + len(val_data),
        "train_episodes": len(train_episodes),
        "val_episodes": len(val_episodes),
        "by_source": by_source,
        "ee_train": str(train_path),
        "ee_val": str(val_path),
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Episodes: {len(episodes)} ({len(train_episodes)} train / {len(val_episodes)} val)")
    print(f"Sequences: {manifest['total_sequences']} ({len(train_data)} train / {len(val_data)} val)")
    print(f"Saved -> {train_path}")
    print(f"Saved -> {val_path}")
    print(f"Split log -> {output_dir / 'split_log.json'}")


if __name__ == "__main__":
    main()
