#!/usr/bin/env python3
"""Recenter an existing packed dataset (frame-0 pelvis -> world XY origin).

Recentering is a rigid per-sequence XY translation (frame-0 pelvis -> world
origin), so it is a pure function of the already-converted ``ee_*.pt`` tensors.
We therefore avoid re-running the full (GPU) conversion and instead transform
the existing ``ee_train.pt`` / ``ee_val.pt`` in place, reusing the exact same
``recenter_sequence_inplace`` helper that :mod:`fom.preprocessing.packing.pack`
applies during packing.

DINOv2 features and goal heatmaps live in image space (not the world frame), so
they are unaffected and are symlinked (default) rather than duplicated.

Usage:
  python -m fom.preprocessing.packing.recenter --input-dir <packed> --output-dir <packed_recentered>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from fom.preprocessing.packing.pack import (
    recenter_sequence_inplace,
)
from fom.preprocessing.packing.stats import compute_statistics

AUX_FILES = [
    "egoview_dinov2_train.pt",
    "egoview_dinov2_val.pt",
    "goal_cond_perframe_train.pt",
    "goal_cond_perframe_val.pt",
]


def far_count(data: dict, thresh: float = 20.0) -> int:
    n = 0
    for s in data.values():
        if float(np.linalg.norm(s["kp3d"][0, 0].numpy()[:2])) > thresh:
            n += 1
    return n


def recenter_split(src: Path, dst: Path) -> dict:
    data = torch.load(src, weights_only=False)
    before = far_count(data)
    for s in data.values():
        recenter_sequence_inplace(s)
    after = far_count(data)
    torch.save(data, dst)
    return {"sequences": len(data), "far_before_20m": before, "far_after_20m": after}


def link_or_copy_aux(v1_dir: Path, v2_dir: Path, *, copy: bool) -> list[str]:
    handled = []
    for name in AUX_FILES:
        src = v1_dir / name
        if not src.is_file():
            continue
        dst = v2_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy:
            import shutil

            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())
        handled.append(name)
    return handled


def compute_stats(ee_train_path: Path, stats_path: Path) -> None:
    compute_statistics(ee_train_path, stats_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Packed dataset to recenter")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--copy-aux",
        action="store_true",
        help="Copy DINOv2/goal feature files instead of symlinking (default: symlink)",
    )
    args = parser.parse_args()

    v1_dir = args.input_dir.resolve()
    v2_dir = args.output_dir.resolve()
    v2_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ("train", "val"):
        src = v1_dir / f"ee_{split}.pt"
        if not src.is_file():
            raise FileNotFoundError(src)
        print(f"[recenter] {split}: {src}")
        summary[split] = recenter_split(src, v2_dir / f"ee_{split}.pt")
        print(f"  {summary[split]}")

    aux = link_or_copy_aux(v1_dir, v2_dir, copy=args.copy_aux)
    print(f"[aux] {'copied' if args.copy_aux else 'symlinked'}: {aux}")

    stats_path = v2_dir / "v4_beta_ee_train_stats.pt"
    print(f"[stats] recomputing -> {stats_path}")
    compute_stats(v2_dir / "ee_train.pt", stats_path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "fom.preprocessing.packing.recenter",
        "input_dir": str(v1_dir),
        "output_dir": str(v2_dir),
        "recenter": "frame-0 pelvis -> world XY origin (rigid XY shift)",
        "splits": summary,
        "aux_files": aux,
        "aux_mode": "copy" if args.copy_aux else "symlink",
        "stats_recomputed": stats_path.is_file(),
    }
    (v2_dir / "recenter_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] recentered dataset -> {v2_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
