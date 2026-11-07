#!/usr/bin/env python3
"""Select sequences where goal-conditioned predictions beat no-goal ones by a large margin (Fig. S3).

Computes per-sequence root-relative MPJPE (mm) of both prediction files vs ground truth, ranks
by ``delta = mpjpe_nogoal - mpjpe_goal`` (higher = goal helps more), keeps episode diversity and
writes the ranking plus slim prediction / GT files for
:mod:`fom.visualization.viewers.goal_vs_nogoal`.

    fom-predict ... --out preds_goal.pkl
    fom-predict ... --drop-goal --out preds_nogoal.pkl
    python -m fom.visualization.select_goal_examples --pred-goal preds_goal.pkl \
        --pred-nogoal preds_nogoal.pkl --gt gt.pkl --out-dir goal_vs_nogoal/
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import torch
from tqdm.auto import tqdm

from fom.body.smplx_model import evaluate_smplx, load_smplx
from fom.evaluation.io import load_sequences
from fom.visualization.common import parse_sequence_name, to_np


def n_frames(item: dict) -> int:
    go = item["smpl_params"]["global_orient"]
    return int(to_np(go).shape[0])


def parse_episode(pred_key: str) -> str:
    parsed = parse_sequence_name(pred_key)
    return parsed[0] if parsed else pred_key.split("___")[0]


@torch.no_grad()
def mpjpe_mm(smpl, gt_item: dict, pred_item: dict, device: torch.device) -> float:
    def _prep(item: dict) -> dict:
        return {
            k: torch.as_tensor(to_np(v), device=device, dtype=torch.float32)
            for k, v in item["smpl_params"].items()
        }

    gt_smpl = _prep(gt_item)
    pred_smpl = _prep(pred_item)
    T = min(gt_smpl["global_orient"].shape[0], pred_smpl["global_orient"].shape[0])
    gt_smpl = {k: v[:T] for k, v in gt_smpl.items()}
    pred_smpl = {k: v[:T] for k, v in pred_smpl.items()}
    gt_j, _, _ = evaluate_smplx(smpl, gt_smpl)
    pred_j, _, _ = evaluate_smplx(smpl, pred_smpl)
    # Align by pelvis (joint 0) then mean L2 over joints/frames - same spirit as metrics.py
    gt_j = gt_j - gt_j[:, :1]
    pred_j = pred_j - pred_j[:, :1]
    err = torch.linalg.norm(gt_j - pred_j, dim=-1).mean().item()
    return float(err * 1000.0)  # meters -> mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-goal", type=Path, required=True)
    ap.add_argument("--pred-nogoal", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-seqs", type=int, default=30)
    ap.add_argument("--min-frames", type=int, default=80)
    ap.add_argument("--min-delta-mm", type=float, default=20.0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading pickles...", flush=True)
    preds_g = load_sequences(args.pred_goal)
    preds_ng = load_sequences(args.pred_nogoal)
    gt = load_sequences(args.gt)
    keys = sorted(set(preds_g) & set(preds_ng) & set(gt))
    print(f"Shared keys: {len(keys)}", flush=True)

    device = torch.device(args.device)
    smpl = load_smplx().to(device)
    rows = []
    for key in tqdm(keys, desc="mpjpe"):
        nf = min(n_frames(preds_g[key]), n_frames(preds_ng[key]), n_frames(gt[key]))
        if nf < args.min_frames:
            continue
        try:
            m_g = mpjpe_mm(smpl, gt[key], preds_g[key], device)
            m_ng = mpjpe_mm(smpl, gt[key], preds_ng[key], device)
        except Exception as exc:
            print(f"skip {key}: {exc}", flush=True)
            continue
        delta = m_ng - m_g
        rows.append(
            {
                "pred_key": key,
                "episode": parse_episode(key),
                "n_frames": nf,
                "mpjpe_goal_mm": m_g,
                "mpjpe_nogoal_mm": m_ng,
                "delta_mm": delta,
            }
        )

    rows.sort(key=lambda r: (-r["delta_mm"], -r["n_frames"]))
    (args.out_dir / "goal_vs_nogoal_all.json").write_text(json.dumps(rows, indent=2))

    # Prefer large positive deltas; keep episode diversity.
    selected = []
    seen_eps: set[str] = set()
    for r in rows:
        if r["delta_mm"] < args.min_delta_mm:
            continue
        if r["episode"] in seen_eps:
            continue
        selected.append(r)
        seen_eps.add(r["episode"])
        if len(selected) >= args.max_seqs:
            break
    # Fill remaining slots with next-best deltas even if episode repeats.
    if len(selected) < args.max_seqs:
        have = {r["pred_key"] for r in selected}
        for r in rows:
            if r["pred_key"] in have:
                continue
            if r["delta_mm"] < args.min_delta_mm:
                continue
            selected.append(r)
            have.add(r["pred_key"])
            if len(selected) >= args.max_seqs:
                break

    (args.out_dir / "goal_advantage_ranking.json").write_text(json.dumps(selected, indent=2))

    slim_g, slim_ng, slim_gt = {}, {}, {}
    for r in selected:
        k = r["pred_key"]
        slim_g[k] = preds_g[k]
        slim_ng[k] = preds_ng[k]
        slim_gt[k] = gt[k]
    joblib.dump(slim_g, args.out_dir / "preds_goal_viz.pkl")
    joblib.dump(slim_ng, args.out_dir / "preds_nogoal_viz.pkl")
    joblib.dump(slim_gt, args.out_dir / "gt_viz.pkl")

    print(f"Selected {len(selected)} / {len(rows)} scored sequences", flush=True)
    for i, r in enumerate(selected[:15], 1):
        print(
            f"  [{i:02d}] delta={r['delta_mm']:.1f}mm  "
            f"goal={r['mpjpe_goal_mm']:.1f}  nogoal={r['mpjpe_nogoal_mm']:.1f}  "
            f"T={r['n_frames']}  {r['pred_key'][:70]}",
            flush=True,
        )
    if selected:
        print(
            f"delta mm: mean={np.mean([r['delta_mm'] for r in selected]):.1f} "
            f"median={np.median([r['delta_mm'] for r in selected]):.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
