"""Compute 3D metrics of saved predictions against ground truth.

    fom-gt   --config <cfg> --out <data_root>/ee_val_gt_for_evaluation.pkl   # once per dataset
    fom-eval --gt <gt.pkl> --preds preds.pkl --out metrics.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fom.body.smplx_model import load_smplx
from fom.config import load_config
from fom.data.dataset import MotionDataset
from fom.evaluation.io import load_sequences, save
from fom.evaluation.runner import evaluate, ground_truth, summarize


def main_eval(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute metrics for a prediction file.")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", required=True, help="Per-sequence metrics .pkl; a .txt summary is written next to it.")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate(load_sequences(args.gt), load_sequences(args.preds), load_smplx(), device)
    save(metrics, args.out)
    text = f"preds: {args.preds}\n\n{summarize(metrics)}\n"
    Path(args.out).with_suffix(".txt").write_text(text)
    print(text, end="")


def main_summarize(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the paper metrics of a metrics .pkl.")
    parser.add_argument("metrics")
    args = parser.parse_args(argv)
    import joblib

    print(summarize(joblib.load(args.metrics)))


def main_ground_truth(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export decoded validation ground truth for fom-eval.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("overrides", nargs="*", help="KEY VALUE config overrides")
    args = parser.parse_intermixed_args(argv)

    cfg = load_config(args.config, args.overrides)
    dataset = MotionDataset(cfg, args.split, load_rgb=False, load_depth=False, load_goal=False)
    save(ground_truth(dataset), args.out)


if __name__ == "__main__":
    main_eval()
