"""Prediction, ground-truth export and metric computation on the packed validation sets.

Prediction / ground-truth files map ``sequence_name -> {"smpl_params": {...}, "camera_T": (T, 4, 4)}``
in world (canonicalized-clip) coordinates; see :mod:`fom.evaluation.io` for reading / writing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from fom.body.smplx_model import evaluate_smplx
from fom.data.dataset import MotionDataset
from fom.evaluation.metrics import compute_metrics
from fom.training.module import FoMModule
from fom.utils.tensors import collate, to_device

# Metrics reported in the paper, in table order.
PAPER_METRICS = [
    ("mpjpe", "MPJPE"),
    ("mpjpe_pa", "PA-MPJPE"),
    ("mpjpe_body", "Body"),
    ("mpjpe_hand", "Hand"),
    ("head_rot_error", "Head Rot."),
    ("head_trans_error", "Head Trans."),
    ("sliding_pred", "Foot Slide"),
    ("contact_validity_pred", "Foot Contact"),
]


@dataclass(frozen=True)
class Ablation:
    """Modalities removed at inference time (same weights, Table 2)."""

    drop_rgb: bool = False
    drop_depth: bool = False
    drop_goal: bool = False
    drop_traj: bool = False

    def apply(self, y: dict) -> dict:
        if self.drop_traj and "traj" in y:
            batch, frames = y["traj"].shape[:2]
            y["traj_mask"] = torch.ones(batch, frames, device=y["traj"].device, dtype=torch.long)
        if self.drop_rgb:
            y.pop("rgb", None)
            y.pop("rgb_valid", None)
        if self.drop_depth:
            y.pop("depth", None)
            y.pop("depth_valid", None)
        if self.drop_goal:
            y.pop("goal", None)
        return y

    def __str__(self) -> str:
        dropped = [name for name in ("rgb", "depth", "goal", "traj") if getattr(self, f"drop_{name}")]
        return "none" if not dropped else ",".join(dropped)


def _export(decoded: dict) -> dict:
    return {"smpl_params": to_device(decoded["smpl_params"], "cpu"), "camera_T": decoded["camera_T"].cpu()}


@torch.inference_mode()
def predict(
    module: FoMModule,
    dataset: MotionDataset,
    batch_size: int,
    ablation: Ablation = Ablation(),
    full_length: bool = False,
    device: torch.device | str = "cuda",
) -> dict[str, dict]:
    """Closed-loop predictions for every sequence of ``dataset``.

    Args:
        full_length: roll out each whole sequence instead of the first ``WINDOW`` frames
            (processed one sequence at a time).
    """
    module = module.to(device).eval()
    preds: dict[str, dict] = {}
    step = 1 if full_length else batch_size
    names = dataset.seq_names
    for start in tqdm(range(0, len(names), step), desc="predict"):
        batch = collate([dataset.get_sequence(n, full_length=full_length) for n in names[start : start + step]])
        y = ablation.apply(to_device(batch["y"], device))
        motion = module.sample(y, len(batch["misc"]["seq_name"])).cpu()
        for name, decoded in zip(batch["misc"]["seq_name"], dataset.to_body(batch, motion)):
            preds[name] = _export(decoded)
    return preds


def ground_truth(dataset: MotionDataset) -> dict[str, dict]:
    """Decoded ground truth matching :func:`predict` (first ``WINDOW`` frames)."""
    out = {}
    for name in tqdm(dataset.seq_names, desc="ground truth"):
        batch = collate([dataset.get_sequence(name)])
        out[name] = _export(dataset.to_body(batch)[0])
    return out




def _decode(entry: dict, body_model, device) -> dict:
    params = to_device(entry["smpl_params"], device)
    joints, vertices, full_pose = evaluate_smplx(body_model, params)
    return to_device(
        {"camera_T": entry["camera_T"], "joints": joints, "vertices": vertices, "full_pose": full_pose}, "cpu"
    )


def evaluate(
    gt: dict[str, dict], preds: dict[str, dict], body_model, device: torch.device | str = "cuda"
) -> dict[str, np.ndarray]:
    """Per-sequence metrics over all ground-truth sequences; returns ``{metric: (N,) array}``."""
    body_model = body_model.to(device)
    parents = list(body_model.parents)
    missing = sorted(set(gt) - set(preds))
    if missing:
        raise KeyError(f"{len(missing)} ground-truth sequences have no prediction, e.g. {missing[:3]}")

    results: dict[str, list] = defaultdict(list)
    for name in tqdm(list(gt), desc="metrics"):
        gt_entry, pred_entry = gt[name], preds[name]
        num_frames = len(gt_entry["smpl_params"]["global_orient"])
        pred_entry = {
            "smpl_params": {k: v[:num_frames] for k, v in pred_entry["smpl_params"].items()},
            "camera_T": (pred_entry["camera_T"] if pred_entry["camera_T"] is not None else gt_entry["camera_T"])[
                :num_frames
            ],
        }
        metrics = compute_metrics(
            _decode(gt_entry, body_model, device), _decode(pred_entry, body_model, device), parents
        )
        for key, value in metrics.items():
            results[key].append(value)
    return {k: np.asarray(v) for k, v in results.items()}


def summarize(metrics: dict[str, np.ndarray]) -> str:
    lines = []
    for key, label in PAPER_METRICS:
        if key in metrics:
            values = np.asarray(metrics[key], dtype=np.float64)
            lines.append(f"{label:14s} {np.nanmean(values):8.4f}   (n={values.size})")
    return "\n".join(lines)
