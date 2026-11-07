"""Names of the packed dataset files inside ``DATA.ROOT``.

These are written by ``fom.preprocessing`` and read by :mod:`fom.data.dataset`.
Changing them breaks loading of existing packed datasets.
"""

from __future__ import annotations

from pathlib import Path

STATS_FILE = "v4_beta_ee_train_stats.pt"
EVAL_GT_FILE = "ee_val_gt_for_evaluation.pkl"


def motion_file(root: str | Path, split: str) -> Path:
    """``{sequence_name: {smpl_params, aria_traj, kp3d, num_frames, floor_height, ...}}``"""
    return Path(root) / f"ee_{split}.pt"


def rgb_feature_file(root: str | Path, split: str) -> Path:
    """``{"feats": {take_name: (N, 1 + registers, 1024) DINOv2 tokens at VISUAL_FPS}}``"""
    return Path(root) / f"egoview_dinov2_{split}.pt"


def goal_feature_file(root: str | Path, split: str) -> Path:
    """``{base_sequence_name: {"status": str, "heatmap": (N, 256)}}``"""
    return Path(root) / f"goal_cond_perframe_{split}.pt"


def depth_feature_file(root: str | Path, split: str) -> Path:
    """``{base_sequence_name: {"patches": (N, 5, 1024), "valid": (N,)}}``"""
    return Path(root) / f"depth_{split}.pt"


def stats_file(root: str | Path) -> Path:
    return Path(root) / STATS_FILE


def eval_gt_file(root: str | Path) -> Path:
    return Path(root) / EVAL_GT_FILE
