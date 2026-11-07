"""SMPL-X body model loading and batched evaluation."""

from __future__ import annotations

import smplx
import torch
from loguru import logger

from fom.paths import body_models_dir

# SMPL-X kinematic-tree joint indices used throughout the project.
JOINT_HEAD = 15
JOINT_LEFT_EYE = 23
NUM_BODY_JOINTS = 22  # pelvis + 21 body joints
HAND_JOINTS = slice(25, 55)  # 15 left + 15 right hand joints


def load_smplx(num_pca_comps: int = 12) -> smplx.SMPLXLayer:
    """Neutral SMPL-X layer with PCA hands.

    The layer exposes all 145 joints; the first 55 follow the SMPL-X kinematic tree
    (22 body + 3 face + 15 left hand + 15 right hand).
    """
    path = body_models_dir() / "smplx"
    logger.info(f"Loading SMPL-X from {path}")
    return smplx.SMPLXLayer(str(path), gender="NEUTRAL", use_pca=True, num_pca_comps=num_pca_comps)


@torch.no_grad()
def evaluate_smplx(
    model: smplx.SMPLXLayer, params: dict[str, torch.Tensor], chunk_size: int = 1000
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the body model over a sequence in chunks.

    Returns:
        ``(joints, vertices, full_pose)`` concatenated over time.
    """
    keys = ["body_pose", "global_orient", "betas", "transl"]
    if "left_hand_pose" in params:
        keys += ["left_hand_pose", "right_hand_pose"]

    joints, vertices, full_pose = [], [], []
    num_frames = len(params["body_pose"])
    for start in range(0, num_frames, chunk_size):
        out = model(**{k: params[k][start : start + chunk_size] for k in keys}, return_full_pose=True)
        joints.append(out.joints)
        vertices.append(out.vertices)
        full_pose.append(out.full_pose)
    return torch.cat(joints), torch.cat(vertices), torch.cat(full_pose)
