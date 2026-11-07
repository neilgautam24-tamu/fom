"""Training objectives: weighted masked reconstruction (Eq. 16) and head-camera consistency (Eq. 19)."""

from __future__ import annotations

import smplx
import torch

from fom.body.representation import GLOBAL_ORIENT_SLICE, GLOBAL_TRANS_SLICE, decode_sequence
from fom.body.smplx_model import JOINT_LEFT_EYE


def weighted_motion_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_frames: torch.Tensor,
    global_orient_weight: float,
    global_trans_weight: float,
) -> torch.Tensor:
    """Masked MSE with up-weighted global root channels, normalized per sequence then averaged."""
    feat_weights = torch.ones(target.shape[-1], device=target.device, dtype=target.dtype)
    feat_weights[GLOBAL_ORIENT_SLICE] = global_orient_weight
    feat_weights[GLOBAL_TRANS_SLICE] = global_trans_weight

    mask = valid_frames.view(*valid_frames.shape, 1).expand_as(target)
    weights = mask * feat_weights.view(1, 1, -1)
    num = (((target - pred) ** 2) * weights).flatten(1).sum(1)
    den = weights.flatten(1).sum(1)
    return (num / (den + 1e-8)).mean()


def headcam_loss(
    pred_motion: torch.Tensor,
    traj: torch.Tensor,
    valid_frames: torch.Tensor,
    traj_mask: torch.Tensor | None,
    body_model: smplx.SMPLXLayer,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared distance between the predicted SMPL-X left eye and the conditioning camera.

    Both ``pred_motion`` and ``traj`` must already be *denormalized*. Frames whose trajectory
    was replaced by the mask token (``traj_mask == 1``) are excluded.

    Returns:
        ``(loss, mean distance in metres)``.
    """
    dists, weights = [], []
    for b in range(pred_motion.shape[0]):
        num_valid = int((valid_frames[b] > 0.5).sum().item())
        if num_valid < 2:
            continue
        weight = torch.ones(num_valid, device=pred_motion.device, dtype=pred_motion.dtype)
        if traj_mask is not None:
            weight = weight * (1.0 - traj_mask[b, :num_valid].to(weight.dtype))

        camera_T, params, _ = decode_sequence(pred_motion[b, :num_valid], traj[b, :num_valid], body_model)
        left_eye = torch.cat(
            [
                body_model(**{k: v[i : i + chunk_size] for k, v in params.items()}).joints[:, JOINT_LEFT_EYE]
                for i in range(0, num_valid, chunk_size)
            ],
            dim=0,
        )
        dists.append(torch.linalg.norm(left_eye - camera_T[:, :3, 3], dim=-1))
        weights.append(weight)

    if not dists:
        zero = pred_motion.sum() * 0.0
        return zero, zero

    dist, weight = torch.cat(dists), torch.cat(weights)
    denom = weight.sum().clamp_min(1e-8)
    return ((dist**2) * weight).sum() / denom, (dist * weight).sum() / denom
