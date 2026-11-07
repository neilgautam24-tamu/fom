"""Forward/inverse kinematics helpers on rotation matrices.

Adapted from EgoEgo (https://github.com/lijiaman/egoego_release).
"""

from __future__ import annotations

from typing import Sequence

import torch

from fom.utils import rotation_conversions as rc


def local_to_global_pose(local_pose: torch.Tensor, parents: Sequence[int]) -> torch.Tensor:
    """Compose parent-relative joint rotations into global rotations.

    Args:
        local_pose: ``(..., J, 3, 3)`` rotations relative to the parent joint.
        parents: kinematic tree, ``parents[j]`` is the parent of joint ``j`` (``-1`` for the root).
    """
    global_pose = local_pose.clone()
    for joint_id, parent_id in enumerate(parents):
        if parent_id >= 0:
            global_pose[..., joint_id, :, :] = torch.matmul(
                global_pose[..., parent_id, :, :], global_pose[..., joint_id, :, :]
            )
    return global_pose


def global_to_local_pose(global_pose: torch.Tensor, parents: Sequence[int]) -> torch.Tensor:
    """Inverse of :func:`local_to_global_pose` for ``(T, J, 3, 3)`` rotations."""
    parents = list(parents)
    return torch.cat(
        [
            global_pose[..., :1, :, :],
            torch.matmul(torch.inverse(global_pose[..., parents[1:], :, :]), global_pose[..., 1:, :, :]),
        ],
        dim=-3,
    )


def rot_trans_to_matrix(rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Stack ``(..., 3, 3)`` rotations and ``(..., 3)`` translations into ``(..., 4, 4)`` transforms."""
    shape = list(rot.shape)[:-2]
    transform = torch.cat([rot, trans[..., :, None]], dim=-1)
    pad = torch.tensor([[0, 0, 0, 1]]).float().to(rot.device).repeat(*shape, 1, 1)
    return torch.cat([transform, pad], dim=-2)


def quaternion_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Unit quaternion(s) rotating ``v1`` onto ``v2`` (both ``(..., 3)``)."""
    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)
    cross = torch.cross(v1, v2, dim=-1)
    dot = (v1 * v2).sum(dim=-1)
    q = torch.cat((1.0 + dot[..., None], cross), dim=-1)
    return q / torch.norm(q, dim=-1, keepdim=True)


def yaw_alignment(rotmat: torch.Tensor) -> torch.Tensor:
    """Rotation about +Z that maps the XY-projected forward axis (+X) of ``rotmat`` back onto +X.

    Args:
        rotmat: ``(B, 3, 3)`` rotations.
    Returns:
        ``(B, 3, 3)`` yaw-only rotations.
    """
    assert rotmat.shape[1:] == (3, 3)
    batch = rotmat.shape[0]
    x_axis = torch.tensor([1.0, 0, 0])[None].repeat(batch, 1)
    forward = rc.quaternion_apply(rc.matrix_to_quaternion(rotmat), x_axis).float()
    forward = torch.tensor([1.0, 1, 0]).float()[None].repeat(batch, 1) * forward
    forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-8)
    return rc.quaternion_to_matrix(rc.quaternion_invert(quaternion_between_vectors(x_axis.float(), forward)))
