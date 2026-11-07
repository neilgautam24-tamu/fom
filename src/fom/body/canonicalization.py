"""Canonicalize a stored motion segment so its first camera frame sits at the XY origin facing +X."""

from __future__ import annotations

import copy

import smplx
import torch

from fom.utils import rotation_conversions as rc
from fom.utils.kinematics import rot_trans_to_matrix, yaw_alignment
from fom.utils.pca import pca_to_matrix


def stored_to_full_sequence(
    camera_traj: torch.Tensor, smpl_params: dict[str, torch.Tensor], body_model: smplx.SMPLXLayer
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Expand the compact on-disk format into rotation matrices.

    On disk, rotations are 6D, the camera trajectory is ``(T, 9)`` (6D rot + translation),
    betas are stored once and hand poses are PCA coefficients.
    If the input is already expanded (``(T, 4, 4)`` camera poses) it is returned as a copy.
    """
    num_frames = camera_traj.shape[0]
    if camera_traj.ndim == 3:
        assert camera_traj.shape == (num_frames, 4, 4)
        return camera_traj.clone(), copy.deepcopy(smpl_params)

    camera_T = torch.eye(4)[None].repeat(num_frames, 1, 1).to(camera_traj.device)
    camera_T[:, :3, :3] = rc.rotation_6d_to_matrix(camera_traj[:, :6])
    camera_T[:, :3, 3] = camera_traj[:, 6:9]

    params = copy.deepcopy(smpl_params)
    params["global_orient"] = rc.rotation_6d_to_matrix(smpl_params["global_orient"])
    params["body_pose"] = rc.rotation_6d_to_matrix(smpl_params["body_pose"])
    params["betas"] = smpl_params["betas"].repeat(num_frames, 1)
    if "left_hand_pose" in smpl_params:
        params["left_hand_pose"] = pca_to_matrix(smpl_params["left_hand_pose"], body_model.left_hand_components)
        params["right_hand_pose"] = pca_to_matrix(smpl_params["right_hand_pose"], body_model.right_hand_components)
    return camera_T, params


def canonicalize_camera(camera_T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Yaw/XY-canonicalize camera poses only; returns ``(canonical poses, (4, 4) transform)``."""
    rot = yaw_alignment(camera_T[:1, :3, :3])[0]
    trans = camera_T[0, :3, 3] * torch.tensor([-1.0, -1, 0], dtype=camera_T.dtype)
    trans = (rot @ trans[:, None])[:, 0]
    transform = rot_trans_to_matrix(rot, trans)
    return transform[None] @ camera_T, transform


def canonicalize_trajectory(
    camera_T: torch.Tensor, smpl_params: dict[str, torch.Tensor], joints: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Yaw-rotate and XY-translate the sequence so camera frame 0 is at the origin facing +X.

    The SMPL-X global orientation is applied about the pelvis (``joints[:, 0]``), not the
    origin, so the body transform is handled through the pelvis position.

    Returns:
        canonical camera poses ``(T, 4, 4)``, canonical SMPL-X params, canonical joints and the
        ``(4, 4)`` canonicalizing transform.
    """
    can_camera_T, transform = canonicalize_camera(camera_T)
    rot, trans = transform[:3, :3], transform[:3, 3]

    pelvis = joints[:, 0]
    body_T = rot_trans_to_matrix(smpl_params["global_orient"], pelvis)
    can_body_T = transform[None] @ body_T

    can_params = copy.deepcopy(smpl_params)
    can_params["global_orient"] = can_body_T[:, :3, :3]
    can_params["transl"] = can_body_T[:, :3, 3] - (pelvis - smpl_params["transl"])

    can_joints = (rot[None, None] @ joints[..., None])[..., 0] + trans[None, None]
    return can_camera_T, can_params, can_joints, transform


def canonicalized_segment(
    smpl_params: dict[str, torch.Tensor],
    camera_traj: torch.Tensor,
    joints: torch.Tensor,
    body_model: smplx.SMPLXLayer,
    start_idx: int,
    end_idx: int,
    motion_fps: int,
) -> dict:
    """Cut ``[start_idx, end_idx]`` (inclusive, motion-fps indices) and canonicalize it.

    ``start_idx`` / ``end_idx`` in the returned dict are converted to the 30 fps index space
    that the visual features are keyed by.
    """
    seg_params = {k: v[start_idx : end_idx + 1].clone() for k, v in smpl_params.items()}
    seg_params["betas"] = smpl_params["betas"][:1].clone()  # betas are stored once per sequence
    seg_camera, seg_params = stored_to_full_sequence(camera_traj[start_idx : end_idx + 1].clone(), seg_params, body_model)
    seg_joints = joints[start_idx : end_idx + 1].clone()

    can_camera_T, can_params, can_joints, transform = canonicalize_trajectory(seg_camera, seg_params, seg_joints)

    fps_ratio = 30 // motion_fps
    return {
        "start_idx": start_idx * fps_ratio,
        "end_idx": end_idx * fps_ratio,
        "camera_T": can_camera_T,
        "smpl_params": can_params,
        "joints": can_joints[:, :76],  # body + hands
        "body_root_offset": can_joints[0, 0] - can_params["transl"][0],
        "transform": transform,
    }
