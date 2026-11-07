"""Head(camera)-centric motion representation (243-D per frame) and camera trajectory features (18-D).

Motion vector layout (``MOTION_DIM = 243``)::

    [0:198)    22 body joints x (6D rot + 3D pos), local to the floor-projected camera frame
    [198:204)  6D rotation of the frame-to-frame delta of that canonical frame   (global orient)
    [204:207)  translation of that delta                                          (global trans)
    [207:219)  left-hand PCA (12)
    [219:231)  right-hand PCA (12)
    [231:233)  left / right foot contact
    [233:243)  SMPL-X betas

Trajectory vector layout (``TRAJ_DIM = 18``): camera pose local to its own floor-projected
frame (6D rot + 3D pos), followed by the frame-to-frame delta of that frame (6D rot + 3D pos).
Deltas are integrated when decoding, so the first frame carries the absolute transform.
"""

from __future__ import annotations

import smplx
import torch

from fom.body.canonicalization import stored_to_full_sequence
from fom.utils import rotation_conversions as rc
from fom.utils.kinematics import global_to_local_pose, local_to_global_pose, rot_trans_to_matrix, yaw_alignment
from fom.utils.pca import matrix_to_pca, pca_to_matrix

MOTION_DIM = 243
TRAJ_DIM = 18

GLOBAL_ORIENT_SLICE = slice(198, 204)
GLOBAL_TRANS_SLICE = slice(204, 207)

_LEFT_TOE, _RIGHT_TOE = 10, 11


def detect_foot_contact(joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary toe contact from velocity and height thresholds. ``joints``: ``(T, J, 3)``, z-up."""
    vel_thresh, height_thresh = 0.02, 0.10

    def _contact(joint: int) -> torch.Tensor:
        vel = (joints[1:, joint] - joints[:-1, joint]).square().sum(-1).sqrt()
        vel = (vel < vel_thresh).float()
        vel = torch.cat([vel[:1], vel], dim=0)
        return vel * (joints[:, joint, 2] < height_thresh).float()

    return _contact(_LEFT_TOE), _contact(_RIGHT_TOE)


def _heading_frames(transforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame floor-projected heading transform and its integrated-delta encoding.

    Returns:
        ``to_local`` ``(T, 4, 4)`` mapping world -> heading frame, and ``delta`` ``(T, 4, 4)``
        where ``delta[0]`` is the absolute inverse and ``delta[t]`` the step from ``t-1`` to ``t``.
    """
    rot = yaw_alignment(transforms[:, :3, :3])
    trans = transforms[:, :3, 3] * torch.tensor([-1.0, -1, 0])
    trans = (rot @ trans[..., None])[..., 0]
    to_local = rot_trans_to_matrix(rot, trans)

    inv = to_local.inverse()
    delta = torch.cat([inv[:1], inv[:-1].inverse() @ inv[1:]], dim=0)
    return to_local, delta


def _rottrans(transform: torch.Tensor) -> torch.Tensor:
    return torch.cat([rc.matrix_to_rotation_6d(transform[..., :3, :3]), transform[..., :3, 3]], dim=-1)


def _from_rottrans(x: torch.Tensor) -> torch.Tensor:
    return rot_trans_to_matrix(rc.rotation_6d_to_matrix(x[..., :6]), x[..., 6:9])


def _integrate(delta: torch.Tensor) -> torch.Tensor:
    out = [delta[0]]
    for t in range(1, delta.shape[0]):
        out.append(out[-1] @ delta[t])
    return torch.stack(out, dim=0)


def encode_sequence(
    camera_traj: torch.Tensor,
    smpl_params: dict[str, torch.Tensor],
    joints: torch.Tensor,
    floor_height: float,
    body_model: smplx.SMPLXLayer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a (canonicalized) sequence into ``(motion (T, 243), traj (T, 18))``."""
    camera_T, params = stored_to_full_sequence(camera_traj, smpl_params, body_model)
    left_hand = matrix_to_pca(params["left_hand_pose"], body_model.left_hand_components)
    right_hand = matrix_to_pca(params["right_hand_pose"], body_model.right_hand_components)

    if floor_height == 0:
        floor_height = joints[:, [_LEFT_TOE, _RIGHT_TOE], 2].mean().item() - 0.02
    offset = torch.tensor([0, 0, floor_height]).float().to(camera_T.device)
    camera_T[:, :3, 3] = camera_T[:, :3, 3] - offset
    joints = joints - offset
    params["transl"] = params["transl"] - offset

    feet_l, feet_r = detect_foot_contact(joints)

    num_frames = camera_T.shape[0]
    full_pose = torch.cat([params["global_orient"][:, None], params["body_pose"]], dim=1)  # T x 22 x 3 x 3
    global_rot = local_to_global_pose(full_pose, list(body_model.parents)[:22])
    global_T = rot_trans_to_matrix(global_rot, joints[:, :22])  # T x 22 x 4 x 4

    # The body is expressed relative to the floor-projected *camera* (head) frame. Historically this
    # was requested as "left eye" (joint 23), which is not among the 22 body joints and therefore
    # resolved to the camera pose; we keep that behaviour so stored statistics remain valid.
    to_local, delta = _heading_frames(camera_T)
    local_T = to_local[:, None] @ global_T

    motion = torch.cat(
        [
            _rottrans(local_T[:, :22]).view(num_frames, 198),
            _rottrans(delta),
            left_hand,
            right_hand,
            feet_l.view(num_frames, 1),
            feet_r.view(num_frames, 1),
            params["betas"],
        ],
        dim=1,
    )

    return motion, encode_trajectory(camera_T)


def encode_trajectory(camera_T: torch.Tensor) -> torch.Tensor:
    """``(T, 4, 4)`` floor-relative camera poses -> ``(T, 18)`` trajectory features."""
    cam_to_local, cam_delta = _heading_frames(camera_T)
    return torch.cat([_rottrans(cam_to_local @ camera_T), _rottrans(cam_delta)], dim=1)


def decode_trajectory(traj: torch.Tensor) -> torch.Tensor:
    """``(T, 18)`` trajectory features -> ``(T, 4, 4)`` camera poses."""
    return _integrate(_from_rottrans(traj[:, 9:18])) @ _from_rottrans(traj[:, :9])


def decode_sequence(
    motion: torch.Tensor | None, traj: torch.Tensor | None, body_model: smplx.SMPLXLayer
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None, torch.Tensor | None]:
    """Decode features back to camera poses, SMPL-X params and global body-joint positions.

    Betas are the mean of the per-frame predicted betas.

    Returns:
        ``(camera_T (T, 4, 4) | None, smpl_params | None, body_joints (T, 22, 3) | None)``
    """
    camera_T = decode_trajectory(traj) if traj is not None else None
    if motion is None:
        return camera_T, None, None

    num_frames = motion.shape[0]
    betas = motion[:, -10:].mean(0)[None]
    local_T = _from_rottrans(motion[:, :198].view(num_frames, 22, 9))
    global_T = _integrate(_from_rottrans(motion[:, 198:207]))[:, None] @ local_T
    global_trans = global_T[:, :, :3, 3]
    local_rot = global_to_local_pose(global_T[:, :, :3, :3], list(body_model.parents)[:22])

    params = {
        "global_orient": local_rot[:, 0],
        "body_pose": local_rot[:, 1:22],
        "betas": betas.view(-1, 10).repeat(num_frames, 1),
        "left_hand_pose": pca_to_matrix(motion[:, 207:219], body_model.left_hand_components),
        "right_hand_pose": pca_to_matrix(motion[:, 219:231], body_model.right_hand_components),
    }
    body_root_offset = body_model.forward(betas=betas.view(-1, 10)).joints[0, 0]
    params["transl"] = global_trans[:, 0] - body_root_offset[None]
    return camera_T, params, global_trans
