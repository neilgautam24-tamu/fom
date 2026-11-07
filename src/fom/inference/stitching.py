"""Rigid transforms and cross-fading of SMPL-X sequences predicted in overlapping windows."""

from __future__ import annotations

import torch

from fom.utils import rotation_conversions as rc

ROT_KEYS = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "left_hand_pose", "right_hand_pose")


def transform_smplx(params: dict[str, torch.Tensor], transform: torch.Tensor, pelvis: torch.Tensor) -> dict:
    """Move an SMPL-X sequence rigidly by ``transform`` (4x4).

    The root rotation acts about the pelvis ``pelvis`` (rest pose, (3,)), hence the translation fix.
    """
    rot, trans = transform[:3, :3], transform[:3, 3]
    out = dict(params)
    out["global_orient"] = rot[None] @ params["global_orient"]
    out["transl"] = (params["transl"] + pelvis) @ rot.T + trans - pelvis
    return out


def nlerp_rotations(rot_a: torch.Tensor, rot_b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Blend ``(T, ..., 3, 3)`` rotations towards ``rot_b`` with per-frame ``weight`` ``(T,)``."""
    qa, qb = rc.matrix_to_quaternion(rot_a), rc.matrix_to_quaternion(rot_b)
    qb = torch.where((qa * qb).sum(-1, keepdim=True) < 0, -qb, qb)
    w = weight.view(-1, *([1] * (qa.ndim - 1)))
    q = (1 - w) * qa + w * qb
    return rc.quaternion_to_matrix(q / q.norm(dim=-1, keepdim=True))


def stitch_windows(windows: list[dict], num_frames: int) -> dict:
    """Concatenate per-window sequences already expressed in one frame, cross-fading overlaps.

    Each window is ``{"start", "end", "smpl": {key: (L, ...)}, "camera": (L, 4, 4)}``.
    """
    seq = {k: v.clone() for k, v in windows[0]["smpl"].items()}
    camera = windows[0]["camera"].clone()
    for prev, cur in zip(windows[:-1], windows[1:]):
        start, overlap = cur["start"], prev["end"] - cur["start"]
        assert seq["transl"].shape[0] == prev["end"]
        weight = (torch.arange(overlap, dtype=torch.float32) + 0.5) / overlap  # 0 -> 1 across the overlap
        for key, value in seq.items():
            a, b = value[start:], cur["smpl"][key][:overlap]
            if key in ROT_KEYS:
                mixed = nlerp_rotations(a, b, weight)
            else:
                w = weight.view(-1, *([1] * (a.ndim - 1)))
                mixed = (1 - w) * a + w * b
            seq[key] = torch.cat([value[:start], mixed, cur["smpl"][key][overlap:]], 0)
        # The camera comes from the shared conditioning trajectory, so no blending is needed.
        camera = torch.cat([camera[: start + overlap], cur["camera"][overlap:]], 0)
    assert seq["transl"].shape[0] == num_frames
    return {"smpl": seq, "camera": camera}
