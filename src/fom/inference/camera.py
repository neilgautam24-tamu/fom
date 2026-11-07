"""ViPE camera poses -> FoM trajectory conditioning.

ViPE returns OpenCV camera-to-world poses (world Y down, camera Y down / Z forward). The model
was trained on Aria poses in a Z-up world whose camera frame is Y-up / Z-forward, so we:

1. remap the world to Z-up (optionally with the gravity-flipped map, see below),
2. right-multiply ``Rz(180)`` so the local camera axes match the Aria convention,
3. resample to the motion rate, canonicalize (yaw + XY origin at frame 0) and put the floor at
   ``z = 0`` assuming the first camera is at standing eye height,
4. encode the 18-D trajectory features (see :mod:`fom.body.representation`).
"""

from __future__ import annotations

import numpy as np
import torch

from fom.body.representation import encode_trajectory
from fom.utils import rotation_conversions as rc

HEAD_HEIGHT_M = 1.55  # assumed eye height of the first frame

# OpenCV world (Y down, Z forward) -> Z-up:  X = X_v, Y = Z_v, Z = -Y_v
WORLD_TO_ZUP = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
# Monocular ViPE occasionally flips gravity on generated clips:  X = X_v, Y = -Z_v, Z = Y_v
WORLD_TO_ZUP_FLIPPED = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
# OpenCV camera axes -> Aria-like (Y up, Z forward). Only used with WORLD_TO_ZUP.
RZ180 = torch.tensor(
    [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)

_WANT_DOWN = ("descend", "sit_sofa", "sit_chair", "_sit_")
_WANT_UP = ("ascend",)


def infer_gravity_flip(clip_name: str, height_change: float) -> bool:
    """Heuristic for generated clips: flip when the vertical motion contradicts the prompt name."""
    name = (clip_name or "").lower()
    if any(k in name for k in _WANT_DOWN) and height_change > 0.2:
        return True
    return any(k in name for k in _WANT_UP) and height_change < -0.2


def zup_world_matrix(flip_gravity: bool = False, level: torch.Tensor | None = None) -> torch.Tensor:
    """4x4 map from the ViPE (OpenCV) world to the Z-up world.

    Args:
        level: optional ``(3, 3)`` rotation applied in the OpenCV world first (floor levelling).
    """
    world = torch.eye(4)
    if flip_gravity:
        world[:3, :3] = WORLD_TO_ZUP_FLIPPED
    else:
        world[:3, :3] = WORLD_TO_ZUP @ (level.float() if level is not None else torch.eye(3))
    return world


def opencv_to_zup(c2w: torch.Tensor, flip_gravity: bool = False, level: torch.Tensor | None = None) -> torch.Tensor:
    """``(T, 4, 4)`` OpenCV camera-to-world -> Z-up world with Aria-like camera axes.

    The camera axes are rotated by ``Rz(180)`` except with the gravity-flipped world map, which
    already yields Y-up camera axes.
    """
    world = zup_world_matrix(flip_gravity, level).to(c2w.dtype)
    if flip_gravity:
        return world[None] @ c2w
    return world[None] @ c2w @ RZ180.to(c2w.dtype)[None]


def resample_poses(c2w: torch.Tensor, src_fps: float, dst_fps: float) -> torch.Tensor:
    """Linear interpolation of translation and 6D rotation between frame rates."""
    num = c2w.shape[0]
    n_out = max(2, int(round((num - 1) / src_fps * dst_fps)) + 1)
    t_src = np.linspace(0.0, 1.0, num)
    t_dst = np.linspace(0.0, 1.0, n_out)

    def interp(x: torch.Tensor) -> torch.Tensor:
        cols = [torch.from_numpy(np.interp(t_dst, t_src, x[:, c].cpu().numpy())) for c in range(x.shape[1])]
        return torch.stack(cols, dim=1).to(dtype=c2w.dtype)

    out = torch.eye(4, dtype=c2w.dtype)[None].repeat(n_out, 1, 1)
    out[:, :3, :3] = rc.rotation_6d_to_matrix(interp(rc.matrix_to_rotation_6d(c2w[:, :3, :3])))
    out[:, :3, 3] = interp(c2w[:, :3, 3])
    return out


def floor_shift(height: float) -> torch.Tensor:
    shift = torch.eye(4)
    shift[2, 3] = -height
    return shift


def trajectory_features(canonical_c2w: torch.Tensor, floor_height: float) -> torch.Tensor:
    """Un-normalized ``(T, 18)`` trajectory features of canonicalized poses."""
    floored = canonical_c2w.clone()
    floored[:, 2, 3] -= floor_height
    return encode_trajectory(floored)
