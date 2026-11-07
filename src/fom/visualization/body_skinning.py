#!/usr/bin/env python3
"""Export an SMPL-X prediction as a skinned-mesh rig for viser (CPU).

Input : motion_smplx_vipe_world.npz from :mod:`fom.inference.run`
Output: body_skinning.npz with the rest mesh, skin weights, per-frame bone poses (world wxyz +
        position for the 55 SMPL-X joints) and a per-frame "head pinned to camera" offset.
The viewer sends the mesh once and only updates bone transforms per frame.

Usage:
    python -m fom.visualization.body_skinning <motion_smplx_vipe_world.npz> <out.npz>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from fom.body.smplx_model import JOINT_LEFT_EYE, load_smplx
from fom.utils import rotation_conversions as rc

N_JOINTS = 55  # 22 body + jaw + 2 eyes + 2x15 hand joints
LEYE = JOINT_LEFT_EYE


@torch.inference_mode()
def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    m = np.load(src)
    smpl = load_smplx()
    N = len(m["time_s"])
    betas = torch.from_numpy(m["betas"][:1]).float()

    rest = smpl.forward(betas=betas)
    rest_verts = rest.vertices[0].numpy()
    rest_joints = rest.joints[0, :N_JOINTS].numpy()
    parents = smpl.parents[:N_JOINTS].numpy()
    weights = smpl.lbs_weights[:, :N_JOINTS].numpy()

    # Local joint rotations in SMPL-X order, then forward kinematics to world rotations.
    local = torch.cat(
        [torch.from_numpy(m[k]).float().reshape(N, -1, 3, 3) for k in
         ["global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "left_hand_pose", "right_hand_pose"]],
        dim=1,
    )
    assert local.shape[1] == N_JOINTS, local.shape
    world = [local[:, 0]]
    for j in range(1, N_JOINTS):
        world.append(world[parents[j]] @ local[:, j])
    world = torch.stack(world, 1)  # (N, 55, 3, 3)
    bone_wxyz = rc.matrix_to_quaternion(world).numpy()  # (N, 55, 4) wxyz
    bone_pos = m["joints"][:, :N_JOINTS].astype(np.float32)  # posed joint positions (world)

    # Check: linear-blend skinning from the rig vs the saved SMPL-X vertices (differs only by pose correctives).
    idx = np.linspace(0, N - 1, 8).astype(int)
    R = world[idx].numpy()
    v = np.einsum("vj,fjab,vb->fva", weights, R, rest_verts) - np.einsum("vj,fjab,jb->fva", weights, R, rest_joints) \
        + np.einsum("vj,fja->fva", weights, bone_pos[idx])
    err = np.linalg.norm(v - m["vertices"][idx], axis=-1)
    print(f"rig vs saved vertices: mean {err.mean() * 100:.2f} cm, max {err.max() * 100:.2f} cm")

    pin_offset = (m["camera_c2w"][:, :3, 3] - bone_pos[:, LEYE]).astype(np.float32)  # move left eye onto camera
    np.savez(
        dst,
        rest_vertices=rest_verts.astype(np.float32),
        faces=m["faces"].astype(np.int32),
        skin_weights=weights.astype(np.float32),
        rest_joints=rest_joints.astype(np.float32),
        parents=parents,
        bone_wxyz=bone_wxyz.astype(np.float32),
        bone_pos=bone_pos,
        pin_offset=pin_offset,
        fps=float(m["fps"]),
        src_frame_index=m["src_frame_index"],
    )
    print(f"wrote {dst} ({N} frames)")


if __name__ == "__main__":
    main()
