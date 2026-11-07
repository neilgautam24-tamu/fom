"""Extract SMPL-X hand sample points in world coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# OpenPose/SMPL-X body joint indices (wrist + finger chain per hand).
LEFT_HAND_JOINT_IDS = [20, *range(25, 40)]
RIGHT_HAND_JOINT_IDS = [21, *range(40, 55)]
NUM_HAND_JOINT_SAMPLES = len(LEFT_HAND_JOINT_IDS)


@dataclass(frozen=True)
class HandSampleSpec:
    name: str
    joint_ids: tuple[int, ...]
    vertex_indices: tuple[int, ...] | None = None


def hand_vertex_indices(model: Any, *, weight_threshold: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Vertices strongly influenced by left / right hand joint chains."""
    weights = model.lbs_weights.detach().cpu().numpy()
    left_joints = np.arange(20, 40)
    right_joints = np.arange(21, 55)
    left = np.where(weights[:, left_joints].sum(axis=1) >= weight_threshold)[0]
    right = np.where(weights[:, right_joints].sum(axis=1) >= weight_threshold)[0]
    return left.astype(np.int32), right.astype(np.int32)


def default_hand_specs(model: Any, *, use_vertices: bool = True) -> tuple[HandSampleSpec, HandSampleSpec]:
    left_verts = right_verts = None
    if use_vertices:
        lv, rv = hand_vertex_indices(model)
        left_verts = tuple(int(i) for i in lv)
        right_verts = tuple(int(i) for i in rv)
    return (
        HandSampleSpec("left", tuple(LEFT_HAND_JOINT_IDS), left_verts),
        HandSampleSpec("right", tuple(RIGHT_HAND_JOINT_IDS), right_verts),
    )


def _pad_batch(arr: np.ndarray, batch_size: int) -> tuple[np.ndarray, int]:
    n = int(arr.shape[0])
    if n >= batch_size:
        return arr[:batch_size], batch_size
    pad = np.repeat(arr[-1:], batch_size - n, axis=0)
    return np.concatenate([arr, pad], axis=0), n


@torch.no_grad()
def precompute_hand_points(
    model: Any,
    motion: dict[str, np.ndarray],
    device: torch.device,
    *,
    batch_size: int = 256,
    use_vertices: bool = True,
) -> dict[str, np.ndarray]:
    """Run batched SMPL-X and return per-frame hand sample points.

    Returns
    -------
    dict with keys ``left_points``, ``right_points`` each ``(N, P, 3)`` float32.
    """
    left_spec, right_spec = default_hand_specs(model, use_vertices=use_vertices)
    n = int(motion["num_frames"])
    bs = max(1, int(batch_size))

    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []

    betas_full = motion["betas"].astype(np.float32)
    if betas_full.ndim == 1:
        betas_full = np.broadcast_to(betas_full, (n, betas_full.shape[0])).copy()

    for start in range(0, n, bs):
        end = min(start + bs, n)
        chunk = end - start

        def _take(key: str) -> tuple[torch.Tensor, int]:
            arr = motion[key][start:end].astype(np.float32)
            padded, valid = _pad_batch(arr, bs)
            return torch.from_numpy(padded).to(device), valid

        go, go_n = _take("global_orient")
        bp, _ = _take("body_pose")
        tr, _ = _take("transl")
        lh, _ = _take("left_hand_pose")
        rh, _ = _take("right_hand_pose")
        jaw, _ = _take("jaw_pose")
        leye, _ = _take("leye_pose")
        reye, _ = _take("reye_pose")
        expr, _ = _take("expression")
        bt_arr, _ = _pad_batch(betas_full[start:end], bs)
        bt = torch.from_numpy(bt_arr).to(device)

        out = model(
            betas=bt,
            global_orient=go,
            body_pose=bp,
            transl=tr,
            left_hand_pose=lh,
            right_hand_pose=rh,
            jaw_pose=jaw,
            leye_pose=leye,
            reye_pose=reye,
            expression=expr,
        )
        joints = out.joints[:go_n].detach().cpu().numpy()
        verts = out.vertices[:go_n].detach().cpu().numpy()

        left_pts = _gather_hand_points(joints, verts, left_spec)
        right_pts = _gather_hand_points(joints, verts, right_spec)
        left_parts.append(left_pts)
        right_parts.append(right_pts)

        if (start // bs) % 20 == 0 or end == n:
            print(f"[hands]   {end:,}/{n:,}", flush=True)

    return {
        "left": np.concatenate(left_parts, axis=0).astype(np.float32),
        "right": np.concatenate(right_parts, axis=0).astype(np.float32),
    }


def _gather_hand_points(
    joints: np.ndarray,
    vertices: np.ndarray,
    spec: HandSampleSpec,
) -> np.ndarray:
    """Return ``(B, P, 3)`` sample points for one hand."""
    j = joints[:, spec.joint_ids, :]
    if spec.vertex_indices:
        v = vertices[:, list(spec.vertex_indices), :]
        return np.concatenate([j, v], axis=1)
    return j
