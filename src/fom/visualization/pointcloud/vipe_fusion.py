#!/usr/bin/env python3
"""Fuse ViPE RGB-D of validation clips into the GT camera frame and cache it for the viewers.

ViPE depth is stored as zipped EXR (``.zip``), RGB as ``.mp4``, poses/intrinsics as ``.npz``.
Run in the ViPE environment (OpenEXR is needed to read depth). Each ViPE camera path is Sim(3)
aligned to the ground-truth camera trajectory before fusion.

    python -m fom.visualization.pointcloud.vipe_fusion --items items.json --vipe-root <vipe_results> \\
        --gt gt.pkl --out <cache>/vipe_pcd

``items.json`` is a list of ``{"pred_key": ..., "vipe_name": optional ViPE output name}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fom.evaluation.io import load_sequences, sequence_key
from fom.visualization.common import viz_cache_dir, voxel_downsample


def sim3_align(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_zero = pred - pred_mean
    target_zero = target - target_mean
    covariance = pred_zero.T @ target_zero / len(pred)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    pred_variance = np.sum(pred_zero**2) / len(pred)
    scale = float(np.sum(singular_values * np.diag(correction)) / max(pred_variance, 1e-12))
    translation = target_mean - scale * (pred_mean @ rotation)
    return scale, rotation, translation




def load_vipe_stack(vipe_dir: Path, name: str):
    from vipe.utils.io import (
        read_depth_artifacts,
        read_intrinsics_artifacts,
        read_pose_artifacts,
        read_rgb_artifacts,
    )

    depth_zip = vipe_dir / "depth" / f"{name}.zip"
    rgb_mp4 = vipe_dir / "rgb" / f"{name}.mp4"
    pose_npz = vipe_dir / "pose" / f"{name}.npz"
    intr_npz = vipe_dir / "intrinsics" / f"{name}.npz"
    cam_txt = vipe_dir / "intrinsics" / f"{name}_camera.txt"

    depths = []
    for _, d in read_depth_artifacts(depth_zip):
        depths.append(d.numpy().astype(np.float32))
    depth = np.stack(depths, axis=0)

    rgbs = []
    for _, img in read_rgb_artifacts(rgb_mp4):
        arr = (img.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        rgbs.append(arr)
    rgb = np.stack(rgbs, axis=0)

    _, se3 = read_pose_artifacts(pose_npz)
    c2w = np.asarray(se3.matrix().numpy(), dtype=np.float64)

    _, intr_t, _ = read_intrinsics_artifacts(
        intr_npz, cam_txt if cam_txt.exists() else None
    )
    intr = intr_t.numpy().astype(np.float32)
    return depth, rgb, intr, c2w


def fuse_aligned(
    depth: np.ndarray,
    rgb: np.ndarray,
    intr: np.ndarray,
    c2w: np.ndarray,
    *,
    temporal_subsample: int = 3,
    spatial_subsample: int = 4,
    min_depth: float = 0.3,
    max_depth: float = 8.0,
    voxel: float = 0.04,
    max_points: int = 150_000,
) -> tuple[np.ndarray, np.ndarray]:
    T = min(len(depth), len(rgb), len(intr), len(c2w))
    ss = max(int(spatial_subsample), 1)
    ts = max(int(temporal_subsample), 1)
    chunks_xyz: list[np.ndarray] = []
    chunks_rgb: list[np.ndarray] = []

    for ti in range(0, T, ts):
        d = depth[ti]
        img = rgb[ti]
        fx, fy, cx, cy = intr[ti][:4]
        h, w = d.shape[:2]
        us = np.arange(0, w, ss, dtype=np.float32)
        vs = np.arange(0, h, ss, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)
        dd = d[::ss, ::ss]
        # RGB may differ slightly in size from depth - resize-ish by crop/stride
        img_ss = img[:: max(1, img.shape[0] // dd.shape[0] or 1), :: max(1, img.shape[1] // dd.shape[1] or 1)]
        img_ss = img_ss[: dd.shape[0], : dd.shape[1]]
        if img_ss.shape[:2] != dd.shape[:2]:
            # nearest pad/crop
            out = np.zeros((*dd.shape, 3), dtype=np.uint8)
            hh, ww = min(dd.shape[0], img_ss.shape[0]), min(dd.shape[1], img_ss.shape[1])
            out[:hh, :ww] = img_ss[:hh, :ww]
            img_ss = out
        valid = np.isfinite(dd) & (dd >= min_depth) & (dd <= max_depth)
        if not np.any(valid):
            continue
        u, v, z = uu[valid], vv[valid], dd[valid]
        x = (u - cx) / max(float(fx), 1e-6) * z
        y = (v - cy) / max(float(fy), 1e-6) * z
        p_cam = np.stack([x, y, z], axis=1)
        R = c2w[ti, :3, :3]
        t = c2w[ti, :3, 3]
        p_w = (R @ p_cam.T).T + t
        chunks_xyz.append(p_w.astype(np.float32))
        chunks_rgb.append(img_ss[valid].astype(np.uint8))

    if not chunks_xyz:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    xyz = np.concatenate(chunks_xyz, axis=0)
    cols = np.concatenate(chunks_rgb, axis=0)
    return voxel_downsample(xyz, cols, voxel, max_points)


def safe_name(pred_key: str) -> str:
    return pred_key.replace("/", "_")


def process_one(item: dict, vipe_root: Path, gt: dict, out_dir: Path) -> Path | None:
    key = sequence_key(item["pred_key"])
    name = item.get("vipe_name") or safe_name(key)
    vipe_dir = vipe_root / name
    if not (vipe_dir / "DONE").is_file():
        print(f"[skip] VIPE not done: {name}")
        return None
    if key not in gt:
        print(f"[skip] no GT for {key}")
        return None

    gt_aria = gt[key]["camera_T"]
    if hasattr(gt_aria, "detach"):
        gt_aria = gt_aria.detach().cpu().numpy()
    gt_aria = np.asarray(gt_aria, dtype=np.float64)

    depth, rgb, intr, c2w_vipe = load_vipe_stack(vipe_dir, name)
    T = min(len(gt_aria), len(c2w_vipe), len(depth), len(rgb), len(intr))
    gt_t = gt_aria[:T, :3, 3]
    vipe_t = c2w_vipe[:T, :3, 3]
    scale, row_R, _ = sim3_align(vipe_t, gt_t)
    R_world = row_R.T
    t_offset = gt_t.mean(axis=0) - scale * (R_world @ vipe_t.mean(axis=0))

    c2w_aligned = np.zeros((T, 4, 4), dtype=np.float64)
    for i in range(T):
        c2w_aligned[i, :3, :3] = R_world @ c2w_vipe[i, :3, :3]
        c2w_aligned[i, :3, 3] = scale * (R_world @ c2w_vipe[i, :3, 3]) + t_offset
        c2w_aligned[i, 3, 3] = 1.0

    xyz, cols = fuse_aligned(depth[:T], rgb[:T], intr[:T], c2w_aligned)
    out = out_dir / f"{safe_name(key)}.npz"
    np.savez_compressed(
        out,
        xyz=xyz,
        rgb=cols,
        pred_key=np.asarray(key),
        scale=np.float32(scale),
        n_points=np.int32(xyz.shape[0]),
    )
    print(f"[pcd] {key[:70]}  points={xyz.shape[0]:,}  scale={scale:.3f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, required=True, help='JSON list of {"pred_key", "vipe_name"?}')
    parser.add_argument("--vipe-root", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True, help="fom-gt output")
    parser.add_argument("--out", type=Path, default=viz_cache_dir("vipe_pcd"))
    args = parser.parse_args()

    items = json.loads(args.items.read_text())
    for it in items:
        it.setdefault("vipe_name", safe_name(it["pred_key"]))

    print(f"Loading GT from {args.gt}")
    gt = load_sequences(args.gt)
    args.out.mkdir(parents=True, exist_ok=True)

    ok = 0
    for item in items:
        try:
            if process_one(item, args.vipe_root, gt, args.out) is not None:
                ok += 1
        except Exception as exc:
            print(f"[fail] {item.get('pred_key')}: {exc}")
    print(f"Done: {ok}/{len(items)} PCD caches in {args.out}")


if __name__ == "__main__":
    main()
