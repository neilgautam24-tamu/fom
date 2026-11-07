#!/usr/bin/env python3
"""Offline paper-teaser renderer: RGB scene PCD + GT/pred SMPL + Aria traj.

Improvements over the first pass:
  - Motion-aware auto camera (side of trajectory, elevated, full path in frame)
  - Sparse, evenly spaced ghost poses (GT-only by default) with clearer alphas
  - PCD cropped around the motion corridor; smaller sphere splats
  - Batch mode: all eval sequences x {VIPE RGB PCD, GT MPS + ego-RGB}

Example (single)::

    python -m fom.visualization.figures.teaser_render --preds preds.pkl --gt gt.pkl \\
      --eval-list keys.json --seq-idx 0 --frame-frac 0.65 --auto-camera \\
      --pcd-source vipe --out teaser.png

Example (batch, both point-cloud sources)::

    python -m fom.visualization.figures.teaser_render ... --all --pcd-modes vipe,gt_mps
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import trimesh
from PIL import Image

from fom.body.smplx_model import load_smplx
from fom.evaluation.io import load_sequences, sequence_key
from fom.visualization.common import find_bundle, to_np, viz_cache_dir

VIZ_DATA = viz_cache_dir("teaser").parent
MPS_PCD_DIR = viz_cache_dir("mps_pcd")
VIPE_PCD_DIR = viz_cache_dir("vipe_pcd")
TEASER_DIR = viz_cache_dir("teaser")
MOTION_FPS = 15

GT_MESH = (0.20, 0.78, 0.40)
PRED_MESH = (0.20, 0.50, 0.98)
GT_TRAJ = (0.10, 0.62, 0.28)
PRED_TRAJ = (0.15, 0.55, 0.95)
GHOST_GT = (0.15, 0.55, 0.32)
GHOST_PRED = (0.20, 0.40, 0.80)




def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    n = np.linalg.norm(forward)
    forward = forward / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        up = np.array([0.0, 1.0, 0.0]) if abs(up[2]) > 0.7 else np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        rn = np.linalg.norm(right)
    right = right / max(rn, 1e-9)
    up = np.cross(right, forward)
    up = up / max(np.linalg.norm(up), 1e-9)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def scene_up_axis(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.array([0.0, 0.0, 1.0])
    span = pts.max(0) - pts.min(0)
    if span[2] >= span[1] * 0.35:
        return np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 1.0, 0.0])


def motion_basis(traj_xyz: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (forward, side) unit vectors from trajectory motion."""
    pts = np.asarray(traj_xyz, dtype=np.float64)
    up = up / max(np.linalg.norm(up), 1e-9)
    if len(pts) < 2:
        fwd = np.array([1.0, 0.0, 0.0])
    else:
        # Robust direction: end-start, fall back to PCA of path.
        fwd = pts[-1] - pts[0]
        if np.linalg.norm(fwd) < 0.15:
            x = pts - pts.mean(0)
            _, _, vt = np.linalg.svd(x, full_matrices=False)
            fwd = vt[0]
        # Remove vertical component.
        fwd = fwd - np.dot(fwd, up) * up
        if np.linalg.norm(fwd) < 1e-6:
            fwd = np.array([1.0, 0.0, 0.0])
        fwd = fwd / np.linalg.norm(fwd)
    side = np.cross(up, fwd)
    sn = np.linalg.norm(side)
    if sn < 1e-6:
        side = np.array([0.0, 1.0, 0.0])
        sn = 1.0
    side = side / sn
    return fwd, side


def auto_camera_pose(
    content_pts: np.ndarray,
    traj_xyz: np.ndarray,
    *,
    elev_deg: float,
    radius_scale: float,
    side_sign: float,
    yfov_deg: float,
    aspect: float,
) -> tuple[np.ndarray, dict]:
    """Side-of-path camera that frames the full motion corridor."""
    pts = np.asarray(content_pts, dtype=np.float64)
    traj = np.asarray(traj_xyz, dtype=np.float64)
    up = scene_up_axis(pts)
    fwd, side = motion_basis(traj, up)
    side = side * float(side_sign)

    center = 0.55 * np.median(pts, axis=0) + 0.45 * traj.mean(axis=0)
    # Span along path / across / vertical for framing.
    rel = pts - center
    span_fwd = float(np.percentile(np.abs(rel @ fwd), 95) * 2)
    span_side = float(np.percentile(np.abs(rel @ side), 95) * 2)
    span_up = float(np.percentile(np.abs(rel @ up), 95) * 2)
    # Distance so the larger of width/height fits with margin.
    yfov = np.deg2rad(yfov_deg)
    xfov = 2.0 * np.arctan(np.tan(yfov * 0.5) * aspect)
    need_h = (max(span_fwd, span_side) * 0.55) / max(np.tan(xfov * 0.5), 1e-6)
    need_v = (max(span_up, 1.6) * 0.65) / max(np.tan(yfov * 0.5), 1e-6)
    radius = float(max(need_h, need_v, 2.2) * radius_scale)

    elev = np.deg2rad(elev_deg)
    # Mostly from the side; tiny rearward bias so path reads left->right.
    eye = (
        center
        + side * (radius * np.cos(elev))
        + up * (radius * np.sin(elev))
        - fwd * (radius * 0.05)
    )
    pose = look_at(eye, center, up)
    meta = {
        "center": center.tolist(),
        "eye": eye.tolist(),
        "radius": radius,
        "fwd": fwd.tolist(),
        "side": side.tolist(),
        "up": up.tolist(),
        "span_fwd": span_fwd,
        "span_side": span_side,
        "span_up": span_up,
    }
    return pose, meta


def orbit_camera(
    center: np.ndarray,
    *,
    radius: float,
    elev_deg: float,
    azim_deg: float,
    up: np.ndarray,
) -> np.ndarray:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    up = up / max(np.linalg.norm(up), 1e-9)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    east = np.cross(up, tmp)
    east = east / max(np.linalg.norm(east), 1e-9)
    north = np.cross(up, east)
    direction = (
        np.cos(elev) * (np.cos(azim) * east + np.sin(azim) * north) + np.sin(elev) * up
    )
    eye = center + radius * direction
    return look_at(eye, center, up=up)


def tube_mesh(
    path: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
    *,
    sections: int = 8,
) -> trimesh.Trimesh | None:
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None
    # Subsample very long paths for speed.
    if len(pts) > 120:
        idx = np.linspace(0, len(pts) - 1, 120).round().astype(int)
        pts = pts[idx]
    parts = []
    rgba = np.array([*color, 1.0])
    for a, b in zip(pts[:-1], pts[1:]):
        vec = b - a
        length = float(np.linalg.norm(vec))
        if length < 1e-5:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
        z = vec / length
        tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(tmp, z)
        x = x / max(np.linalg.norm(x), 1e-9)
        y = np.cross(z, x)
        T = np.eye(4)
        T[:3, 0] = x
        T[:3, 1] = y
        T[:3, 2] = z
        T[:3, 3] = (a + b) * 0.5
        cyl.apply_transform(T)
        cyl.visual.vertex_colors = np.tile((rgba * 255).astype(np.uint8), (len(cyl.vertices), 1))
        parts.append(cyl)
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def colored_body_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    color: tuple[float, float, float],
    *,
    alpha: float = 1.0,
) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=False)
    rgba = np.array([*color, alpha], dtype=np.float64)
    mesh.visual.vertex_colors = np.tile((rgba * 255).astype(np.uint8), (len(mesh.vertices), 1))
    return mesh


def to_pyrender_mesh(tri: trimesh.Trimesh, *, alpha: float = 1.0):
    import pyrender

    if alpha >= 0.999:
        return pyrender.Mesh.from_trimesh(tri, smooth=True)
    vc = np.asarray(tri.visual.vertex_colors)
    if vc.ndim == 2 and len(vc):
        base = (vc[0].astype(np.float64) / 255.0).tolist()
        if len(base) == 3:
            base = [*base, alpha]
        else:
            base[3] = alpha
    else:
        base = [0.8, 0.8, 0.8, alpha]
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=base,
        metallicFactor=0.0,
        roughnessFactor=0.9,
        alphaMode="BLEND",
    )
    return pyrender.Mesh.from_trimesh(tri, material=mat, smooth=True)


def points_as_spheres(
    xyz: np.ndarray,
    rgb: np.ndarray,
    radius: float,
    *,
    max_points: int = 50_000,
    brightness: float = 1.15,
) -> trimesh.Trimesh | None:
    if xyz.shape[0] == 0 or radius <= 0:
        return None
    pts = np.asarray(xyz, np.float32)
    cols = np.asarray(rgb, np.uint8)
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(pts.shape[0], max_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    cols_f = np.clip(cols.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    template = trimesh.creation.icosphere(subdivisions=0, radius=float(radius))
    tv = np.asarray(template.vertices, np.float32)
    tf = np.asarray(template.faces, np.int64)
    n, vcount = pts.shape[0], tv.shape[0]
    verts = (tv[None, :, :] + pts[:, None, :]).reshape(-1, 3)
    faces = (tf[None, :, :] + (np.arange(n, dtype=np.int64) * vcount)[:, None, None]).reshape(
        -1, 3
    )
    vcols = np.repeat(cols_f, vcount, axis=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.vertex_colors = np.concatenate(
        [vcols, np.full((len(vcols), 1), 255, np.uint8)], axis=1
    )
    return mesh


def crop_pcd_to_motion(
    xyz: np.ndarray,
    rgb: np.ndarray,
    traj_xyz: np.ndarray,
    body_pts: np.ndarray,
    *,
    margin: float = 2.2,
    max_points: int = 50_000,
    body_clearance: float = 0.32,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep PCD near the motion corridor so the figure isn't buried in walls."""
    if xyz.shape[0] == 0:
        return xyz, rgb
    anchor = np.concatenate([traj_xyz, body_pts], axis=0)
    lo = anchor.min(0) - margin
    hi = anchor.max(0) + margin
    up = scene_up_axis(anchor)
    if abs(up[2]) > 0.7:
        lo[2] = anchor[:, 2].min() - 0.3
        hi[2] = anchor[:, 2].max() + 1.6
    elif abs(up[1]) > 0.7:
        lo[1] = anchor[:, 1].min() - 0.3
        hi[1] = anchor[:, 1].max() + 1.6
    mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    if traj_xyz.shape[0] >= 2:
        t = traj_xyz.astype(np.float32)
        if len(t) > 40:
            t = t[np.linspace(0, len(t) - 1, 40).round().astype(int)]
        d2 = ((xyz[:, None, :] - t[None, :, :]) ** 2).sum(-1).min(axis=1)
        mask &= d2 <= (margin * margin)

    # Carve out space around bodies so meshes/ghosts aren't buried in points.
    if body_pts.shape[0] > 0 and body_clearance > 0:
        b = body_pts.astype(np.float32)
        if len(b) > 800:
            rng = np.random.default_rng(0)
            b = b[rng.choice(len(b), 800, replace=False)]
        bd2 = ((xyz[:, None, :] - b[None, :, :]) ** 2).sum(-1).min(axis=1)
        mask &= bd2 >= (body_clearance * body_clearance)

    xyz_c, rgb_c = xyz[mask], rgb[mask]
    if xyz_c.shape[0] < 400:
        mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        xyz_c, rgb_c = xyz[mask], rgb[mask]
    if xyz_c.shape[0] > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(xyz_c.shape[0], max_points, replace=False)
        xyz_c, rgb_c = xyz_c[sel], rgb_c[sel]
    return xyz_c.astype(np.float32), rgb_c.astype(np.uint8)


def cull_pcd_near_camera(
    xyz: np.ndarray,
    rgb: np.ndarray,
    cam_eye: np.ndarray,
    focus_pts: np.ndarray,
    *,
    near_frac: float = 0.72,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop foreground points that sit between the camera and the subjects."""
    if xyz.shape[0] == 0:
        return xyz, rgb
    eye = np.asarray(cam_eye, np.float64)
    focus = np.asarray(focus_pts, np.float64)
    d_focus = np.linalg.norm(focus - eye, axis=1)
    cutoff = float(np.percentile(d_focus, 20) * near_frac)
    d = np.linalg.norm(xyz.astype(np.float64) - eye, axis=1)
    keep = d >= cutoff
    if keep.sum() < 200:
        return xyz, rgb
    return xyz[keep], rgb[keep]


def ghost_frame_indices(frame: int, count: int) -> list[int]:
    """Evenly spaced past frames, excluding the current frame."""
    if count <= 0 or frame <= 0:
        return []
    if count == 1:
        return [max(0, frame // 2)]
    raw = np.linspace(0, frame - 1, count).round().astype(int)
    uniq = []
    min_gap = max(1, frame // (count * 2))
    for t in raw.tolist():
        if not uniq or (t - uniq[-1]) >= min_gap:
            uniq.append(int(t))
        else:
            uniq[-1] = int(t)
    return [t for t in uniq if 0 <= t < frame]


def ghost_frame_indices_spatial(
    traj_xyz: np.ndarray,
    frame: int,
    count: int,
    *,
    min_sep: float = 0.45,
) -> list[int]:
    """Pick past frames spaced in path length so ghosts don't pile up."""
    if count <= 0 or frame <= 0:
        return []
    pts = np.asarray(traj_xyz[: frame + 1, :3], dtype=np.float64)
    if len(pts) < 2:
        return ghost_frame_indices(frame, count)
    # Cumulative path length.
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[frame]) if frame < len(cum) else float(cum[-1])
    if total < min_sep * 1.5:
        # Almost stationary - fall back to temporal spacing with fewer ghosts.
        return ghost_frame_indices(frame, max(1, min(count, 2)))
    targets = np.linspace(0.0, max(total - 0.05, 0.0), count + 1)[:-1]  # exclude end~current
    chosen = []
    for tgt in targets:
        t = int(np.argmin(np.abs(cum[:frame] - tgt)))
        if chosen:
            # Enforce spatial separation from last chosen.
            if np.linalg.norm(pts[t] - pts[chosen[-1]]) < min_sep:
                # Walk forward until separated or hit frame.
                t2 = t
                while t2 < frame - 1 and np.linalg.norm(pts[t2] - pts[chosen[-1]]) < min_sep:
                    t2 += 1
                t = t2
            if t in chosen or t >= frame:
                continue
        chosen.append(t)
    return chosen[:count]


def load_pcd(
    key: str,
    *,
    source: str,
    mps_dir: Path,
    vipe_dir: Path,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    stem = key.replace("/", "_")
    vipe_path = vipe_dir / f"{stem}.npz"
    mps_path = mps_dir / f"{stem}.npz"

    def _load(path: Path, tag: str):
        z = np.load(path)
        xyz = np.asarray(z["xyz"], np.float32)
        rgb = np.asarray(z["rgb"], np.uint8)
        if rgb.ndim == 1:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        if xyz.shape[0] > max_points:
            rng = np.random.default_rng(0)
            sel = rng.choice(xyz.shape[0], max_points, replace=False)
            xyz, rgb = xyz[sel], rgb[sel]
        return xyz, rgb, tag

    if source == "vipe":
        if not vipe_path.is_file():
            raise FileNotFoundError(f"VIPE PCD missing: {vipe_path}")
        return _load(vipe_path, "vipe")
    if source == "mps":
        if not mps_path.is_file():
            raise FileNotFoundError(f"MPS PCD missing: {mps_path}")
        return _load(mps_path, "mps")
    if vipe_path.is_file():
        return _load(vipe_path, "vipe")
    if mps_path.is_file():
        return _load(mps_path, "mps")
    raise FileNotFoundError(f"No PCD cache for {key}")


def project_ego_rgb_onto_pcd(
    xyz: np.ndarray,
    aria_traj_T: np.ndarray,
    bundle: Path,
    *,
    max_frames: int = 28,
    max_src_mb: float = 4000.0,
    cache_path: Path | None = None,
) -> np.ndarray | None:
    if cache_path is not None and cache_path.is_file():
        z = np.load(cache_path)
        if z["rgb"].shape[0] == xyz.shape[0]:
            return np.asarray(z["rgb"], np.uint8)

    rgb_path = bundle / "ego_rgb.npz"
    if not rgb_path.is_file():
        print(f"[project-rgb] no ego_rgb.npz at {bundle}", flush=True)
        return None
    size_mb = rgb_path.stat().st_size / 1e6
    if size_mb > max_src_mb:
        print(
            f"[project-rgb] skip {rgb_path.name} ({size_mb:.0f} MB > {max_src_mb:.0f} MB)",
            flush=True,
        )
        return None

    tr = np.load(bundle / "trajectory.npz")
    fov_y = float(tr["rgb_fov_y_rad"][0]) if "rgb_fov_y_rad" in tr.files else 1.0
    z = np.load(rgb_path, mmap_mode="r")
    frames = z["rgb"]
    n_src = int(frames.shape[0])
    n_traj = int(aria_traj_T.shape[0])
    if n_src <= 0 or n_traj <= 0:
        return None

    n_use = min(max_frames, n_traj, n_src)
    traj_idx = np.linspace(0, n_traj - 1, n_use).round().astype(int)
    src_idx = np.linspace(0, n_src - 1, n_use).round().astype(int)

    n_pts = xyz.shape[0]
    best_depth = np.full(n_pts, np.inf, dtype=np.float32)
    out_rgb = np.zeros((n_pts, 3), dtype=np.uint8)

    for ti, si in zip(traj_idx, src_idx):
        fr = np.asarray(frames[int(si)], dtype=np.uint8)
        upright = np.ascontiguousarray(np.rot90(fr, k=1))
        h, w = upright.shape[:2]
        fy = 0.5 * h / max(np.tan(0.5 * fov_y), 1e-6)
        fx = fy
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        c2w = np.asarray(aria_traj_T[int(ti)], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        Xh = np.concatenate([xyz.astype(np.float64), np.ones((n_pts, 1))], axis=1)
        Xc = (w2c @ Xh.T).T[:, :3]
        zc = Xc[:, 2]
        valid = zc > 0.05
        u = fx * (Xc[:, 0] / np.clip(zc, 1e-6, None)) + cx
        v = fy * (Xc[:, 1] / np.clip(zc, 1e-6, None)) + cy
        ui = np.nan_to_num(u, nan=-1.0).round().astype(np.int32)
        vi = np.nan_to_num(v, nan=-1.0).round().astype(np.int32)
        valid &= (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        closer = valid & (zc.astype(np.float32) < best_depth)
        if not np.any(closer):
            continue
        idxs = np.where(closer)[0]
        out_rgb[idxs] = upright[vi[idxs], ui[idxs]]
        best_depth[idxs] = zc[idxs].astype(np.float32)

    hit = np.isfinite(best_depth) & (best_depth < np.inf)
    print(
        f"[project-rgb] colored {int(hit.sum()):,}/{n_pts:,} points from {n_use} frames",
        flush=True,
    )
    if hit.sum() < max(100, n_pts // 50):
        return None
    out = out_rgb.copy()
    out[~hit] = np.array([170, 170, 170], dtype=np.uint8)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, rgb=out)
    return out


def prepare_motion(key: str, preds: dict, gt: dict, smpl):
    from fom.body.smplx_model import evaluate_smplx

    pred = preds[key]
    g = gt[key]
    pred_smpl = {k: torch.as_tensor(to_np(v)) for k, v in pred["smpl_params"].items()}
    gt_smpl = {k: torch.as_tensor(to_np(v)) for k, v in g["smpl_params"].items()}
    with torch.no_grad():
        _, pred_verts, _ = evaluate_smplx(smpl, pred_smpl)
        _, gt_verts, _ = evaluate_smplx(smpl, gt_smpl)
    pred_verts = pred_verts.cpu().numpy()
    gt_verts = gt_verts.cpu().numpy()
    n = min(len(pred_verts), len(gt_verts))
    return {
        "pred_verts": pred_verts[:n],
        "gt_verts": gt_verts[:n],
        "pred_aria": to_np(pred["camera_T"])[:n],
        "gt_aria": to_np(g["camera_T"])[:n],
        "n": n,
        "faces": np.asarray(smpl.faces),
    }


def render_scene(
    *,
    xyz: np.ndarray,
    rgb: np.ndarray,
    gt_verts: np.ndarray,
    pred_verts: np.ndarray,
    faces: np.ndarray,
    gt_aria: np.ndarray,
    pred_aria: np.ndarray,
    frame: int,
    width: int,
    height: int,
    elev: float,
    azim: float,
    radius_scale: float,
    auto_camera: bool,
    side_sign: float,
    pcd_sphere_radius: float,
    traj_radius: float,
    show_pred_traj: bool,
    show_gt_traj: bool,
    show_frustum: bool,
    fov_y: float,
    ghost_count: int,
    ghost_mode: str,
    bg: tuple[float, float, float],
) -> tuple[np.ndarray, dict]:
    import pyrender

    ghost_ts = ghost_frame_indices_spatial(gt_aria[:, :3, 3], frame, ghost_count, min_sep=0.50)
    content_chunks = [gt_verts[frame], pred_verts[frame], gt_aria[:, :3, 3], pred_aria[:, :3, 3]]
    for t in ghost_ts:
        content_chunks.append(gt_verts[t])
        content_chunks.append(pred_verts[t])
    content = np.concatenate(content_chunks, axis=0)
    aspect = width / max(height, 1)
    cam_meta = {}
    if auto_camera:
        cam_pose, cam_meta = auto_camera_pose(
            content,
            gt_aria[:, :3, 3],
            elev_deg=elev,
            radius_scale=radius_scale,
            side_sign=side_sign,
            yfov_deg=42.0,
            aspect=aspect,
        )
    else:
        center = np.median(content, axis=0).astype(np.float64)
        extent = np.percentile(np.linalg.norm(content - center, axis=1), 92)
        radius = float(max(extent, 1.5) * radius_scale)
        up = scene_up_axis(content)
        cam_pose = orbit_camera(center, radius=radius, elev_deg=elev, azim_deg=azim, up=up)
        cam_meta = {"mode": "orbit", "azim": azim, "elev": elev, "radius": radius}

    # Cull foreground PCD using the chosen camera.
    xyz, rgb = cull_pcd_near_camera(xyz, rgb, cam_pose[:3, 3], content)

    scene = pyrender.Scene(bg_color=[*bg, 1.0], ambient_light=(0.62, 0.62, 0.62))

    if xyz.shape[0] > 0:
        if pcd_sphere_radius > 0:
            pc_mesh = points_as_spheres(xyz, rgb, pcd_sphere_radius)
            if pc_mesh is not None:
                scene.add(pyrender.Mesh.from_trimesh(pc_mesh, smooth=False), name="pcd")
        else:
            cols = rgb.astype(np.float32) / 255.0
            if cols.shape[1] == 3:
                cols = np.concatenate([cols, np.ones((len(cols), 1), np.float32)], axis=1)
            scene.add(pyrender.Mesh.from_points(xyz.astype(np.float32), colors=cols), name="pcd")

    for i, t in enumerate(ghost_ts):
        alpha = 0.42 + 0.40 * ((i + 1) / max(len(ghost_ts), 1))
        if ghost_mode in ("gt", "both"):
            scene.add(
                to_pyrender_mesh(
                    colored_body_mesh(gt_verts[t], faces, GHOST_GT, alpha=alpha),
                    alpha=alpha,
                ),
                name=f"gt_ghost_{t}",
            )
        if ghost_mode in ("pred", "both"):
            scene.add(
                to_pyrender_mesh(
                    colored_body_mesh(pred_verts[t], faces, GHOST_PRED, alpha=alpha),
                    alpha=alpha,
                ),
                name=f"pred_ghost_{t}",
            )

    scene.add(
        to_pyrender_mesh(colored_body_mesh(gt_verts[frame], faces, GT_MESH, alpha=1.0)),
        name="gt_body",
    )
    scene.add(
        to_pyrender_mesh(colored_body_mesh(pred_verts[frame], faces, PRED_MESH, alpha=1.0)),
        name="pred_body",
    )

    if show_gt_traj:
        tm = tube_mesh(gt_aria[:, :3, 3], traj_radius, GT_TRAJ)
        if tm is not None:
            scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False), name="gt_traj")
    if show_pred_traj:
        tm = tube_mesh(pred_aria[:, :3, 3], traj_radius * 0.85, PRED_TRAJ)
        if tm is not None:
            scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False), name="pred_traj")

    if show_frustum:
        fy = float(fov_y)
        far = 0.35
        h_f = np.tan(fy * 0.5) * far
        w_f = h_f
        corners = np.array(
            [[0, 0, 0], [-w_f, -h_f, far], [w_f, -h_f, far], [w_f, h_f, far], [-w_f, h_f, far]],
            dtype=np.float64,
        )
        c2w = gt_aria[frame]
        world = (c2w @ np.c_[corners, np.ones(len(corners))].T).T[:, :3]
        for i, j in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]:
            tm = tube_mesh(world[[i, j]], 0.006, (1.0, 0.78, 0.25), sections=5)
            if tm is not None:
                scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))

    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(42.0), znear=0.05, zfar=80.0)
    scene.add(camera, pose=cam_pose, name="camera")
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    scene.add(light, pose=cam_pose)
    fill_eye = cam_pose[:3, 3] + cam_pose[:3, 0] * 1.5 + cam_pose[:3, 1] * 1.0
    fill_pose = look_at(fill_eye, cam_pose[:3, 3] - cam_pose[:3, 2] * 2.0, cam_pose[:3, 1])
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=0.6), pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    flags = pyrender.RenderFlags.RGBA
    try:
        color, _ = renderer.render(scene, flags=flags | pyrender.RenderFlags.ALL_SOLID)
    except Exception:
        color, _ = renderer.render(scene, flags=flags)
    renderer.delete()
    cam_meta["ghost_frames"] = ghost_ts
    cam_meta["n_pcd_drawn"] = int(xyz.shape[0])
    return color[:, :, :3].astype(np.uint8), cam_meta


def pick_better_side(
    render_kwargs: dict,
) -> tuple[np.ndarray, dict, float]:
    """Render both side signs; keep the one with more non-background body pixels."""
    best_img, best_meta, best_sign, best_score = None, None, 1.0, -1.0
    bg = np.array(render_kwargs["bg"], dtype=np.float32) * 255.0
    for sign in (1.0, -1.0):
        kw = dict(render_kwargs)
        kw["side_sign"] = sign
        img, meta = render_scene(**kw)
        # Score: fraction of saturated green/blue body-ish pixels vs gray bg.
        diff = np.linalg.norm(img.astype(np.float32) - bg[None, None, :], axis=-1)
        score = float((diff > 35).mean())
        # Prefer views where green and blue both appear (GT+pred).
        g = (img[:, :, 1] > img[:, :, 0] + 25) & (img[:, :, 1] > img[:, :, 2] + 10)
        b = (img[:, :, 2] > img[:, :, 0] + 25) & (img[:, :, 2] > img[:, :, 1] + 10)
        score += 0.5 * float(g.mean() + b.mean())
        if score > best_score:
            best_score, best_img, best_meta, best_sign = score, img, meta, sign
    best_meta["side_sign"] = best_sign
    best_meta["side_score"] = best_score
    return best_img, best_meta, best_sign


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key", type=str, default=None)
    p.add_argument("--seq-idx", type=int, default=None)
    p.add_argument("--all", action="store_true", help="Render all keys in --eval-list")
    p.add_argument("--eval-list", type=Path, required=True, help='JSON list of {"pred_key": ...}')
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--mps-pcd-dir", type=Path, default=MPS_PCD_DIR)
    p.add_argument("--vipe-pcd-dir", type=Path, default=VIPE_PCD_DIR)
    p.add_argument(
        "--pcd-source",
        choices=("auto", "vipe", "mps"),
        default="auto",
    )
    p.add_argument(
        "--pcd-modes",
        type=str,
        default="vipe,gt_mps",
        help="Comma list for --all: vipe and/or gt_mps (MPS + ego RGB projection)",
    )
    p.add_argument("--project-rgb", action="store_true")
    p.add_argument("--max-pcd-points", type=int, default=80_000)
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--frame-frac", type=float, default=0.65)
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=1000)
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=40.0)
    p.add_argument("--radius-scale", type=float, default=1.55)
    p.add_argument("--auto-camera", action="store_true", default=True)
    p.add_argument("--manual-orbit", action="store_true", help="Disable auto camera; use --azim/--elev")
    p.add_argument("--side-sign", type=float, default=None, help="Force +1/-1 side; default auto-pick")
    p.add_argument("--traj-radius", type=float, default=0.018)
    p.add_argument("--pcd-sphere-radius", type=float, default=0.010)
    p.add_argument("--pcd-margin", type=float, default=2.0)
    p.add_argument("--ghost-count", type=int, default=4)
    p.add_argument("--ghost-mode", choices=("gt", "pred", "both", "none"), default="gt")
    p.add_argument("--no-pred-traj", action="store_true")
    p.add_argument("--no-gt-traj", action="store_true")
    p.add_argument("--frustum", action="store_true")
    p.add_argument("--bg", type=float, nargs=3, default=[0.96, 0.96, 0.96])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--out-root", type=Path, default=TEASER_DIR / "batch")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def resolve_keys(args) -> list[str]:
    items = json.loads(Path(args.eval_list).read_text())
    if args.all:
        return [sequence_key(x["pred_key"]) for x in items]
    if args.key:
        return [sequence_key(args.key)]
    if args.seq_idx is not None:
        return [sequence_key(items[int(args.seq_idx)]["pred_key"])]
    raise SystemExit("Provide --key, --seq-idx, or --all")


def render_one(
    key: str,
    *,
    preds,
    gt,
    smpl,
    args,
    pcd_mode: str,
    out_path: Path,
) -> dict:
    if args.skip_existing and out_path.is_file():
        print(f"[skip] {out_path}", flush=True)
        return {"key": key, "out": str(out_path), "skipped": True}

    motion = prepare_motion(key, preds, gt, smpl)
    n = motion["n"]
    frame = args.frame if args.frame is not None else int(round(args.frame_frac * (n - 1)))
    frame = int(np.clip(frame, 0, n - 1))

    if pcd_mode == "vipe":
        xyz, rgb, used = load_pcd(
            key,
            source="vipe",
            mps_dir=args.mps_pcd_dir,
            vipe_dir=args.vipe_pcd_dir,
            max_points=args.max_pcd_points,
        )
    elif pcd_mode == "gt_mps":
        xyz, rgb, used = load_pcd(
            key,
            source="mps",
            mps_dir=args.mps_pcd_dir,
            vipe_dir=args.vipe_pcd_dir,
            max_points=args.max_pcd_points,
        )
        take = key.split("___")[0]
        bundle = find_bundle(take)
        cache = TEASER_DIR / "_rgb_proj_cache" / f"{key.replace('/', '_')}.npz"
        if bundle is not None:
            projected = project_ego_rgb_onto_pcd(
                xyz, motion["gt_aria"], bundle, cache_path=cache
            )
            if projected is not None:
                rgb = projected
                used = "mps+ego_rgb"
    else:
        # single-run: honor --pcd-source / --project-rgb
        xyz, rgb, used = load_pcd(
            key,
            source=args.pcd_source,
            mps_dir=args.mps_pcd_dir,
            vipe_dir=args.vipe_pcd_dir,
            max_points=args.max_pcd_points,
        )
        if args.project_rgb and used == "mps":
            take = key.split("___")[0]
            bundle = find_bundle(take)
            cache = TEASER_DIR / "_rgb_proj_cache" / f"{key.replace('/', '_')}.npz"
            if bundle is not None:
                projected = project_ego_rgb_onto_pcd(
                    xyz, motion["gt_aria"], bundle, cache_path=cache
                )
                if projected is not None:
                    rgb = projected
                    used = "mps+ego_rgb"

    body_chunks = [motion["gt_verts"][frame], motion["pred_verts"][frame]]
    for t in ghost_frame_indices(
        frame, 0 if args.ghost_mode == "none" else args.ghost_count
    ):
        body_chunks.append(motion["gt_verts"][t])
    body_pts = np.concatenate(body_chunks, axis=0)
    xyz, rgb = crop_pcd_to_motion(
        xyz,
        rgb,
        motion["gt_aria"][:, :3, 3],
        body_pts,
        margin=args.pcd_margin,
        max_points=min(args.max_pcd_points, 50_000),
        body_clearance=0.30,
    )

    ghost_count = 0 if args.ghost_mode == "none" else args.ghost_count

    take = key.split("___")[0]
    bundle = find_bundle(take)
    fov_y = 1.0
    if bundle is not None:
        tr = np.load(bundle / "trajectory.npz")
        if "rgb_fov_y_rad" in tr.files:
            fov_y = float(tr["rgb_fov_y_rad"][0])

    render_kwargs = dict(
        xyz=xyz,
        rgb=rgb,
        gt_verts=motion["gt_verts"],
        pred_verts=motion["pred_verts"],
        faces=motion["faces"],
        gt_aria=motion["gt_aria"],
        pred_aria=motion["pred_aria"],
        frame=frame,
        width=args.width,
        height=args.height,
        elev=args.elev,
        azim=args.azim,
        radius_scale=args.radius_scale,
        auto_camera=args.auto_camera and not args.manual_orbit,
        side_sign=1.0,
        pcd_sphere_radius=args.pcd_sphere_radius,
        traj_radius=args.traj_radius,
        show_pred_traj=not args.no_pred_traj,
        show_gt_traj=not args.no_gt_traj,
        show_frustum=args.frustum,
        fov_y=fov_y,
        ghost_count=ghost_count,
        ghost_mode=args.ghost_mode,
        bg=tuple(args.bg),
    )

    if render_kwargs["auto_camera"] and args.side_sign is None:
        img, cam_meta, side = pick_better_side(render_kwargs)
    else:
        render_kwargs["side_sign"] = float(args.side_sign if args.side_sign is not None else 1.0)
        img, cam_meta = render_scene(**render_kwargs)
        side = render_kwargs["side_sign"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    meta = {
        "key": key,
        "frame": frame,
        "n_frames": n,
        "pcd_mode": pcd_mode,
        "pcd_source": used,
        "n_points": int(xyz.shape[0]),
        "ghost_count": ghost_count,
        "ghost_mode": args.ghost_mode,
        "side_sign": side,
        "camera": cam_meta,
        "out": str(out_path),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(
        f"Wrote {out_path} ({img.shape[1]}x{img.shape[0]}) "
        f"pcd={used} pts={xyz.shape[0]:,} ghosts={cam_meta.get('ghost_frames')}",
        flush=True,
    )
    return meta


def main():
    args = parse_args()
    keys = resolve_keys(args)
    print(f"Sequences: {len(keys)}", flush=True)


    print(f"Loading preds: {args.preds}", flush=True)
    preds = load_sequences(args.preds)
    print(f"Loading GT: {args.gt}", flush=True)
    gt = load_sequences(args.gt)
    print("Loading SMPL-X...", flush=True)
    smpl = load_smplx().to(torch.device(args.device))

    if args.all:
        modes = [m.strip() for m in args.pcd_modes.split(",") if m.strip()]
        summary = []
        for i, key in enumerate(keys):
            if key not in preds or key not in gt:
                print(f"[warn] missing key in preds/gt: {key}", flush=True)
                continue
            ep = key.split("_vgoal_seq_")[0].split("_mvgoal_seq_")[0]
            short = ep  # readable folder name
            for mode in modes:
                out = args.out_root / mode / f"{short}_f{args.frame if args.frame is not None else 'auto'}.png"
                # Fix filename once we know frame inside render_one - use key stem.
                out = args.out_root / mode / f"{key.replace('/', '_')}.png"
                try:
                    if mode == "vipe":
                        stem = key.replace("/", "_")
                        if not (args.vipe_pcd_dir / f"{stem}.npz").is_file():
                            print(f"[skip-vipe] no cache for {short}", flush=True)
                            summary.append({"key": key, "mode": mode, "status": "no_vipe_pcd"})
                            continue
                    meta = render_one(
                        key, preds=preds, gt=gt, smpl=smpl, args=args, pcd_mode=mode, out_path=out
                    )
                    summary.append({"key": key, "mode": mode, "status": "ok", **{k: meta.get(k) for k in ("out", "frame", "pcd_source")}})
                except Exception as exc:
                    print(f"[error] {key} mode={mode}: {exc}", flush=True)
                    traceback.print_exc()
                    summary.append({"key": key, "mode": mode, "status": "error", "error": str(exc)})
            print(f"=== progress {i+1}/{len(keys)} ===", flush=True)
        summary_path = args.out_root / "summary.json"
        args.out_root.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        n_ok = sum(1 for s in summary if s.get("status") == "ok")
        print(f"Done. {n_ok}/{len(summary)} renders OK -> {summary_path}", flush=True)
        return

    # Single render
    key = keys[0]
    if key not in preds or key not in gt:
        raise SystemExit(f"Key missing from preds/gt: {key}")
    mode = "single"
    if args.pcd_source == "vipe":
        mode = "vipe"
    elif args.pcd_source == "mps" or args.project_rgb:
        mode = "gt_mps" if args.project_rgb else "single"
    out = args.out
    if out is None:
        TEASER_DIR.mkdir(parents=True, exist_ok=True)
        out = TEASER_DIR / f"{key.replace('/', '_')}.png"
    render_one(key, preds=preds, gt=gt, smpl=smpl, args=args, pcd_mode=mode, out_path=out)


if __name__ == "__main__":
    main()
