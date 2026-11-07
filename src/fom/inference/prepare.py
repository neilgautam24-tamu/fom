#!/usr/bin/env python3
"""Step 1 of video inference: ViPE outputs -> square-cropped RGB / depth (+ floor levelling).

Run inside the ViPE environment (needs ``vipe.utils.io``) after ``vipe infer`` on the clip:

  * centre square crop for any resolution / orientation, intrinsics shifted to match;
  * RGB and depth stored losslessly at 512 x 512, the resolution the training frames had;
  * optionally (``--level``, recommended for real footage) a levelling rotation fitted to the floor
    in ViPE's metric depth. ViPE's world frame is the first camera's frame, so a camera pitched
    down on frame 0 tilts the whole trajectory (walking forward would read as walking downhill).

Usage:
    python -m fom.inference.prepare <vipe_results_dir> <clip_id> <out_dir> [--level]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from vipe.utils.io import read_depth_artifacts

FEAT_SIZE = 512  # == fom.inference.conditions.FEAT_SIZE (not imported: this module runs in the ViPE env)


def centre_square(h: int, w: int) -> tuple[int, int, int]:
    s = min(h, w)
    return s, (w - s) // 2, (h - s) // 2


def rotation_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b (Rodrigues)."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def estimate_levelling(depths, c2w, K, frame_step=10, px_step=8, max_depth=8.0, iters=500, thresh=0.02, seed=0):
    """Fit the floor plane in ViPE world (OpenCV, +Y ~ down) and return R with R @ n_floor_down = +Y."""
    pts = []
    for i in range(0, len(depths), frame_step):
        d = depths[i][::px_step, ::px_step]
        v, u = np.mgrid[0 : depths[i].shape[0] : px_step, 0 : depths[i].shape[1] : px_step].astype(np.float32)
        ok = np.isfinite(d) & (d > 0) & (d < max_depth)
        fx, fy, cx, cy = K[i]
        p = np.stack([(u[ok] - cx) / fx * d[ok], (v[ok] - cy) / fy * d[ok], d[ok]], 1)
        pts.append(p @ c2w[i, :3, :3].T + c2w[i, :3, 3])
    pts = np.concatenate(pts).astype(np.float64)
    cams = c2w[:, :3, 3]
    cand = pts[pts[:, 1] > np.median(cams[:, 1]) + 0.8]  # well below the cameras

    rng = np.random.default_rng(seed)
    best_n, best_inl = None, None
    for _ in range(iters):
        s = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((cand - s[0]) @ n) < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_n, best_inl = n, inl
    P = cand[best_inl]
    n = np.linalg.svd(P - P.mean(0))[2][-1]
    n = n if n[1] > 0 else -n  # point down (+Y in OpenCV world)
    tilt = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
    heights = (P.mean(0) - cams) @ n
    info = {
        "floor_normal_down_vipe": n.tolist(),
        "tilt_deg": tilt,
        "floor_inliers": int(best_inl.sum()),
        "floor_candidates": int(len(cand)),
        "camera_height_median_m": float(np.median(heights)),
    }
    if tilt > 30.0:  # implausible for a handheld walk; don't trust it
        print(f"[warn] floor tilt {tilt:.1f} deg > 30 deg, skipping levelling", flush=True)
        return np.eye(3), {**info, "applied": False}
    return rotation_a_to_b(n, np.array([0.0, 1.0, 0.0])), {**info, "applied": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vipe_out", type=Path)
    ap.add_argument("clip_id")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--level", action="store_true", help="Fit the floor plane and level the trajectory")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    rgb_path = args.vipe_out / "rgb" / f"{args.clip_id}.mp4"
    cap = cv2.VideoCapture(str(rgb_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.stack(frames)
    T, H, W = frames.shape[:3]
    s, x0, y0 = centre_square(H, W)
    print(f"{args.clip_id}: {T} frames {W}x{H} @ {fps:.2f} fps -> square {s} at ({x0},{y0})", flush=True)

    rgb_512 = np.stack([cv2.resize(f[y0 : y0 + s, x0 : x0 + s], (FEAT_SIZE, FEAT_SIZE), interpolation=cv2.INTER_AREA) for f in frames])
    np.save(out / "rgb_512.npy", rgb_512)
    iio.imwrite(out / "frame000_rgb.png", rgb_512[0])

    depths = [d.numpy().astype(np.float32) for _, d in read_depth_artifacts(args.vipe_out / "depth" / f"{args.clip_id}.zip")]
    assert len(depths) == T, f"depth {len(depths)} vs rgb {T}"
    depth_512 = np.stack([cv2.resize(d[y0 : y0 + s, x0 : x0 + s], (FEAT_SIZE, FEAT_SIZE), interpolation=cv2.INTER_AREA) for d in depths])
    np.save(out / "depth_512.npy", depth_512)

    intr = np.load(args.vipe_out / "intrinsics" / f"{args.clip_id}.npz")
    K = intr["data"].astype(np.float32)
    K_sq = K.copy()
    K_sq[:, 2] -= x0
    K_sq[:, 3] -= y0
    np.savez(out / "intrinsics_square.npz", data=K_sq, inds=intr["inds"])

    pose = np.load(args.vipe_out / "pose" / f"{args.clip_id}.npz")
    c2w = pose["data"].astype(np.float64)
    np.savez(out / "pose.npz", data=pose["data"], inds=pose["inds"])

    if args.level:
        R_level, level_info = estimate_levelling(depths, c2w, K)
        print(f"levelling: tilt {level_info['tilt_deg']:.1f} deg, camera height {level_info['camera_height_median_m']:.2f} m, "
              f"floor inliers {level_info['floor_inliers']}/{level_info['floor_candidates']}", flush=True)
    else:
        R_level, level_info = None, {"applied": False}

    meta = {
        "clip_id": args.clip_id,
        "vipe_out": str(args.vipe_out),
        "src_fps": fps,
        "n_frames": int(T),
        "src_wh": [int(W), int(H)],
        "crop_size": int(s),
        "crop_xy": [int(x0), int(y0)],
        "feat_size": FEAT_SIZE,
        "R_level_vipe": R_level.tolist() if R_level is not None else None,
        "levelling": level_info,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "DONE").write_text("ok\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
