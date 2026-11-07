#!/usr/bin/env python3
"""High-quality MPS + ego-RGB point cloud for teaser / composer viz.

VIPE fused clouds look dense and colorful because they are RGB-D fused and
tightly cropped. Raw MPS obs caches are sparse, height-colored, and often
include distant SLAM outliers (~20m span) that destroy geometry in the view.

This module:
  1. Clips points to a corridor around the Aria trajectory
  2. Drops statistical floaters
  3. Projects ego RGB with all window frames (best rotation)
  4. Boosts saturation so walls/floors read clearly
  5. Fills unprojected points with height color (never flat gray)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fom.visualization.pointcloud.mps import find_bundle


def clip_to_traj_corridor(
    xyz: np.ndarray,
    rgb: np.ndarray,
    traj_xyz: np.ndarray,
    *,
    margin: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] == 0:
        return xyz, rgb
    traj = np.asarray(traj_xyz, np.float32)
    if len(traj) > 64:
        traj = traj[np.linspace(0, len(traj) - 1, 64).round().astype(int)]
    d2 = ((xyz[:, None, :] - traj[None, :, :]) ** 2).sum(-1).min(axis=1)
    keep = d2 <= (margin * margin)
    z0, z1 = float(traj[:, 2].min()) - 0.8, float(traj[:, 2].max()) + 2.2
    keep &= (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1)
    if int(keep.sum()) < 1500:
        keep = d2 <= ((margin + 1.5) ** 2)
    return xyz[keep].astype(np.float32), rgb[keep].astype(np.uint8)


def remove_statistical_outliers(
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 3.0,
    max_check: int = 40_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop sparse floaters (approx Open3D statistical outlier removal)."""
    n = xyz.shape[0]
    if n < nb_neighbors * 4:
        return xyz, rgb
    pts = xyz.astype(np.float32)
    # Subsample for neighbor search if huge, then map back via distance threshold.
    rng = np.random.default_rng(0)
    if n > max_check:
        # Compute on all via chunked brute force to random landmarks.
        landmarks = pts[rng.choice(n, size=min(3000, n), replace=False)]
        # Mean distance to 8 nearest landmarks as density proxy.
        d = np.sqrt(((pts[:, None, :] - landmarks[None, :, :]) ** 2).sum(-1))
        d.sort(axis=1)
        mean_d = d[:, :8].mean(axis=1)
    else:
        # Exact-ish: distance to k random other points as proxy (fast).
        idx = rng.integers(0, n, size=(n, nb_neighbors))
        # Avoid self roughly.
        samp = pts[idx]
        mean_d = np.linalg.norm(samp - pts[:, None, :], axis=2).mean(axis=1)
    mu, sigma = float(mean_d.mean()), float(mean_d.std() + 1e-9)
    keep = mean_d <= (mu + std_ratio * sigma)
    if int(keep.sum()) < max(500, n // 5):
        return xyz, rgb
    return xyz[keep], rgb[keep]


def boost_saturation(rgb: np.ndarray, sat: float = 1.55, value: float = 1.12) -> np.ndarray:
    """Simple HSV-ish saturation/value boost in RGB space."""
    x = rgb.astype(np.float32) / 255.0
    mx = x.max(axis=1, keepdims=True)
    mn = x.min(axis=1, keepdims=True)
    mid = (mx + mn) * 0.5
    chroma = mx - mn
    # Push away from gray toward dominant channel.
    boosted = mid + (x - mid) * sat
    boosted = mid + (boosted - mid)  # noqa: keep structure
    boosted = x.mean(axis=1, keepdims=True) + (x - x.mean(axis=1, keepdims=True)) * sat
    boosted = np.clip(boosted * value, 0.0, 1.0)
    # Preserve very dark points.
    dark = mx[:, 0] < 0.05
    out = (boosted * 255.0).astype(np.uint8)
    out[dark] = rgb[dark]
    _ = chroma  # silence lint
    return out


def project_ego_rgb_hq(
    xyz: np.ndarray,
    aria_traj_T: np.ndarray,
    bundle: Path,
    *,
    max_frames: int = 96,
    rot_k: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ego RGB -> (rgb_uint8, hit_mask).

    ``rot_k=2`` empirically maximizes chroma for the CW90-rotated Nymeria bundles
    when paired with canonical-frame Aria poses.
    """
    rgb_path = bundle / "ego_rgb.npz"
    if not rgb_path.is_file():
        return np.zeros((len(xyz), 3), np.uint8), np.zeros(len(xyz), bool)

    tr = np.load(bundle / "trajectory.npz")
    fov_y = float(tr["rgb_fov_y_rad"][0]) if "rgb_fov_y_rad" in tr.files else 1.0
    z = np.load(rgb_path, mmap_mode="r")
    frames = z["rgb"]
    n_src = int(frames.shape[0])
    n_traj = int(aria_traj_T.shape[0])
    n_use = min(max_frames, n_traj, n_src)
    if n_use <= 0:
        return np.zeros((len(xyz), 3), np.uint8), np.zeros(len(xyz), bool)

    traj_idx = np.linspace(0, n_traj - 1, n_use).round().astype(int)
    src_idx = np.linspace(0, n_src - 1, n_use).round().astype(int)

    n_pts = xyz.shape[0]
    best_depth = np.full(n_pts, np.inf, dtype=np.float32)
    out = np.zeros((n_pts, 3), dtype=np.uint8)
    # Accumulate for mild temporal median on multi-hit points.
    acc = np.zeros((n_pts, 3), dtype=np.float32)
    counts = np.zeros(n_pts, dtype=np.int32)

    for ti, si in zip(traj_idx, src_idx):
        fr = np.asarray(frames[int(si)], dtype=np.uint8)
        upright = np.ascontiguousarray(np.rot90(fr, k=int(rot_k)))
        h, w = upright.shape[:2]
        fy = 0.5 * h / max(np.tan(0.5 * fov_y), 1e-6)
        fx = fy
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        c2w = np.asarray(aria_traj_T[int(ti)], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        Xc = (w2c @ np.c_[xyz.astype(np.float64), np.ones(n_pts)].T).T[:, :3]
        zc = Xc[:, 2]
        valid = zc > 0.08
        u = fx * (Xc[:, 0] / np.clip(zc, 1e-6, None)) + cx
        v = fy * (Xc[:, 1] / np.clip(zc, 1e-6, None)) + cy
        ui = np.nan_to_num(u, nan=-1.0)
        vi = np.nan_to_num(v, nan=-1.0)
        ui = ui.round().astype(np.int32)
        vi = vi.round().astype(np.int32)
        valid &= (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        # Soft edge margin - avoid distorted border pixels.
        valid &= (ui >= 2) & (ui < w - 2) & (vi >= 2) & (vi < h - 2)
        if not np.any(valid):
            continue
        idxs = np.where(valid)[0]
        cols = upright[vi[idxs], ui[idxs]].astype(np.float32)
        acc[idxs] += cols
        counts[idxs] += 1
        closer = valid & (zc.astype(np.float32) < best_depth)
        if np.any(closer):
            cidx = np.where(closer)[0]
            out[cidx] = upright[vi[cidx], ui[cidx]]
            best_depth[cidx] = zc[cidx].astype(np.float32)

    hit = counts > 0
    # Prefer averaged color when a point was seen in several frames.
    multi = counts >= 2
    if np.any(multi):
        out[multi] = (acc[multi] / counts[multi, None]).astype(np.uint8)
    return out, hit


def radius_outlier_filter(
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    radius: float = 0.08,
    min_neighbors: int = 6,
    max_ref: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep points that have enough neighbors in a local ball (removes dust)."""
    n = xyz.shape[0]
    if n < 500:
        return xyz, rgb
    pts = xyz.astype(np.float32)
    rng = np.random.default_rng(0)
    ref = pts if n <= max_ref else pts[rng.choice(n, max_ref, replace=False)]
    keep = np.zeros(n, dtype=bool)
    bs = 4000
    r2 = radius * radius
    for i0 in range(0, n, bs):
        chunk = pts[i0 : i0 + bs]
        d2 = ((chunk[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        cnt = (d2 <= r2).sum(axis=1)
        keep[i0 : i0 + bs] = cnt >= min_neighbors
    if int(keep.sum()) < max(800, n // 10):
        return xyz, rgb
    return xyz[keep], rgb[keep]


def densify_edges(
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    link_dist: float = 0.06,
    max_links_per_point: int = 2,
    fill_per_link: int = 1,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill small local gaps so already-traced edges/boundaries read as solid.

    MPS semidense points concentrate on high-gradient scene structure
    (corners, frames, edges) rather than flat interiors - the surviving
    point set already *outlines* the room, it's just dotted rather than
    solid along those outlines. Linking each point to its ``k`` nearest
    neighbors within ``link_dist`` and inserting interpolated points along
    each link thickens exactly that traced structure. It deliberately does
    *not* bridge the larger gaps between unrelated structures (a real
    doorway, an unobserved wall) - those neighbors are simply farther than
    ``link_dist`` and never linked. Fully vectorized (one KDTree query, no
    Python-level point loop) so it stays cheap even at tens of thousands of
    points; output growth is capped at ``max_links_per_point *
    fill_per_link`` per point.

    If ``mask`` is set, only links with at least one endpoint in the mask
    are filled (used for floor / ceiling / geometric-edge targeting).
    """
    n = xyz.shape[0]
    if n < 8 or fill_per_link <= 0 or max_links_per_point <= 0:
        return xyz, rgb
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    k = min(max_links_per_point + 1, n)
    dists, idxs = tree.query(xyz, k=k)
    if k == 1:
        return xyz, rgb
    i_idx = np.repeat(np.arange(n), k - 1)
    j_idx = idxs[:, 1:].reshape(-1)
    d = dists[:, 1:].reshape(-1)
    valid = (d > 1e-6) & (d <= link_dist)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape[0] != n:
            raise ValueError("mask length must match xyz")
        valid &= m[i_idx] | m[j_idx]
    i_idx, j_idx = i_idx[valid], j_idx[valid]
    if i_idx.size == 0:
        return xyz, rgb
    lo, hi = np.minimum(i_idx, j_idx), np.maximum(i_idx, j_idx)
    edges = np.unique(np.stack([lo, hi], axis=1), axis=0)

    ts = np.linspace(0.0, 1.0, fill_per_link + 2, dtype=np.float32)[1:-1]
    p0, p1 = xyz[edges[:, 0]], xyz[edges[:, 1]]
    c0 = rgb[edges[:, 0]].astype(np.float32)
    c1 = rgb[edges[:, 1]].astype(np.float32)
    new_xyz = p0[:, None, :] * (1 - ts)[None, :, None] + p1[:, None, :] * ts[None, :, None]
    new_rgb = c0[:, None, :] * (1 - ts)[None, :, None] + c1[:, None, :] * ts[None, :, None]
    new_xyz = new_xyz.reshape(-1, 3).astype(np.float32)
    new_rgb = np.clip(new_rgb.reshape(-1, 3), 0, 255).astype(np.uint8)
    return np.concatenate([xyz, new_xyz], axis=0), np.concatenate([rgb, new_rgb], axis=0)


def _strength_params(strength: float) -> tuple[float, int, int]:
    """Map UI strength 0..3 -> (link_dist, max_links, fill_per_link)."""
    s = float(np.clip(strength, 0.0, 3.0))
    if s <= 1e-6:
        return 0.0, 0, 0
    # Mild -> aggressive: longer links + more interpolated fills.
    link = 0.045 + 0.035 * s
    links = int(round(1 + s))
    fills = int(round(max(1, s)))
    return float(link), int(links), int(fills)


def geometric_edge_mask(
    xyz: np.ndarray,
    *,
    k: int = 12,
    keep_frac: float = 0.35,
) -> np.ndarray:
    """Mark points that look like edges/corners (non-planar local neighborhoods)."""
    n = xyz.shape[0]
    if n < k + 2:
        return np.ones(n, dtype=bool)
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    _, idxs = tree.query(xyz, k=min(k + 1, n))
    neigh = xyz[idxs[:, 1:]]  # (N, k, 3)
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    # Covariance eigenvalues via batched SVD on small kx3 - planarity score.
    # sigma2/sigma0 small => planar; larger => edge/corner.
    try:
        # (N, 3, 3)
        cov = np.matmul(centered.transpose(0, 2, 1), centered) / max(neigh.shape[1] - 1, 1)
        evals = np.linalg.eigvalsh(cov)  # ascending
        e0 = np.clip(evals[:, 0], 1e-12, None)
        e2 = np.clip(evals[:, 2], 1e-12, None)
        score = e0 / e2  # high on edges/corners, low on flats
    except Exception:
        return np.ones(n, dtype=bool)
    thr = float(np.quantile(score, 1.0 - float(np.clip(keep_frac, 0.05, 0.9))))
    return score >= thr


def height_band_masks(
    xyz: np.ndarray,
    *,
    floor_frac: float = 0.18,
    top_frac: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for floor (low Z) and top/ceiling (high Z) bands."""
    z = xyz[:, 2]
    z0, z1 = float(np.percentile(z, 2)), float(np.percentile(z, 98))
    span = max(z1 - z0, 1e-3)
    floor_hi = z0 + float(np.clip(floor_frac, 0.05, 0.45)) * span
    top_lo = z1 - float(np.clip(top_frac, 0.05, 0.45)) * span
    return z <= floor_hi, z >= top_lo


def apply_pcd_density(
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    overall: float = 1.0,
    edge: float = 1.0,
    floor: float = 0.0,
    top: float = 0.0,
    floor_frac: float = 0.18,
    top_frac: float = 0.18,
    max_points: int = 350_000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Interactive densification on top of a base MPS-RGB cloud.

    Parameters are UI strengths in ``[0, 3]``:
      - ``overall``: global gap-fill along all local links
      - ``edge``: extra fill on geometric edges/corners only
      - ``floor`` / ``top``: extra fill in low / high Z bands
    """
    meta = {
        "n_in": int(xyz.shape[0]),
        "overall": float(overall),
        "edge": float(edge),
        "floor": float(floor),
        "top": float(top),
    }
    if xyz.shape[0] < 8:
        meta["n_out"] = int(xyz.shape[0])
        return xyz, rgb, meta

    out_xyz, out_rgb = xyz, rgb

    link, n_links, n_fill = _strength_params(overall)
    if n_fill > 0:
        out_xyz, out_rgb = densify_edges(
            out_xyz,
            out_rgb,
            link_dist=link,
            max_links_per_point=n_links,
            fill_per_link=n_fill,
        )

    link, n_links, n_fill = _strength_params(edge)
    if n_fill > 0:
        # Edge mask on the *current* cloud so newly filled points can
        # participate in a second pass if overall already ran.
        emask = geometric_edge_mask(out_xyz, keep_frac=0.40)
        out_xyz, out_rgb = densify_edges(
            out_xyz,
            out_rgb,
            link_dist=link * 0.85,
            max_links_per_point=n_links + 1,
            fill_per_link=n_fill + 1,
            mask=emask,
        )

    floor_m, top_m = height_band_masks(
        out_xyz, floor_frac=floor_frac, top_frac=top_frac
    )
    link, n_links, n_fill = _strength_params(floor)
    if n_fill > 0 and floor_m.any():
        out_xyz, out_rgb = densify_edges(
            out_xyz,
            out_rgb,
            link_dist=link * 1.15,
            max_links_per_point=n_links + 1,
            fill_per_link=n_fill + 1,
            mask=floor_m,
        )
        # Refresh bands after growth.
        floor_m, top_m = height_band_masks(
            out_xyz, floor_frac=floor_frac, top_frac=top_frac
        )

    link, n_links, n_fill = _strength_params(top)
    if n_fill > 0 and top_m.any():
        out_xyz, out_rgb = densify_edges(
            out_xyz,
            out_rgb,
            link_dist=link * 1.15,
            max_links_per_point=n_links + 1,
            fill_per_link=n_fill + 1,
            mask=top_m,
        )

    if out_xyz.shape[0] > max_points:
        rng = np.random.default_rng(1)
        # Prefer keeping original points, subsample the extras.
        n0 = xyz.shape[0]
        if n0 >= max_points:
            sel = rng.choice(out_xyz.shape[0], max_points, replace=False)
            out_xyz, out_rgb = out_xyz[sel], out_rgb[sel]
        else:
            extra = np.arange(n0, out_xyz.shape[0])
            keep_extra = rng.choice(
                extra, max_points - n0, replace=False
            )
            sel = np.concatenate([np.arange(n0), keep_extra])
            out_xyz, out_rgb = out_xyz[sel], out_rgb[sel]

    meta["n_out"] = int(out_xyz.shape[0])
    return out_xyz.astype(np.float32), out_rgb.astype(np.uint8), meta


def build_mps_rgb_viz(
    xyz: np.ndarray,
    rgb_height: np.ndarray,
    aria_traj_T: np.ndarray,
    bundle: Path | None,
    *,
    margin: float = 3.5,
    sat_boost: float = 1.55,
    value_boost: float = 1.12,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return cleaned (xyz, rgb) ready for interactive / paper viz."""
    meta: dict = {"n_in": int(xyz.shape[0])}
    xyz_c, rgb_h = clip_to_traj_corridor(
        xyz, rgb_height, aria_traj_T[:, :3, 3], margin=margin
    )
    meta["n_clip"] = int(xyz_c.shape[0])
    xyz_c, rgb_h = remove_statistical_outliers(xyz_c, rgb_h, std_ratio=2.0)
    xyz_c, rgb_h = radius_outlier_filter(xyz_c, rgb_h, radius=0.07, min_neighbors=5)
    meta["n_inlier"] = int(xyz_c.shape[0])

    if bundle is None or xyz_c.shape[0] == 0:
        out = boost_saturation(rgb_h, sat=1.2, value=1.05)
        xyz_c, out = densify_edges(xyz_c, out)
        meta["n_out"] = int(xyz_c.shape[0])
        return xyz_c, out, meta

    proj, hit = project_ego_rgb_hq(xyz_c, aria_traj_T, bundle)
    meta["hit_frac"] = float(hit.mean()) if len(hit) else 0.0
    # Always keep every inlier point - RGB-textured where the ego camera
    # actually saw it, height-color fallback elsewhere. Previously, points
    # without an RGB hit were dropped outright whenever overall coverage
    # cleared a threshold; that silently thinned exactly the boundary/edge
    # geometry a head-mounted camera under-samples (ceilings, peripheral
    # walls), which is the opposite of what a "show clear geometry" viz
    # wants - a spottily-textured point is still real 3D structure.
    out = rgb_h.copy()
    if hit.any():
        out[hit] = boost_saturation(proj[hit], sat=sat_boost, value=value_boost)
    if (~hit).any():
        out[~hit] = boost_saturation(out[~hit], sat=1.2, value=1.05)

    xyz_c, out = densify_edges(xyz_c, out)
    meta["n_out"] = int(xyz_c.shape[0])
    return xyz_c, out, meta


def load_dense_mps_xyz(
    key: str,
    aria_traj_T: np.ndarray,
    *,
    dense_dir: Path,
    voxel: float = 0.008,
    max_points: int = 200_000,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Rebuild a denser obs-filtered MPS cloud (finer voxel) for viz.

    Caching is left to ``build_sequence_mps_obs_pcd`` so its cache-version check applies.
    """
    from fom.visualization.pointcloud.mps import build_sequence_mps_obs_pcd

    dense_dir.mkdir(parents=True, exist_ok=True)
    return build_sequence_mps_obs_pcd(
        key,
        aria_traj_T,
        cache_dir=dense_dir,
        voxel=voxel,
        max_points=max_points,
        force=False,
        obs_uid_cache_dir=dense_dir / "_obs_uid_cache",
    )


def load_or_build_mps_rgb(
    key: str,
    xyz: np.ndarray,
    rgb_height: np.ndarray,
    aria_traj_T: np.ndarray,
    *,
    cache_dir: Path,
    force: bool = False,
    prefer_dense_rebuild: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Bump the version suffix whenever build_mps_rgb_viz changes, so stale caches are rebuilt.
    cache = cache_dir / f"{key.replace('/', '_')}_mpsrgb_v6.npz"
    if cache.is_file() and not force:
        z = np.load(cache)
        return (
            np.asarray(z["xyz"], np.float32),
            np.asarray(z["rgb"], np.uint8),
            "mps_rgb_v6(cache)",
        )
    src_xyz, src_rgb = xyz, rgb_height
    if prefer_dense_rebuild:
        try:
            dense = load_dense_mps_xyz(
                key,
                aria_traj_T,
                dense_dir=cache_dir / "_dense_mps",
                voxel=0.008,
                max_points=200_000,
            )
            if dense is not None and dense[0].shape[0] > src_xyz.shape[0]:
                src_xyz, src_rgb = dense
                print(
                    f"[mps-rgb] using denser MPS rebuild n={src_xyz.shape[0]:,}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[mps-rgb] dense rebuild skipped: {exc}", flush=True)

    take = key.split("___")[0]
    bundle = find_bundle(take)
    xyz_o, rgb_o, meta = build_mps_rgb_viz(src_xyz, src_rgb, aria_traj_T, bundle)
    np.savez_compressed(
        cache, xyz=xyz_o, rgb=rgb_o, **{k: np.asarray(v) for k, v in meta.items()}
    )
    print(
        f"[mps-rgb] {key[:50]}... in={meta['n_in']:,} -> out={meta.get('n_out', 0):,} "
        f"hit={meta.get('hit_frac', 0):.0%}",
        flush=True,
    )
    return xyz_o, rgb_o, "mps_rgb_v6"
