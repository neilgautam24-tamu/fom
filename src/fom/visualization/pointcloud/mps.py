#!/usr/bin/env python3
"""Filter Nymeria Plus MPS semidense points to those observed during a sequence window.

Points come from ``recording_head/mps/slam/semidense_points.csv.gz``.
Visibility is defined by ``semidense_observations.csv.gz`` rows whose
``frame_tracking_timestamp_us`` falls in the extracted sequence's device-time
range. Filtered XYZ are rigidly mapped into the canonical clip frame used by
predictions (Umeyama on Aria camera translations vs the GT camera poses ``camera_T``).

Correspondence for that Umeyama fit must replicate exactly how the GT camera
poses were downsampled from the bundle's raw per-frame trajectory
(:mod:`fom.preprocessing.packing.pack`: fixed integer stride to 15fps, then
the first ``window`` samples only - *not* a proportional resample across the
whole bundle). See ``_infer_frame_stride`` / ``world_to_can_from_traj``.

Raw MPS data is read from ``$FOM_NYMERIA_PLUS``, bundles from ``$FOM_BUNDLE_ROOTS``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fom.visualization.common import find_bundle as _find_bundle
from fom.visualization.common import nymeria_plus_root, voxel_downsample

# Bump when build_sequence_mps_obs_pcd's output changes so stale cached .npz files are rebuilt.
CACHE_VERSION = 2


def find_bundle(take_or_key: str) -> Path | None:
    """Bundle directory holding ``trajectory.npz`` for a sequence name."""
    return _find_bundle(take_or_key, marker_file="trajectory.npz")


def parse_pred_key(key: str) -> tuple[str, str, str]:
    """Return (episode, bundle_name, take_without_window)."""
    take = key.split("___")[0]
    for marker in ("_vgoal_seq_", "_mvgoal_seq_"):
        if marker in take:
            ep, rest = take.split(marker, 1)
            bundle = marker[1:] + rest
            return ep, bundle, take
    raise ValueError(f"Cannot parse episode/bundle from key: {key}")




def plus_recording_head(ep: str) -> Path | None:
    root = nymeria_plus_root() / ep / "recording_head"
    slam = root / "mps" / "slam"
    pts = slam / "semidense_points.csv.gz"
    obs = slam / "semidense_observations.csv.gz"
    if pts.is_file() and obs.is_file():
        return root
    return None


def mps_paths(ep: str) -> dict[str, Path] | None:
    head = plus_recording_head(ep)
    if head is None:
        return None
    slam = head / "mps" / "slam"
    return {
        "recording_head": head,
        "semidense_points": slam / "semidense_points.csv.gz",
        "semidense_observations": slam / "semidense_observations.csv.gz",
    }


def bundle_device_time_range_us(bundle: Path) -> tuple[int, int]:
    tr = np.load(bundle / "trajectory.npz")
    ts = np.asarray(tr["device_timestamp_us"], dtype=np.int64)
    return int(ts[0]), int(ts[-1])


def bundle_device_timestamps_us(bundle: Path) -> np.ndarray:
    tr = np.load(bundle / "trajectory.npz")
    return np.asarray(tr["device_timestamp_us"], dtype=np.int64)


def _infer_frame_stride(device_timestamp_us: np.ndarray, target_fps: float) -> int:
    """Integer decimation stride matching fom.preprocessing.packing.pack.

    That script infers the bundle's native capture fps from median frame
    spacing, then strides down to ``target_fps`` with
    ``smpl_step = round(source_fps / target_fps)`` - a fixed integer stride,
    not a proportional resample. Reproduce it here so trajectory indices line
    up frame-for-frame with GT ``aria_traj_T``.
    """
    if device_timestamp_us.size < 2:
        return 1
    dt = np.diff(device_timestamp_us.astype(np.float64))
    dt = dt[dt > 0]
    if dt.size == 0:
        return 1
    source_fps = 1.0e6 / float(np.median(dt))
    return max(1, int(round(source_fps / max(target_fps, 1e-6))))


def bundle_camera_c2w(bundle: Path) -> np.ndarray:
    tr = np.load(bundle / "trajectory.npz")
    R = np.asarray(tr["camera_rotation"], dtype=np.float64)
    t = np.asarray(tr["camera_position"], dtype=np.float64)
    T = np.zeros((len(t), 4, 4), dtype=np.float64)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    return T


def umeyama_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R (3x3), t (3,) mapping src -> dst (no scale)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    U, _, Vt = np.linalg.svd(Y.T @ X)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1] *= -1
        R = U @ Vt
    t = mu_d - R @ mu_s
    return R.astype(np.float64), t.astype(np.float64)


def world_to_can_from_traj(
    world_c2w: np.ndarray,
    can_aria: np.ndarray,
    *,
    device_timestamp_us: np.ndarray | None = None,
    target_fps: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid map Aria-world -> canonical clip frame using camera translation tracks.

    GT ``can_aria`` (``aria_traj_T``) is a fixed-stride downsample of the
    *first* ``len(can_aria)`` decimated samples of ``world_c2w``, not a
    proportional resample across the bundle's full length (``one_sample_per_
    seq=True`` always starts the GT window at raw frame 0 - see module
    docstring). When ``device_timestamp_us`` is available, replicate that
    exact stride so Umeyama fits truly-corresponding points; otherwise fall
    back to a proportional resample (only correct if
    ``can_aria`` really does span the whole clip).
    """
    n_dst = len(can_aria)
    if device_timestamp_us is not None and len(world_c2w) > 0 and n_dst > 0:
        stride = _infer_frame_stride(device_timestamp_us, target_fps)
        idx = np.arange(0, len(world_c2w), stride)[:n_dst]
        if len(idx) >= 3:
            src = world_c2w[idx, :3, 3]
            dst = np.asarray(can_aria[: len(idx), :3, 3], dtype=np.float64)
            return umeyama_rigid(src, dst)
        print(
            f"[mps-obs] stride-matched correspondence too short "
            f"({len(idx)} pts) - falling back to proportional resample",
            flush=True,
        )

    n = min(len(world_c2w), n_dst)
    if len(world_c2w) != n_dst:
        idx = np.linspace(0, len(world_c2w) - 1, n).round().astype(int)
        src = world_c2w[idx, :3, 3]
    else:
        src = world_c2w[:n, :3, 3]
    dst = np.asarray(can_aria[:n, :3, 3], dtype=np.float64)
    return umeyama_rigid(src, dst)


def clip_points_near_traj(
    xyz: np.ndarray,
    rgb: np.ndarray,
    traj_xyz: np.ndarray,
    *,
    max_dist: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop semidense outliers far from the sequence camera path."""
    if xyz.shape[0] == 0 or traj_xyz.shape[0] == 0:
        return xyz, rgb
    # Chunked min-distance to avoid huge NxT memory.
    traj = np.asarray(traj_xyz, dtype=np.float32)
    dmin = np.full(xyz.shape[0], np.inf, dtype=np.float32)
    for i in range(0, len(traj), 32):
        chunk = traj[i : i + 32]
        # (N, C)
        d = np.linalg.norm(xyz[:, None, :] - chunk[None, :, :], axis=2).min(axis=1)
        dmin = np.minimum(dmin, d)
    keep = dmin <= float(max_dist)
    return xyz[keep], rgb[keep]


def transform_points(xyz: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (xyz.astype(np.float64) @ R.T + t[None]).astype(np.float32)


def colors_by_height(xyz: np.ndarray) -> np.ndarray:
    z = xyz[:, 2]
    lo, hi = np.percentile(z, [5, 95]) if z.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    # simple teal->sand ramp
    c0 = np.array([40, 90, 110], dtype=np.float32)
    c1 = np.array([220, 200, 150], dtype=np.float32)
    return (c0[None] * (1.0 - t[:, None]) + c1[None] * t[:, None]).astype(np.uint8)


def _obs_uid_cache_path(
    cache_dir: Path, observations_csv: Path, t0: int, t1: int
) -> Path:
    tag = hashlib.md5(
        f"{observations_csv.resolve()}|{t0}|{t1}".encode()
    ).hexdigest()[:12]
    return cache_dir / f"obs_uids_{tag}_{t0}_{t1}.npz"


def uids_observed_in_time_range(
    observations_csv_gz: Path,
    t0_us: int,
    t1_us: int,
    *,
    cache_dir: Path | None = None,
    chunksize: int = 1_000_000,
) -> np.ndarray:
    """Unique point uids with >=1 observation in ``[t0_us, t1_us]``."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = _obs_uid_cache_path(cache_dir, observations_csv_gz, t0_us, t1_us)
        if cpath.is_file():
            return np.load(cpath)["uids"].astype(np.int64)

    seen: set[int] = set()
    rows = 0
    hits = 0
    for chunk in pd.read_csv(
        observations_csv_gz,
        chunksize=chunksize,
        dtype={"uid": np.int64, "frame_tracking_timestamp_us": np.int64},
        usecols=["uid", "frame_tracking_timestamp_us"],
    ):
        rows += len(chunk)
        ts = chunk["frame_tracking_timestamp_us"].to_numpy()
        mask = (ts >= t0_us) & (ts <= t1_us)
        if not np.any(mask):
            continue
        hits += int(mask.sum())
        for uid in chunk["uid"].to_numpy()[mask]:
            seen.add(int(uid))
        if rows % 20_000_000 == 0:
            print(
                f"[mps-obs] scanned {rows:,} rows -> {len(seen):,} uids "
                f"({hits:,} hits in window)",
                flush=True,
            )

    uids = np.fromiter(seen, dtype=np.int64)
    uids.sort()
    print(
        f"[mps-obs] time window [{t0_us}, {t1_us}]: {len(uids):,} uids "
        f"from {hits:,}/{rows:,} observation rows",
        flush=True,
    )
    if cache_dir is not None:
        np.savez_compressed(cpath, uids=uids)
    return uids


def load_semidense_xyz_uid(
    points_csv_gz: Path,
    *,
    inverse_distance_std_threshold: float = 0.005,
    distance_std_threshold: float = 0.15,
    filter_confidence: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load semidense XYZ+uid. Defaults match Aria docs (looser than viz defaults)."""
    import projectaria_tools.core.mps as mps
    from projectaria_tools.core.mps.utils import filter_points_from_confidence

    points = mps.read_global_point_cloud(str(points_csv_gz))
    if filter_confidence:
        points = filter_points_from_confidence(
            points, inverse_distance_std_threshold, distance_std_threshold
        )
    xyz = np.array([p.position_world for p in points], dtype=np.float32)
    uids = np.array([p.uid for p in points], dtype=np.int64)
    return xyz, uids




def build_sequence_mps_obs_pcd(
    key: str,
    can_aria_traj_T: np.ndarray,
    *,
    cache_dir: Path,
    obs_uid_cache_dir: Path | None = None,
    voxel: float = 0.01,
    max_points: int = 400_000,
    distance_std_threshold: float = 0.15,
    inverse_distance_std_threshold: float = 0.005,
    filter_confidence: bool = True,
    max_dist: float = 5.0,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (xyz_can, rgb) or None if MPS assets missing.

    ``filter_confidence`` (default True) drops MPS semidense points with
    poor inverse-distance/distance-std - without it, a handful of
    low-parallax SLAM points can land thousands of meters away and blow up
    the scene extent. ``max_dist`` bounds how far a point may sit from the
    nearest sampled trajectory position; 5m covers a room-scale capture
    (checked against ViPE dense reconstructions of the same sequences)
    without keeping the diffuse halo of distant points.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{key.replace('/', '_')}.npz"
    if out.is_file() and not force:
        z = np.load(out)
        cached_version = int(z["cache_version"]) if "cache_version" in z.files else -1
        if cached_version == CACHE_VERSION:
            return np.asarray(z["xyz"], np.float32), np.asarray(z["rgb"], np.uint8)
        print(f"[mps-obs] stale cache (pre-v{CACHE_VERSION}) -> rebuilding {out.name}", flush=True)

    ep, _, take = parse_pred_key(key)
    paths = mps_paths(ep)
    if paths is None:
        print(f"[mps-obs] missing slam+observations under plus_data/{ep}", flush=True)
        return None

    bundle = find_bundle(take)
    if bundle is None:
        print(f"[mps-obs] no bundle for {take}", flush=True)
        return None

    t0, t1 = bundle_device_time_range_us(bundle)
    uid_cache = obs_uid_cache_dir or (cache_dir / "_obs_uid_cache")
    keep_uids = uids_observed_in_time_range(
        paths["semidense_observations"], t0, t1, cache_dir=uid_cache
    )
    if keep_uids.size == 0:
        print(f"[mps-obs] no observed uids in window for {key[:70]}", flush=True)
        return None

    print(f"[mps-obs] loading semidense points for {ep}...", flush=True)
    xyz_w, uids = load_semidense_xyz_uid(
        paths["semidense_points"],
        filter_confidence=filter_confidence,
        inverse_distance_std_threshold=inverse_distance_std_threshold,
        distance_std_threshold=distance_std_threshold,
    )
    keep = np.isin(uids, keep_uids, assume_unique=False)
    xyz_w = xyz_w[keep]
    print(
        f"[mps-obs] obs-window {xyz_w.shape[0]:,} / {keep.size:,} "
        f"(uids={keep_uids.size:,}, confidence_filter={filter_confidence})",
        flush=True,
    )
    if xyz_w.shape[0] == 0:
        return None

    world_c2w = bundle_camera_c2w(bundle)
    device_ts = bundle_device_timestamps_us(bundle)
    R, t = world_to_can_from_traj(
        world_c2w, can_aria_traj_T, device_timestamp_us=device_ts
    )
    xyz = transform_points(xyz_w, R, t)
    rgb = colors_by_height(xyz)
    n_before = xyz.shape[0]
    xyz, rgb = clip_points_near_traj(
        xyz, rgb, np.asarray(can_aria_traj_T[:, :3, 3], np.float32), max_dist=max_dist
    )
    print(
        f"[mps-obs] traj-clip {xyz.shape[0]:,}/{n_before:,} pts within {max_dist:g}m of cam",
        flush=True,
    )
    xyz, rgb = voxel_downsample(xyz, rgb, voxel=voxel, max_points=max_points)

    np.savez_compressed(
        out,
        xyz=xyz,
        rgb=rgb,
        t0_us=np.int64(t0),
        t1_us=np.int64(t1),
        n_uids=np.int64(keep_uids.size),
        ep=np.array(ep),
        cache_version=np.int64(CACHE_VERSION),
    )
    print(f"[mps-obs] cached {xyz.shape[0]:,} pts -> {out.name}", flush=True)
    return xyz, rgb
