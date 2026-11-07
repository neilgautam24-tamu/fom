"""Load Nymeria Plus Aria SLAM data for viser visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

RECORDING_DEVICES = {
    "head": "recording_head",
    "left_wrist": "recording_lwrist",
    "right_wrist": "recording_rwrist",
    "observer": "recording_observer",
}


def se3_from_aria_transform(transform: dict) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from Aria ``T_*`` JSON."""
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = transform["Translation"]
    w, xyz = transform["UnitQuaternion"]
    mat[:3, :3] = Rotation.from_quat([xyz[0], xyz[1], xyz[2], w]).as_matrix()
    return mat


def wxyz_from_matrix(rot: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rot).as_quat(scalar_first=True)


@dataclass(frozen=True)
class RgbCameraCalibration:
    label: str
    width: int
    height: int
    fov_y_rad: float
    aspect: float
    t_device_camera: np.ndarray

    @property
    def t_world_camera(self) -> np.ndarray:
        return self.t_device_camera


def _fisheye_vertical_fov(params: list[float], height: int) -> float:
    focal = float(params[0])
    return float(2.0 * np.arctan(max(height, 1) / (2.0 * max(focal, 1e-6))))


def load_rgb_camera_calibration(
    calibration_jsonl: Path,
    camera_label: str = "camera-rgb",
) -> RgbCameraCalibration:
    if not calibration_jsonl.is_file():
        raise FileNotFoundError(f"Missing online calibration: {calibration_jsonl}")

    with calibration_jsonl.open() as handle:
        record = json.loads(handle.readline())

    for cam in record["CameraCalibrations"]:
        if cam.get("Label") != camera_label:
            continue
        params = cam["Projection"]["Params"]
        width = int(cam.get("ImageWidth") or 1408)
        height = int(cam.get("ImageHeight") or 1408)
        return RgbCameraCalibration(
            label=camera_label,
            width=width,
            height=height,
            fov_y_rad=_fisheye_vertical_fov(params, height),
            aspect=float(width) / float(height),
            t_device_camera=se3_from_aria_transform(cam["T_Device_Camera"]),
        )

    raise ValueError(f"Camera {camera_label!r} not found in {calibration_jsonl}")


@dataclass
class ClosedLoopTrajectory:
    timestamps_us: np.ndarray
    positions: np.ndarray
    rotations: Rotation

    @classmethod
    def from_csv(cls, csv_path: Path) -> ClosedLoopTrajectory:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing closed-loop trajectory: {csv_path}")

        df = pd.read_csv(
            csv_path,
            usecols=[
                "tracking_timestamp_us",
                "tx_world_device",
                "ty_world_device",
                "tz_world_device",
                "qx_world_device",
                "qy_world_device",
                "qz_world_device",
                "qw_world_device",
            ],
        )
        quat_xyzw = df[
            ["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]
        ].to_numpy(dtype=np.float64)
        return cls(
            timestamps_us=df["tracking_timestamp_us"].to_numpy(dtype=np.int64),
            positions=df[
                ["tx_world_device", "ty_world_device", "tz_world_device"]
            ].to_numpy(dtype=np.float64),
            rotations=Rotation.from_quat(quat_xyzw),
        )

    def pose_at_us(self, timestamp_us: int) -> tuple[np.ndarray, np.ndarray]:
        ts = self.timestamps_us
        if timestamp_us <= ts[0]:
            idx = 0
        elif timestamp_us >= ts[-1]:
            idx = len(ts) - 1
        else:
            idx = int(np.searchsorted(ts, timestamp_us))
            idx = min(max(idx, 1), len(ts) - 1)

        t0, t1 = ts[idx - 1], ts[idx]
        if t1 <= t0:
            alpha = 0.0
        else:
            alpha = float(timestamp_us - t0) / float(t1 - t0)
            alpha = min(max(alpha, 0.0), 1.0)

        position = (1.0 - alpha) * self.positions[idx - 1] + alpha * self.positions[idx]
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.concatenate([self.rotations[idx - 1], self.rotations[idx]]),
        )
        return position, slerp([alpha]).as_matrix()[0]


def camera_pose_world(
    trajectory: ClosedLoopTrajectory,
    t_device_camera: np.ndarray,
    timestamp_us: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return OpenCV camera pose in the shared Nymeria world frame."""
    t_world_device = np.eye(4, dtype=np.float64)
    position, rotation = trajectory.pose_at_us(timestamp_us)
    t_world_device[:3, :3] = rotation
    t_world_device[:3, 3] = position
    t_world_camera = t_world_device @ t_device_camera
    return t_world_camera[:3, 3], t_world_camera[:3, :3]


def compute_body_to_device_offset_us(motion_vrs: Path, body_timestamp_us: int) -> int:
    """Map the first body timestamp to Aria device time using motion.vrs TIME_CODE."""
    from projectaria_tools.core import data_provider

    provider = data_provider.create_vrs_data_provider(str(motion_vrs))
    if provider is None:
        raise RuntimeError(f"Could not open motion VRS: {motion_vrs}")

    timecode_ns = int(body_timestamp_us) * 1000
    device_ns = provider.convert_from_timecode_to_device_time_ns(timecode_ns)
    return int(device_ns // 1000 - body_timestamp_us)


def body_timestamp_to_device_us(body_timestamp_us: int, offset_us: int) -> int:
    return int(body_timestamp_us) + int(offset_us)


@dataclass
class PointCloudScene:
    """Single-layer or Rerun-style dual-layer point cloud for viser."""

    points: np.ndarray
    colors: np.ndarray
    fg_points: np.ndarray | None = None
    fg_colors: np.ndarray | None = None
    bg_point_size: float = 0.006
    fg_point_size: float = 0.012

    @property
    def is_dual_layer(self) -> bool:
        return self.fg_points is not None and self.fg_points.shape[0] > 0


def _voxel_keep_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0 or voxel_size <= 0.0:
        return np.arange(points.shape[0])

    vox = np.floor(points / voxel_size).astype(np.int32)
    vox -= vox.min(axis=0)
    dims = vox.max(axis=0) + 1
    keys = (
        vox[:, 0].astype(np.int64) * int(dims[1]) + vox[:, 1]
    ) * int(dims[2]) + vox[:, 2]
    _, idx = np.unique(keys, return_index=True)
    return np.sort(idx)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep one point per voxel for spatially uniform coverage."""
    return points[_voxel_keep_indices(points, voxel_size)]


def adaptive_voxel_downsample(
    points: np.ndarray,
    max_points: int,
    voxel_size: float | None = None,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Downsample with voxels (not random picks) so structure stays readable."""
    downsampled, _ = adaptive_voxel_downsample_indexed(
        points,
        np.arange(points.shape[0]),
        max_points=max_points,
        voxel_size=voxel_size,
        seed=seed,
    )
    return downsampled


def adaptive_voxel_downsample_indexed(
    points: np.ndarray,
    indices: np.ndarray,
    max_points: int,
    voxel_size: float | None = None,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel downsample while keeping parallel index arrays (e.g. point uids)."""
    if points.shape[0] <= max_points:
        return points, indices

    if voxel_size is None or voxel_size <= 0.0:
        extent = points.max(axis=0) - points.min(axis=0)
        volume = float(np.prod(np.maximum(extent, 1e-3)))
        voxel_size = max((volume / float(max(max_points, 1))) ** (1.0 / 3.0), 0.012)

    keep = np.arange(points.shape[0])
    for _ in range(16):
        keep = _voxel_keep_indices(points, voxel_size)
        if keep.shape[0] <= max_points:
            return points[keep], indices[keep]
        voxel_size *= 1.10

    keep = _voxel_keep_indices(points, voxel_size)
    points = points[keep]
    indices = indices[keep]
    if points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        pick = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[pick]
        indices = indices[pick]
    return points, indices


def crop_points_to_aabb(
    points: np.ndarray,
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
) -> np.ndarray:
    if points.size == 0:
        return points
    mask = np.all((points >= min_xyz) & (points <= max_xyz), axis=1)
    return points[mask]


def load_camera_serial_map(calibration_jsonl: Path) -> dict[str, str]:
    """Map Aria camera serial numbers to VRS stream labels."""
    with calibration_jsonl.open() as handle:
        record = json.loads(handle.readline())
    return {
        cam["SerialNumber"]: cam["Label"]
        for cam in record.get("CameraCalibrations", [])
        if cam.get("SerialNumber") and cam.get("Label")
    }


def estimate_point_normals(points: np.ndarray, neighbors: int = 24) -> np.ndarray:
    """Estimate surface normals with local PCA (batch eigen-decomposition)."""
    if points.shape[0] < 4:
        return np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (points.shape[0], 1))

    from scipy.spatial import cKDTree

    k = min(neighbors, points.shape[0])
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    nbrs = points[idx]
    centered = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered)
    _, vectors = np.linalg.eigh(cov)
    normals = vectors[:, :, 0].astype(np.float32)
    flip = normals[:, 2] < 0.0
    normals[flip] *= -1.0
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return normals / norms


def apply_normal_shading(
    base_colors: np.ndarray,
    normals: np.ndarray,
    *,
    light_dir: tuple[float, float, float] = (0.35, -0.25, 0.90),
    ambient: float = 0.42,
    diffuse: float = 0.58,
) -> np.ndarray:
    """Lambertian shading to emphasize local surface orientation."""
    if base_colors.size == 0:
        return base_colors

    light = np.asarray(light_dir, dtype=np.float32)
    light /= max(float(np.linalg.norm(light)), 1e-6)
    ndotl = np.clip(normals @ light, 0.0, 1.0)
    shade = ambient + diffuse * ndotl
    shaded = base_colors.astype(np.float32) * shade[:, None]
    return np.clip(shaded, 0.0, 255.0).astype(np.uint8)


def point_cloud_colors_by_height(points: np.ndarray) -> np.ndarray:
    """Height colormap: deep blue (low) -> teal -> sand (high), tuned for white bg."""
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    z = points[:, 2]
    z0, z1 = np.percentile(z, [3.0, 97.0])
    if z1 <= z0:
        z1 = z0 + 1e-3
    t = np.clip((z - z0) / (z1 - z0), 0.0, 1.0)
    stops = np.array(
        [
            [0.00, 0.12, 0.28, 0.62],
            [0.40, 0.18, 0.55, 0.72],
            [0.70, 0.42, 0.72, 0.68],
            [1.00, 0.82, 0.68, 0.46],
        ],
        dtype=np.float32,
    )
    colors = np.zeros((points.shape[0], 3), dtype=np.float32)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i, 0], stops[i, 1:]
        t1, c1 = stops[i + 1, 0], stops[i + 1, 1:]
        mask = (t >= t0) & (t <= t1 if i == len(stops) - 2 else t < t1)
        if not np.any(mask):
            continue
        alpha = (t[mask] - t0) / max(t1 - t0, 1e-6)
        colors[mask] = (1.0 - alpha)[:, None] * c0 + alpha[:, None] * c1
    return (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)


# Muted slate background so photo-colored foreground points read clearly on white.
POINT_CLOUD_BG_RGB = np.array([128, 136, 152], dtype=np.uint8)

_OBS_INDEX_VERSION = 2


def _intensity_to_photo_rgb(intensity: np.ndarray) -> np.ndarray:
    """Map SLAM grayscale samples to a warm, lifted photo tone."""
    x = intensity.astype(np.float32) / 255.0
    x = np.clip((x - 0.06) / 0.88, 0.0, 1.0)
    rgb = np.stack(
        [
            np.clip(45.0 + x * 210.0, 0.0, 255.0),
            np.clip(40.0 + x * 200.0, 0.0, 255.0),
            np.clip(35.0 + x * 185.0, 0.0, 255.0),
        ],
        axis=-1,
    )
    return rgb.astype(np.uint8)


def _colors_from_turbo_depth(depths_m: np.ndarray) -> np.ndarray:
    """High-contrast turbo depth (near=warm, far=cool) fallback palette."""
    valid = np.isfinite(depths_m) & (depths_m > 0.0)
    near_m, far_m = 0.25, 4.5
    t = np.clip((depths_m - near_m) / (far_m - near_m + 1e-12), 0.0, 1.0)
    t = 1.0 - t  # near -> 1, far -> 0
    # Polynomial approximation of matplotlib turbo.
    r = np.clip(34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))), 0, 255)
    g = np.clip(23.31 + t * (557.33 + t * (-1220.04 + t * (8624.82 + t * (-13627.20 + t * 5300.15)))), 0, 255)
    b = np.clip(27.20 + t * (321.24 + t * (2323.04 + t * (-7608.48 + t * (8550.59 + t * (-3330.12))))), 0, 255)
    colors = np.stack([r, g, b], axis=-1).astype(np.uint8)
    colors[~valid] = (80, 85, 95)
    return colors


@dataclass
class ObservationIndex:
    uids: np.ndarray
    timestamps_us: np.ndarray
    u: np.ndarray
    v: np.ndarray
    serial_ids: np.ndarray | None = None
    serials: tuple[str, ...] = ()

    @property
    def has_serials(self) -> bool:
        return self.serial_ids is not None and len(self.serials) > 0

    def lookup(
        self, query_uids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (timestamps, u, v, serial_ids, found_mask) for query uids."""
        idx = np.searchsorted(self.uids, query_uids)
        idx = np.clip(idx, 0, max(len(self.uids) - 1, 0))
        matched = self.uids[idx] == query_uids
        n = len(query_uids)
        ts = np.zeros(n, dtype=np.int64)
        uu = np.zeros(n, dtype=np.float32)
        vv = np.zeros(n, dtype=np.float32)
        sid = np.zeros(n, dtype=np.uint8)
        hit = idx[matched]
        ts[matched] = self.timestamps_us[hit]
        uu[matched] = self.u[hit]
        vv[matched] = self.v[hit]
        if self.serial_ids is not None:
            sid[matched] = self.serial_ids[hit]
        return ts, uu, vv, sid, matched


def _observation_index_cache_path(observations_csv_gz: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = observations_csv_gz.parent.parent.name
    return cache_dir / f"obs_index_v{_OBS_INDEX_VERSION}_{tag}.npz"


def load_observation_index(
    observations_csv_gz: Path,
    cache_dir: Path | None,
    *,
    rebuild: bool = False,
) -> ObservationIndex:
    """Build or load a uid-sorted first-observation index (one-time ~3 min scan)."""
    cache_path = (
        _observation_index_cache_path(observations_csv_gz, cache_dir)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and cache_path.is_file() and not rebuild:
        data = np.load(cache_path, allow_pickle=True)
        print(f"[aria] loaded observation index from {cache_path.name}")
        serials = tuple(str(s) for s in data["serials"].tolist()) if "serials" in data else ()
        serial_ids = (
            data["serial_ids"].astype(np.uint8) if "serial_ids" in data else None
        )
        return ObservationIndex(
            uids=data["uids"].astype(np.int64),
            timestamps_us=data["timestamps_us"].astype(np.int64),
            u=data["u"].astype(np.float32),
            v=data["v"].astype(np.float32),
            serial_ids=serial_ids,
            serials=serials,
        )

    if not observations_csv_gz.is_file():
        return ObservationIndex(
            uids=np.zeros(0, dtype=np.int64),
            timestamps_us=np.zeros(0, dtype=np.int64),
            u=np.zeros(0, dtype=np.float32),
            v=np.zeros(0, dtype=np.float32),
        )

    print(f"[aria] building observation index from {observations_csv_gz.name} (one-time)")
    seen: set[int] = set()
    uid_list: list[int] = []
    ts_list: list[int] = []
    u_list: list[float] = []
    v_list: list[float] = []
    serial_list: list[str] = []
    serial_to_id: dict[str, int] = {}
    rows_seen = 0
    for chunk in pd.read_csv(
        observations_csv_gz,
        chunksize=1_000_000,
        dtype={
            "uid": np.int64,
            "frame_tracking_timestamp_us": np.int64,
            "camera_serial": str,
            "u": np.float32,
            "v": np.float32,
        },
        usecols=["uid", "frame_tracking_timestamp_us", "camera_serial", "u", "v"],
    ):
        rows_seen += len(chunk)
        for row in chunk.itertuples(index=False):
            uid = int(row.uid)
            if uid in seen:
                continue
            seen.add(uid)
            uid_list.append(uid)
            ts_list.append(int(row.frame_tracking_timestamp_us))
            u_list.append(float(row.u))
            v_list.append(float(row.v))
            serial_list.append(str(row.camera_serial))
            if str(row.camera_serial) not in serial_to_id:
                serial_to_id[str(row.camera_serial)] = len(serial_to_id)
        if rows_seen % 10_000_000 == 0:
            print(f"[aria]   {rows_seen:,} rows, {len(uid_list):,} unique uids", flush=True)

    serials = tuple(serial_to_id.keys())
    sid_array = np.array(
        [serial_to_id[s] for s in serial_list], dtype=np.uint8
    )
    order = np.argsort(uid_list)
    index = ObservationIndex(
        uids=np.asarray(uid_list, dtype=np.int64)[order],
        timestamps_us=np.asarray(ts_list, dtype=np.int64)[order],
        u=np.asarray(u_list, dtype=np.float32)[order],
        v=np.asarray(v_list, dtype=np.float32)[order],
        serial_ids=sid_array[order],
        serials=serials,
    )
    print(f"[aria] observation index: {index.uids.shape[0]:,} uids from {rows_seen:,} rows")
    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            uids=index.uids,
            timestamps_us=index.timestamps_us,
            u=index.u,
            v=index.v,
            serial_ids=index.serial_ids,
            serials=np.array(serials, dtype=object),
        )
        print(f"[aria] cached observation index to {cache_path.name}")
    return index


def _device_z_depth(
    points_world: np.ndarray,
    trajectory: ClosedLoopTrajectory,
    timestamps_us: np.ndarray,
) -> np.ndarray:
    depths = np.zeros(points_world.shape[0], dtype=np.float32)
    for ts in np.unique(timestamps_us):
        if ts <= 0:
            continue
        mask = timestamps_us == ts
        pos, rot = trajectory.pose_at_us(int(ts))
        depths[mask] = (points_world[mask] - pos) @ rot[:, 2]
    return depths


def _colors_from_slam_photos(
    timestamps_us: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    serial_ids: np.ndarray,
    serials: tuple[str, ...],
    image_reader: AriaSlamImageReader,
) -> np.ndarray:
    colors = np.zeros((len(timestamps_us), 3), dtype=np.uint8)
    groups: dict[tuple[str, int], list[tuple[int, float, float]]] = {}
    for i, (ts, ui, vi, sid) in enumerate(
        zip(timestamps_us.tolist(), u.tolist(), v.tolist(), serial_ids.tolist())
    ):
        if sid >= len(serials):
            continue
        groups.setdefault((serials[sid], int(ts)), []).append((i, ui, vi))

    for (serial, ts_us), samples in groups.items():
        try:
            gray = image_reader._gray_image(serial, ts_us)
        except Exception:
            continue
        h, w = gray.shape[:2]
        for i, ui, vi in samples:
            ui_i = int(np.clip(round(ui), 0, w - 1))
            vi_i = int(np.clip(round(vi), 0, h - 1))
            colors[i] = _intensity_to_photo_rgb(np.array([int(gray[vi_i, ui_i])]))[0]
    return colors


def _downsample_by_count(
    points: np.ndarray,
    extra: np.ndarray,
    max_count: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_count:
        return points, extra
    rng = np.random.default_rng(seed)
    pick = rng.choice(points.shape[0], size=max_count, replace=False)
    return points[pick], extra[pick]


def _stream_observations_for_uids(
    observations_csv_gz: Path,
    uid_set: set[int],
    *,
    chunk_size: int = 750_000,
) -> dict[int, tuple[int, str, float, float]]:
    """Scan observations once and keep the first view for each point uid."""
    if not observations_csv_gz.is_file() or not uid_set:
        return {}

    found: dict[int, tuple[int, str, float, float]] = {}
    rows_seen = 0
    for chunk in pd.read_csv(
        observations_csv_gz,
        chunksize=chunk_size,
        dtype={
            "uid": np.int64,
            "frame_tracking_timestamp_us": np.int64,
            "camera_serial": str,
            "u": np.float32,
            "v": np.float32,
        },
    ):
        rows_seen += len(chunk)
        sub = chunk[chunk["uid"].isin(uid_set)]
        for row in sub.itertuples(index=False):
            uid = int(row.uid)
            if uid in found:
                continue
            found[uid] = (
                int(row.frame_tracking_timestamp_us),
                str(row.camera_serial),
                float(row.u),
                float(row.v),
            )
        if len(found) >= len(uid_set):
            break
        if rows_seen % (chunk_size * 8) == 0:
            print(
                f"[aria] observations scan: {rows_seen:,} rows, "
                f"{len(found):,}/{len(uid_set):,} uids",
                flush=True,
            )

    print(
        f"[aria] observations matched {len(found):,}/{len(uid_set):,} points "
        f"({rows_seen:,} rows scanned)",
        flush=True,
    )
    return found


class AriaSlamImageReader:
    """Fetch grayscale SLAM camera images used to color semidense points."""

    def __init__(self, data_vrs: Path, serial_to_label: dict[str, str]):
        from projectaria_tools.core import data_provider

        self._provider = data_provider.create_vrs_data_provider(str(data_vrs))
        if self._provider is None:
            raise RuntimeError(f"Could not open VRS: {data_vrs}")

        self._streams: dict[str, object] = {}
        for stream_id in self._provider.get_all_streams():
            label = self._provider.get_label_from_stream_id(stream_id)
            for serial, mapped in serial_to_label.items():
                if mapped == label:
                    self._streams[serial] = stream_id

    @lru_cache(maxsize=96)
    def _gray_image(self, camera_serial: str, tracking_timestamp_us: int) -> np.ndarray:
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

        stream_id = self._streams[camera_serial]
        image_data, _record = self._provider.get_image_data_by_time_ns(
            stream_id,
            int(tracking_timestamp_us) * 1000,
            TimeDomain.DEVICE_TIME,
            TimeQueryOptions.CLOSEST,
        )
        gray = image_data.to_numpy_array()
        if gray.ndim == 3:
            gray = gray[..., 0]
        lo, hi = np.percentile(gray, [2.0, 98.0])
        if hi > lo:
            gray = np.clip((gray.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
            gray = (gray * 255.0).astype(np.uint8)
        return gray

    def sample_rgb(
        self,
        camera_serial: str,
        tracking_timestamp_us: int,
        u: float,
        v: float,
    ) -> np.ndarray:
        gray = self._gray_image(camera_serial, tracking_timestamp_us)
        h, w = gray.shape[:2]
        ui = int(np.clip(round(u), 0, w - 1))
        vi = int(np.clip(round(v), 0, h - 1))
        value = int(gray[vi, ui])
        return np.array([value, value, value], dtype=np.uint8)


def color_points_from_observations(
    uids: np.ndarray,
    observations: dict[int, tuple[int, str, float, float]],
    image_reader: AriaSlamImageReader,
    *,
    fallback_colors: np.ndarray,
) -> np.ndarray:
    colors = fallback_colors.copy()
    groups: dict[tuple[str, int], list[tuple[int, float, float]]] = {}
    for i, uid in enumerate(uids.tolist()):
        obs = observations.get(int(uid))
        if obs is None:
            continue
        ts_us, serial, u, v = obs
        groups.setdefault((serial, ts_us), []).append((i, u, v))

    for (serial, ts_us), samples in groups.items():
        try:
            gray = image_reader._gray_image(serial, ts_us)
        except Exception:
            continue
        h, w = gray.shape[:2]
        for i, u, v in samples:
            ui = int(np.clip(round(u), 0, w - 1))
            vi = int(np.clip(round(v), 0, h - 1))
            value = int(gray[vi, ui])
            colors[i] = (value, value, value)
    return colors


def _point_cloud_cache_path(
    cache_dir: Path,
    points_csv_gz: Path,
    *,
    max_points: int,
    distance_std_threshold: float,
    voxel_size: float | None,
    color_mode: str,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = f"{color_mode}_v2" if color_mode == "rerun" else color_mode
    tag = (
        f"{points_csv_gz.parent.parent.name}_"
        f"{max_points}_{distance_std_threshold:.3f}_"
        f"{0.0 if voxel_size is None else voxel_size:.3f}_{mode_tag}"
    )
    return cache_dir / f"semidense_pc_{tag}.npz"


def crop_point_cloud_scene(
    scene: PointCloudScene,
    crop_aabb: tuple[np.ndarray, np.ndarray],
) -> PointCloudScene:
    """Spatially crop a point cloud (including optional foreground layer)."""
    lo = np.asarray(crop_aabb[0], dtype=np.float32)
    hi = np.asarray(crop_aabb[1], dtype=np.float32)
    if scene.points.size == 0:
        return scene

    mask = np.all((scene.points >= lo) & (scene.points <= hi), axis=1)
    points = scene.points[mask]
    colors = scene.colors[mask]
    fg_points = scene.fg_points
    fg_colors = scene.fg_colors
    if fg_points is not None and fg_colors is not None and fg_points.size:
        fg_mask = np.all((fg_points >= lo) & (fg_points <= hi), axis=1)
        fg_points = fg_points[fg_mask]
        fg_colors = fg_colors[fg_mask]
    return PointCloudScene(
        points=points,
        colors=colors,
        fg_points=fg_points,
        fg_colors=fg_colors,
        bg_point_size=scene.bg_point_size,
        fg_point_size=scene.fg_point_size,
    )


def _load_cached_point_cloud(cache_path: Path) -> PointCloudScene:
    cached = np.load(cache_path)
    fg_points = cached["fg_points"].astype(np.float32) if "fg_points" in cached else None
    fg_colors = cached["fg_colors"].astype(np.uint8) if "fg_colors" in cached else None
    return PointCloudScene(
        points=cached["points"].astype(np.float32),
        colors=cached["colors"].astype(np.uint8),
        fg_points=fg_points,
        fg_colors=fg_colors,
        bg_point_size=float(cached["bg_point_size"]) if "bg_point_size" in cached else 0.006,
        fg_point_size=float(cached["fg_point_size"]) if "fg_point_size" in cached else 0.012,
    )


def _save_cached_point_cloud(cache_path: Path, scene: PointCloudScene) -> None:
    payload = {
        "points": scene.points,
        "colors": scene.colors,
        "bg_point_size": np.float32(scene.bg_point_size),
        "fg_point_size": np.float32(scene.fg_point_size),
    }
    if scene.fg_points is not None and scene.fg_colors is not None:
        payload["fg_points"] = scene.fg_points
        payload["fg_colors"] = scene.fg_colors
    np.savez_compressed(cache_path, **payload)


def load_semidense_point_cloud(
    points_csv_gz: Path,
    *,
    max_points: int = 500_000,
    inverse_distance_std_threshold: float = 0.001,
    distance_std_threshold: float = 0.05,
    voxel_size: float | None = None,
    crop_aabb: tuple[np.ndarray, np.ndarray] | None = None,
    observations_csv_gz: Path | None = None,
    data_vrs: Path | None = None,
    camera_serial_map: dict[str, str] | None = None,
    trajectory: ClosedLoopTrajectory | None = None,
    color_mode: str = "rerun",
    cache_dir: Path | None = None,
    obs_cache_dir: Path | None = None,
    use_cache: bool = True,
    rebuild_obs_index: bool = False,
    seed: int = 0,
) -> PointCloudScene:
    import projectaria_tools.core.mps as mps
    from projectaria_tools.core.mps.utils import filter_points_from_confidence

    if color_mode not in {"rerun", "height", "shaded", "image"}:
        raise ValueError(f"Unknown color_mode {color_mode!r}")

    cache_path = None
    scene: PointCloudScene | None = None
    if cache_dir is not None and use_cache:
        cache_path = _point_cloud_cache_path(
            cache_dir,
            points_csv_gz,
            max_points=max_points,
            distance_std_threshold=distance_std_threshold,
            voxel_size=voxel_size,
            color_mode=color_mode,
        )
        if cache_path.is_file():
            print(f"[aria] loaded cached point cloud from {cache_path.name}")
            scene = _load_cached_point_cloud(cache_path)

    if scene is not None:
        if crop_aabb is not None:
            scene = crop_point_cloud_scene(scene, crop_aabb)
        return scene

    if not points_csv_gz.is_file():
        raise FileNotFoundError(f"Missing semidense point cloud: {points_csv_gz}")

    points = mps.read_global_point_cloud(str(points_csv_gz))
    filtered = filter_points_from_confidence(
        points,
        inverse_distance_std_threshold,
        distance_std_threshold,
    )
    xyz = np.array([p.position_world for p in filtered], dtype=np.float32)
    uids = np.array([p.uid for p in filtered], dtype=np.int64)
    if xyz.size == 0:
        return PointCloudScene(
            points=xyz.reshape(0, 3),
            colors=np.zeros((0, 3), dtype=np.uint8),
        )

    if crop_aabb is not None:
        mask = np.all((xyz >= crop_aabb[0]) & (xyz <= crop_aabb[1]), axis=1)
        xyz = xyz[mask]
        uids = uids[mask]
        if xyz.size == 0:
            return PointCloudScene(
                points=xyz.reshape(0, 3),
                colors=np.zeros((0, 3), dtype=np.uint8),
            )

    if color_mode == "rerun":
        bg_xyz, bg_uids = _downsample_by_count(xyz, uids, max_points, seed=seed)
        bg_colors = np.broadcast_to(POINT_CLOUD_BG_RGB, (bg_xyz.shape[0], 3)).copy()

        fg_xyz = bg_xyz
        fg_colors = np.broadcast_to(POINT_CLOUD_BG_RGB, (bg_xyz.shape[0], 3)).copy()
        if (
            observations_csv_gz is not None
            and trajectory is not None
            and observations_csv_gz.is_file()
        ):
            obs_index = load_observation_index(
                observations_csv_gz,
                obs_cache_dir if obs_cache_dir is not None else cache_dir,
                rebuild=rebuild_obs_index,
            )
            ts, obs_u, obs_v, obs_sid, found = obs_index.lookup(bg_uids)
            if np.any(found):
                fg_xyz = bg_xyz[found]
                fg_ts = ts[found]
                fg_u = obs_u[found]
                fg_v = obs_v[found]
                fg_sid = obs_sid[found]
                use_photos = (
                    obs_index.has_serials
                    and data_vrs is not None
                    and camera_serial_map
                    and data_vrs.is_file()
                )
                if use_photos:
                    slam_reader = AriaSlamImageReader(data_vrs, camera_serial_map)
                    fg_colors = _colors_from_slam_photos(
                        fg_ts, fg_u, fg_v, fg_sid, obs_index.serials, slam_reader
                    )
                    color_label = "SLAM photo"
                else:
                    depths = _device_z_depth(fg_xyz, trajectory, fg_ts)
                    fg_colors = _colors_from_turbo_depth(depths)
                    color_label = "turbo depth"
                print(
                    f"[aria] rerun foreground: {fg_xyz.shape[0]:,} points ({color_label})"
                )
        scene = PointCloudScene(
            points=bg_xyz,
            colors=bg_colors,
            fg_points=fg_xyz,
            fg_colors=fg_colors,
            bg_point_size=0.005,
            fg_point_size=0.013,
        )
    else:
        vs = None if voxel_size is None or voxel_size <= 0.0 else float(voxel_size)
        xyz, uids = adaptive_voxel_downsample_indexed(
            xyz, uids, max_points=max_points, voxel_size=vs, seed=seed
        )
        height_colors = point_cloud_colors_by_height(xyz)
        if color_mode == "height":
            colors = height_colors
        elif color_mode == "shaded":
            normals = estimate_point_normals(xyz)
            colors = apply_normal_shading(height_colors, normals)
        else:
            base = height_colors
            if (
                observations_csv_gz is not None
                and data_vrs is not None
                and camera_serial_map
                and observations_csv_gz.is_file()
                and data_vrs.is_file()
            ):
                print("[aria] coloring points from SLAM observations")
                obs = _stream_observations_for_uids(
                    observations_csv_gz, set(int(u) for u in uids.tolist())
                )
                slam_reader = AriaSlamImageReader(data_vrs, camera_serial_map)
                photo = color_points_from_observations(
                    uids, obs, slam_reader, fallback_colors=base
                )
                normals = estimate_point_normals(xyz)
                colors = apply_normal_shading(photo, normals, ambient=0.55, diffuse=0.45)
            else:
                normals = estimate_point_normals(xyz)
                colors = apply_normal_shading(height_colors, normals)
        scene = PointCloudScene(points=xyz, colors=colors)

    if cache_path is not None:
        _save_cached_point_cloud(cache_path, scene)
        print(f"[aria] cached point cloud to {cache_path.name}")
    return scene


class AriaRgbReader:
    """Fetch RGB frames from an Aria ``data.vrs`` recording."""

    def __init__(self, data_vrs: Path):
        from projectaria_tools.core import data_provider

        if not data_vrs.is_file():
            raise FileNotFoundError(f"Missing RGB VRS: {data_vrs}")

        self._provider = data_provider.create_vrs_data_provider(str(data_vrs))
        if self._provider is None:
            raise RuntimeError(f"Could not open RGB VRS: {data_vrs}")

        self._stream_id = None
        for stream_id in self._provider.get_all_streams():
            if self._provider.get_label_from_stream_id(stream_id) == "camera-rgb":
                self._stream_id = stream_id
                break
        if self._stream_id is None:
            raise RuntimeError(f"No camera-rgb stream in {data_vrs}")

        cfg = self._provider.get_image_configuration(self._stream_id)
        self.width = int(getattr(cfg, "image_width", 1408) or 1408)
        self.height = int(getattr(cfg, "image_height", 1408) or 1408)

    @lru_cache(maxsize=64)
    def rgb_at_device_time_us(self, device_time_us: int, max_edge: int = 512) -> np.ndarray:
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

        device_time_ns = int(device_time_us) * 1000
        image_data, _record = self._provider.get_image_data_by_time_ns(
            self._stream_id,
            device_time_ns,
            TimeDomain.DEVICE_TIME,
            TimeQueryOptions.CLOSEST,
        )
        rgb = image_data.to_numpy_array()
        if max_edge > 0:
            h, w = rgb.shape[:2]
            edge = max(h, w)
            if edge > max_edge:
                scale = max_edge / float(edge)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                try:
                    import cv2

                    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                except ImportError:
                    step_y = max(1, h // new_h)
                    step_x = max(1, w // new_w)
                    rgb = rgb[::step_y, ::step_x]
        return rgb


def recording_paths(sequence_dir: Path, device_key: str) -> dict[str, Path]:
    if device_key not in RECORDING_DEVICES:
        raise KeyError(
            f"Unknown device {device_key!r}. Expected one of {sorted(RECORDING_DEVICES)}"
        )
    subdir = RECORDING_DEVICES[device_key]
    root = sequence_dir / subdir
    return {
        "recording": root,
        "data_vrs": root / "data" / "data.vrs",
        "motion_vrs": root / "data" / "motion.vrs",
        "closed_loop_csv": root / "mps" / "slam" / "closed_loop_trajectory.csv",
        "calibration_jsonl": root / "mps" / "slam" / "online_calibration.jsonl",
        "semidense_points": root / "mps" / "slam" / "semidense_points.csv.gz",
        "semidense_observations": root / "mps" / "slam" / "semidense_observations.csv.gz",
    }
