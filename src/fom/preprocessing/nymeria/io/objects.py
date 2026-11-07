"""Load Nymeria Plus object 3D bounding boxes from ``objects/boxy``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

# High-contrast, colorblind-friendly palette (Paul Tol bright).
CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "Chair": (0, 114, 178),
    "Couch": (213, 94, 0),
    "Door": (240, 228, 66),
    "Floor": (170, 170, 170),
    "Lamp": (255, 176, 0),
    "Mirror": (102, 204, 238),
    "Plant": (0, 158, 115),
    "Screen": (86, 180, 233),
    "Sink": (0, 92, 230),
    "Stairs": (178, 118, 0),
    "Storage": (204, 121, 167),
    "Table": (230, 159, 0),
    "Wall": (187, 187, 187),
    "WallArt": (213, 62, 79),
    "WasherDryer": (117, 112, 179),
    "Window": (109, 204, 218),
}


_BOX_CORNER_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
_BOX_EDGES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)


def category_color(category: str, object_uid: int) -> tuple[int, int, int]:
    if category in CATEGORY_COLORS:
        return CATEGORY_COLORS[category]
    hue = (object_uid * 47) % 360
    return _hsv_to_rgb(hue / 360.0, 0.85, 0.95)


def darken_rgb(rgb: tuple[int, int, int], factor: float = 0.35) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, round(c * factor)))) for c in rgb)


def mute_point_cloud_colors(
    colors: np.ndarray,
    *,
    saturation: float = 0.35,
    lift: float = 0.78,
) -> np.ndarray:
    """Desaturate and lighten point colors so object boxes read more clearly."""
    c = colors.astype(np.float32) / 255.0
    gray = c.mean(axis=-1, keepdims=True)
    c = gray + float(saturation) * (c - gray)
    c = c * (1.0 - float(lift)) + float(lift)
    return np.clip(c * 255.0, 0.0, 255.0).astype(np.uint8)


def box_corners_world(
    center: np.ndarray,
    dimensions: np.ndarray,
    wxyz: np.ndarray,
) -> np.ndarray:
    half = np.asarray(dimensions, dtype=np.float64) * 0.5
    local = _BOX_CORNER_SIGNS * half
    rot = Rotation.from_quat(np.asarray(wxyz, dtype=np.float64), scalar_first=True)
    return rot.apply(local) + np.asarray(center, dtype=np.float64)


def box_edge_line_segments(obj: ObjectBBox3D) -> np.ndarray:
    """Return line segments with shape ``(12, 2, 3)`` in world coordinates."""
    corners = box_corners_world(obj.center_world, obj.dimensions, obj.wxyz)
    return corners[_BOX_EDGES].astype(np.float32)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


@dataclass(frozen=True)
class ObjectBBox3D:
    object_uid: int
    category: str
    instance_name: str
    motion_type: str
    center_world: np.ndarray
    dimensions: np.ndarray
    wxyz: np.ndarray
    timestamp_ns: int  # static objects use -1


def boxy_dir(sequence_dir: Path) -> Path:
    return sequence_dir / "objects" / "boxy"


def load_object_bboxes(
    sequence_dir: Path,
    *,
    categories: set[str] | None = None,
    skip_floor_wall: bool = True,
) -> list[ObjectBBox3D]:
    """Merge ``3dbb.csv``, ``scene_objects.csv``, and ``instances.json``."""
    root = boxy_dir(sequence_dir)
    boxes_path = root / "3dbb.csv"
    poses_path = root / "scene_objects.csv"
    instances_path = root / "instances.json"
    for path in (boxes_path, poses_path, instances_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing object annotation file: {path}")

    boxes = pd.read_csv(boxes_path)
    poses = pd.read_csv(poses_path)
    instances: dict[str, dict] = json.loads(instances_path.read_text())

    pose_by_uid: dict[int, pd.DataFrame] = {}
    for uid, group in poses.groupby("object_uid"):
        pose_by_uid[int(uid)] = group.sort_values("timestamp[ns]")

    out: list[ObjectBBox3D] = []
    for _, row in boxes.iterrows():
        uid = int(row["object_uid"])
        meta = instances.get(str(uid), {})
        category = str(meta.get("category", "Unknown"))
        if categories is not None and category not in categories:
            continue
        if skip_floor_wall and category in {"Floor", "Wall"}:
            continue

        xmin = float(row["p_local_obj_xmin[m]"])
        xmax = float(row["p_local_obj_xmax[m]"])
        ymin = float(row["p_local_obj_ymin[m]"])
        ymax = float(row["p_local_obj_ymax[m]"])
        zmin = float(row["p_local_obj_zmin[m]"])
        zmax = float(row["p_local_obj_zmax[m]"])
        center_local = np.array(
            [(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5],
            dtype=np.float64,
        )
        dimensions = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.float64)

        pose_rows = pose_by_uid.get(uid)
        if pose_rows is None or pose_rows.empty:
            continue
        query_ns = int(row["timestamp[ns]"])
        t_wo, wxyz = pose_at_timestamp_ns(pose_rows, query_ns)
        rot = Rotation.from_quat(wxyz, scalar_first=True)
        center_world = rot.apply(center_local) + t_wo

        out.append(
            ObjectBBox3D(
                object_uid=uid,
                category=category,
                instance_name=str(meta.get("instance_name", category)),
                motion_type=str(meta.get("motion_type", "static")),
                center_world=center_world.astype(np.float32),
                dimensions=dimensions.astype(np.float32),
                wxyz=wxyz.astype(np.float32),
                timestamp_ns=query_ns,
            )
        )
    out.sort(key=lambda o: (o.category, o.object_uid))
    return out


def object_bbox_by_uid(sequence_dir: Path, object_uid: int) -> ObjectBBox3D | None:
    """Return one Boxy OBB from the Nymeria Plus sequence directory."""
    for obj in load_object_bboxes(sequence_dir, skip_floor_wall=True):
        if int(obj.object_uid) == int(object_uid):
            return obj
    return None


def pose_at_timestamp_ns(
    pose_rows: pd.DataFrame,
    query_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t_wo, wxyz)`` for a query timestamp (linear pos + slerp rot)."""
    ts = pose_rows["timestamp[ns]"].to_numpy(dtype=np.int64)
    if len(ts) == 1 or np.all(ts == -1):
        row = pose_rows.iloc[0]
    else:
        valid = ts[ts >= 0]
        if len(valid) == 0:
            row = pose_rows.iloc[0]
        elif query_ns <= valid[0]:
            row = pose_rows.loc[pose_rows["timestamp[ns]"] == valid[0]].iloc[0]
        elif query_ns >= valid[-1]:
            row = pose_rows.loc[pose_rows["timestamp[ns]"] == valid[-1]].iloc[0]
        else:
            idx = int(np.searchsorted(valid, query_ns, side="right") - 1)
            t0, t1 = int(valid[idx]), int(valid[idx + 1])
            r0 = pose_rows.loc[pose_rows["timestamp[ns]"] == t0].iloc[0]
            r1 = pose_rows.loc[pose_rows["timestamp[ns]"] == t1].iloc[0]
            alpha = (query_ns - t0) / max(t1 - t0, 1)
            t_wo = (
                (1.0 - alpha)
                * np.array([r0["t_wo_x[m]"], r0["t_wo_y[m]"], r0["t_wo_z[m]"]])
                + alpha * np.array([r1["t_wo_x[m]"], r1["t_wo_y[m]"], r1["t_wo_z[m]"]])
            )
            rots = Rotation.from_quat(
                [
                    [r0["q_wo_w"], r0["q_wo_x"], r0["q_wo_y"], r0["q_wo_z"]],
                    [r1["q_wo_w"], r1["q_wo_x"], r1["q_wo_y"], r1["q_wo_z"]],
                ],
                scalar_first=True,
            )
            wxyz = Slerp([0.0, 1.0], rots)([alpha]).as_quat(scalar_first=True)[0]
            return t_wo.astype(np.float32), wxyz.astype(np.float32)

    t_wo = np.array(
        [float(row["t_wo_x[m]"]), float(row["t_wo_y[m]"]), float(row["t_wo_z[m]"])],
        dtype=np.float32,
    )
    wxyz = np.array(
        [
            float(row["q_wo_w"]),
            float(row["q_wo_x"]),
            float(row["q_wo_y"]),
            float(row["q_wo_z"]),
        ],
        dtype=np.float32,
    )
    return t_wo, wxyz
