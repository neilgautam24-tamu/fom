"""Fast batched hand-object contact queries against Boxy OBBs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from fom.preprocessing.nymeria.body.smplx_hands import NUM_HAND_JOINT_SAMPLES
from fom.preprocessing.nymeria.contacts.detection import ContactEvent
from fom.preprocessing.nymeria.contacts.geometry import closest_point_on_obb, points_to_obb_distance


@dataclass
class ObjectOBBArray:
    object_uid: np.ndarray
    category: list[str]
    instance_name: list[str]
    center_world: np.ndarray
    dimensions: np.ndarray
    wxyz: np.ndarray
    rot_inv: Rotation


def build_object_array(objects, *, bbox_shrink: float = 0.0) -> ObjectOBBArray:
    n = len(objects)
    centers = np.empty((n, 3), dtype=np.float64)
    dims = np.empty((n, 3), dtype=np.float64)
    wxyz = np.empty((n, 4), dtype=np.float64)
    uids = np.empty(n, dtype=np.int32)
    categories: list[str] = []
    names: list[str] = []

    for i, obj in enumerate(objects):
        uids[i] = int(obj.object_uid)
        categories.append(str(obj.category))
        names.append(str(obj.instance_name))
        centers[i] = obj.center_world
        d = np.asarray(obj.dimensions, dtype=np.float64)
        if bbox_shrink > 0:
            d = np.maximum(d - 2.0 * bbox_shrink, 0.05)
        dims[i] = d
        wxyz[i] = obj.wxyz

    rot_inv = Rotation.from_quat(wxyz, scalar_first=True).inv()
    return ObjectOBBArray(
        object_uid=uids,
        category=categories,
        instance_name=names,
        center_world=centers,
        dimensions=dims,
        wxyz=wxyz,
        rot_inv=rot_inv,
    )


def _nearby_object_mask(points: np.ndarray, obbs: ObjectOBBArray, margin_m: float) -> np.ndarray:
    """Return boolean mask of objects that may contact ``points``."""
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    hand_radius = float(np.linalg.norm(pts - center, axis=1).max()) if len(pts) else 0.0
    reach = hand_radius + margin_m + 0.25
    dist = np.linalg.norm(obbs.center_world - center, axis=1)
    obj_radius = np.linalg.norm(obbs.dimensions, axis=1) * 0.5
    return dist <= (reach + obj_radius)


def detect_contacts_fast(
    hand_points: dict[str, np.ndarray],
    timestamps_ns: np.ndarray,
    obbs: ObjectOBBArray,
    *,
    contact_threshold_m: float = 0.03,
    progress_every: int = 2000,
    joints_only: bool = True,
) -> list[ContactEvent]:
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    t0 = int(ts[0]) if ts.size else 0
    rel_s = (ts - t0) * 1e-6
    threshold = float(contact_threshold_m)

    events: list[ContactEvent] = []
    n_frames = int(next(iter(hand_points.values())).shape[0])

    for frame_idx in range(n_frames):
        if progress_every > 0 and frame_idx > 0 and frame_idx % progress_every == 0:
            print(f"[contact]   frame {frame_idx:,}/{n_frames:,}", flush=True)

        for hand_name, all_pts in hand_points.items():
            pts = all_pts[frame_idx]
            if joints_only:
                pts = pts[:NUM_HAND_JOINT_SAMPLES]
            mask = _nearby_object_mask(pts, obbs, threshold)
            if not np.any(mask):
                continue

            idxs = np.flatnonzero(mask)
            for oi in idxs:
                dists = points_to_obb_distance(
                    pts,
                    obbs.center_world[oi],
                    obbs.dimensions[oi],
                    obbs.wxyz[oi],
                )
                best_idx = int(np.argmin(dists))
                best_dist = float(dists[best_idx])
                if best_dist > threshold:
                    continue

                hand_pt = pts[best_idx].astype(np.float64)
                obj_pt = closest_point_on_obb(
                    hand_pt,
                    obbs.center_world[oi],
                    obbs.dimensions[oi],
                    obbs.wxyz[oi],
                )
                events.append(
                    ContactEvent(
                        frame_idx=frame_idx,
                        timestamp_ns=int(ts[frame_idx]),
                        time_s=float(rel_s[frame_idx]),
                        hand=hand_name,
                        object_uid=int(obbs.object_uid[oi]),
                        category=obbs.category[oi],
                        instance_name=obbs.instance_name[oi],
                        distance_m=best_dist,
                        hand_point_x=float(hand_pt[0]),
                        hand_point_y=float(hand_pt[1]),
                        hand_point_z=float(hand_pt[2]),
                        object_point_x=float(obj_pt[0]),
                        object_point_y=float(obj_pt[1]),
                        object_point_z=float(obj_pt[2]),
                        sample_idx=best_idx,
                    )
                )

    events.sort(key=lambda e: (e.timestamp_ns, e.hand, e.object_uid))
    return events
