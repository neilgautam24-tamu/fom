"""Helpers shared by the viewers and figure renderers.

Data locations are read from environment variables:

``FOM_BUNDLE_ROOTS``  ``:``-separated directories of processed Nymeria episodes, each holding
                      ``<episode>/visible_goal_sequence_bundles`` (or the multi-object variant).
``FOM_NYMERIA_PLUS``  Raw Nymeria-Plus downloads (``<episode>/recording_head/mps/...``).
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from fom.paths import cache_dir

BUNDLE_DIR_NAMES = ("visible_goal_sequence_bundles", "multi_object_visible_goal_bundles")
_BUNDLE_MARKERS = ("_vgoal_seq_", "_mvgoal_seq_")


def bundle_roots() -> list[Path]:
    value = os.environ.get("FOM_BUNDLE_ROOTS", "")
    return [Path(p).expanduser() for p in value.split(":") if p]


def nymeria_plus_root() -> Path:
    value = os.environ.get("FOM_NYMERIA_PLUS")
    if not value:
        raise RuntimeError("Set FOM_NYMERIA_PLUS to the raw Nymeria-Plus download directory.")
    return Path(value).expanduser()


def viz_cache_dir(name: str) -> Path:
    path = cache_dir() / "viz" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_sequence_name(name: str) -> tuple[str, str] | None:
    """``<episode>_vgoal_seq_XXX[___a___b]`` -> ``(episode, bundle_dir_name)``."""
    take = name.split("___")[0]
    for marker in _BUNDLE_MARKERS:
        if marker in take:
            episode, rest = take.split(marker, 1)
            return episode, marker[1:] + rest
    return None


def find_bundle(name: str, roots: list[Path] | None = None, marker_file: str = "ego_rgb.npz") -> Path | None:
    """Locate the exported bundle directory of a packed sequence name."""
    parsed = parse_sequence_name(name)
    if parsed is None:
        return None
    episode, bundle = parsed
    for root in roots if roots is not None else bundle_roots():
        for dir_name in BUNDLE_DIR_NAMES:
            candidate = root / episode / dir_name / bundle
            if (candidate / marker_file).is_file():
                return candidate
    return None


def host_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 1))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rotmat_to_wxyz(rot: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(rot).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def resize_max_edge(rgb: np.ndarray, max_edge: int) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    if max(h, w) <= max_edge:
        return rgb
    scale = max_edge / float(max(h, w))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR))


def rot180(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(np.asarray(rgb, dtype=np.uint8), 2))


def voxel_downsample(
    xyz: np.ndarray, rgb: np.ndarray, voxel: float, max_points: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] == 0:
        return xyz.astype(np.float32), rgb.astype(np.uint8)
    keys = np.floor(xyz / max(voxel, 1e-4)).astype(np.int32)
    keys = keys - keys.min(axis=0, keepdims=True)
    span = keys.max(axis=0) + 1
    if int(np.prod(span)) > 50_000_000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(xyz.shape[0], min(max_points * 4, xyz.shape[0]), replace=False)
        xyz, rgb, keys = xyz[idx], rgb[idx], keys[idx]
        keys = keys - keys.min(axis=0, keepdims=True)
        span = keys.max(axis=0) + 1
    flat = keys[:, 0] + span[0] * (keys[:, 1] + span[1] * keys[:, 2])
    _, uniq = np.unique(flat, return_index=True)
    xyz, rgb = xyz[uniq], rgb[uniq]
    if xyz.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(xyz.shape[0], max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz.astype(np.float32), rgb.astype(np.uint8)
