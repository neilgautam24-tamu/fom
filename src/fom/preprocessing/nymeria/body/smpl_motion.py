"""Load and slice Nymeria Plus SMPL body motion (original ``xdata_smpl_neutral.npz``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SMPL_MOTION_KEYS = (
    "betas",
    "body_pose",
    "global_orient",
    "transl",
    "timestamps",
)


def load_smpl_motion(npz_path: Path, stride: int = 1) -> dict[str, np.ndarray]:
    """Load ``xdata_smpl_neutral.npz`` with optional temporal subsampling."""
    data = np.load(npz_path)
    out: dict[str, np.ndarray] = {}
    for key in SMPL_MOTION_KEYS:
        if key in ("betas", "timestamps"):
            continue
        arr = data[key].astype(np.float32)
        if stride > 1:
            arr = arr[::stride]
        out[key] = arr

    betas = data["betas"].astype(np.float32)
    n = out["body_pose"].shape[0]
    if betas.ndim == 1:
        out["betas"] = np.broadcast_to(betas, (n, betas.shape[0])).copy()
    else:
        out["betas"] = betas[::stride] if stride > 1 else betas

    out["timestamps"] = data["timestamps"].astype(np.int64)
    if stride > 1:
        out["timestamps"] = out["timestamps"][::stride]
    out["num_frames"] = n
    return out


def slice_smpl_motion(motion: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    """Return SMPL arrays for inclusive frame range ``[start, end]``."""
    start = int(start)
    end = int(end)
    out: dict[str, np.ndarray] = {}
    for key in SMPL_MOTION_KEYS:
        if key in ("betas", "timestamps"):
            continue
        out[key] = motion[key][start : end + 1].astype(np.float32)
    out["timestamps"] = motion["timestamps"][start : end + 1].astype(np.int64)
    n = end - start + 1
    out["num_frames"] = n
    out["global_start_frame_idx"] = start
    out["global_end_frame_idx"] = end
    if "betas" in motion:
        betas = motion["betas"]
        if betas.ndim == 1:
            out["betas"] = np.broadcast_to(betas.astype(np.float32), (n, betas.shape[0])).copy()
        else:
            out["betas"] = betas[start : end + 1].astype(np.float32)
    return out


def save_smpl_motion_npz(motion_slice: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in motion_slice.items() if isinstance(v, np.ndarray)}
    payload["num_frames"] = np.array([motion_slice["num_frames"]], dtype=np.int32)
    payload["global_start_frame_idx"] = np.array(
        [motion_slice.get("global_start_frame_idx", 0)], dtype=np.int32
    )
    payload["global_end_frame_idx"] = np.array(
        [motion_slice.get("global_end_frame_idx", 0)], dtype=np.int32
    )
    np.savez_compressed(path, **payload)
