"""Load Nymeria Plus SMPL-X body motion (same layout as visualization_code)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MOTION_KEYS = (
    "body_pose",
    "global_orient",
    "transl",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "expression",
    "timestamps",
)


def load_smplx_motion(npz_path: Path, stride: int = 1) -> dict[str, np.ndarray]:
    """Load ``xdata_smplx_neutral.npz`` with optional temporal subsampling."""
    data = np.load(npz_path)
    out: dict[str, np.ndarray] = {}
    for key in MOTION_KEYS:
        if key == "timestamps":
            continue
        out[key] = data[key].astype(np.float32)
        if stride > 1:
            out[key] = out[key][::stride]

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


def slice_motion(motion: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    """Return SMPL-X arrays for inclusive frame range ``[start, end]``."""
    start = int(start)
    end = int(end)
    out: dict[str, np.ndarray] = {}
    for key in MOTION_KEYS:
        if key == "timestamps":
            out[key] = motion[key][start : end + 1].astype(np.int64)
        else:
            out[key] = motion[key][start : end + 1].astype(np.float32)
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


def save_motion_npz(motion_slice: dict[str, np.ndarray], path: Path) -> None:
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
