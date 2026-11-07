"""SMPL model loading helpers (chumpy cleanup + motion I/O)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from fom.preprocessing.nymeria.io.paths import smpl_clean_dir, smpl_model_dir


class _Surrogate:
    pass


_surrogate_cache: dict[tuple[str, str], type] = {}


def _make_surrogate(module: str, name: str) -> type:
    key = (module, name)
    if key not in _surrogate_cache:
        cls = type(f"S_{name}", (_Surrogate,), {})
        cls.__module__ = module
        _surrogate_cache[key] = cls
    return _surrogate_cache[key]


class _ChumpyTolerantUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("chumpy") or module.startswith("scipy.sparse"):
            return _make_surrogate(module, name)
        return super().find_class(module, name)


def _resolve(obj):
    from scipy.sparse import csc_matrix

    if isinstance(obj, _Surrogate):
        mod = type(obj).__module__
        state = obj.__dict__
        if mod.startswith("chumpy"):
            return np.asarray(_resolve(state["x"]))
        if mod.startswith("scipy.sparse"):
            data = np.asarray(state["data"])
            indices = np.asarray(state["indices"])
            indptr = np.asarray(state["indptr"])
            shape = state.get("_shape") or state.get("shape")
            return csc_matrix((data, indices, indptr), shape=shape)
        raise TypeError(f"Unhandled surrogate from module {mod!r}")
    if isinstance(obj, dict):
        return {k: _resolve(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_resolve(v) for v in obj)
    return obj


def ensure_clean_smpl_model(
    model_dir: Path | None = None,
    clean_dir: Path | None = None,
    gender: str = "neutral",
) -> Path:
    """Return a folder ``smplx.create`` accepts for ``model_type='smpl'``."""
    model_dir = model_dir or smpl_model_dir()
    clean_dir = clean_dir or smpl_clean_dir()
    gender_uc = gender.upper()
    src = model_dir / f"SMPL_{gender_uc}.pkl"
    if not src.is_file():
        raise FileNotFoundError(f"SMPL model not found: {src}")

    out_subdir = clean_dir / "smpl"
    out_subdir.mkdir(parents=True, exist_ok=True)
    dst = out_subdir / f"SMPL_{gender_uc}.pkl"
    if not dst.is_file() or dst.stat().st_mtime < src.stat().st_mtime:
        print(f"[smpl] converting {src.name} -> {dst}")
        with open(src, "rb") as f:
            raw = _ChumpyTolerantUnpickler(f, encoding="latin1").load()
        with open(dst, "wb") as f:
            pickle.dump(_resolve(raw), f, protocol=pickle.HIGHEST_PROTOCOL)
    return clean_dir


def load_smpl_motion(npz_path: Path, stride: int = 1) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    betas = data["betas"].astype(np.float32)
    body_pose = data["body_pose"].astype(np.float32)
    global_orient = data["global_orient"].astype(np.float32)
    transl = data["transl"].astype(np.float32)
    timestamps = data["timestamps"].astype(np.int64)

    if stride > 1:
        sl = slice(None, None, stride)
        betas, body_pose = betas[sl], body_pose[sl]
        global_orient, transl = global_orient[sl], transl[sl]
        timestamps = timestamps[sl]

    return {
        "betas": betas,
        "body_pose": body_pose,
        "global_orient": global_orient,
        "transl": transl,
        "timestamps": timestamps,
        "num_frames": int(betas.shape[0]),
    }


def smpl_body_pose_to_smplx(body_pose_smpl: np.ndarray) -> np.ndarray:
    """Map SMPL (23 joints) body_pose to SMPL-X body layout (21 joints).

    Both models store body_pose without the pelvis (root is ``global_orient``).
    Nymeria SMPL joints 0..20 are left_hip .. right_wrist; joints 21..22 are
    the rigid hand tips and are dropped for SMPL-X body_pose.
    """
    flat = np.asarray(body_pose_smpl, dtype=np.float32)
    if flat.ndim == 1:
        flat = flat.reshape(1, -1)
    return flat.reshape(flat.shape[0], 23, 3)[:, :21].reshape(flat.shape[0], -1)
