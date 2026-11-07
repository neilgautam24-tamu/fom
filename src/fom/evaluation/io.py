"""Reading / writing prediction and ground-truth files.

Files map ``sequence_name -> {"smpl_params": {...}, "camera_T": (T, 4, 4)}`` and are written with
joblib. This module has no heavy dependencies so that tools running in other environments (e.g.
the ViPE environment) can read them; it falls back to ``pickle`` when joblib is unavailable.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

try:
    import joblib
except ImportError:  # pragma: no cover - only in minimal environments
    joblib = None


def sequence_key(key: str) -> str:
    """Canonical sequence name (strips a ``_start_<frame>`` suffix from keys)."""
    return re.sub(r"_start_\d+$", "", key)


def _load(path: str | Path):
    if joblib is not None:
        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_sequences(path: str | Path) -> dict[str, dict]:
    """Load a prediction / ground-truth file (also accepts the ``aria_traj_T`` field name)."""
    out = {}
    for key, entry in _load(path).items():
        camera_T = entry["camera_T"] if "camera_T" in entry else entry.get("aria_traj_T")
        out[sequence_key(key)] = {"smpl_params": entry["smpl_params"], "camera_T": camera_T}
    return out


def save(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(obj, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
