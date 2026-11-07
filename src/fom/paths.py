"""Filesystem locations that differ between machines.

Everything machine-specific is read from environment variables so that code never
hard-codes cluster paths:

=====================  ==========================================================
Variable               Meaning
=====================  ==========================================================
``FOM_BODY_MODELS``    Directory that contains ``smplx/SMPLX_NEUTRAL.npz`` etc.
``FOM_SMPL_MODELS``    Directory with SMPL (not -X) models; only used by the
                       Nymeria SMPL -> SMPL-X refit during preprocessing.
``FOM_VIPE_ROOT``      Checkout of NVIDIA ViPE (depth backend + video poses).
``FOM_TORCH_HUB``      Optional torch.hub cache (DINOv2 / MiDaS weights).
``FOM_CACHE``          Optional cache for derived files (default ``~/.cache/fom``).
=====================  ==========================================================
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Export it (see fom/paths.py for what it should point to)."
        )
    return Path(value).expanduser()


def body_models_dir() -> Path:
    """Root of the SMPL-X body model files (``<dir>/smplx/SMPLX_NEUTRAL.*``)."""
    return _env_path("FOM_BODY_MODELS")


def smpl_models_dir() -> Path:
    """Root of the SMPL body model files used by the SMPL -> SMPL-X refit."""
    return _env_path("FOM_SMPL_MODELS")


def vipe_root() -> Path:
    """Checkout of NVIDIA ViPE."""
    return _env_path("FOM_VIPE_ROOT")


def torch_hub_dir() -> Path | None:
    value = os.environ.get("FOM_TORCH_HUB")
    return Path(value).expanduser() if value else None


def cache_dir() -> Path:
    """Cache for derived artefacts (SMPL->SMPL-X correspondences, cleaned SMPL models, ...)."""
    path = Path(os.environ.get("FOM_CACHE", "~/.cache/fom")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path
