"""File-name conventions of Nymeria(-Plus) episodes and body-model lookup."""

from __future__ import annotations

from pathlib import Path

from fom.paths import body_models_dir, cache_dir, smpl_models_dir

SMPL_NPZ_CANDIDATES = ("xdata_smpl_neutral.npz",)
SMPLX_NPZ_NAME = "xdata_smplx_neutral.npz"
XDATA_NPZ = "xdata.npz"


def smpl_clean_dir() -> Path:
    """Where SMPL models with chumpy objects stripped are cached."""
    return cache_dir() / "smpl_models_clean"


def correspondence_cache_dir() -> Path:
    return cache_dir() / "smpl2smplx"


def resolve_existing(body_dir: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = body_dir / name
        if path.is_file():
            return path
    return body_dir / candidates[0]


def default_smpl_npz(body_dir: Path) -> Path:
    """SMPL motion of an episode: ``<episode>/body/xdata_smpl_neutral.npz``."""
    return resolve_existing(body_dir, SMPL_NPZ_CANDIDATES)


def default_smplx_npz(body_dir: Path) -> Path:
    """SMPL-X refit written by :mod:`fom.preprocessing.nymeria.body.smpl_to_smplx`."""
    return body_dir / SMPLX_NPZ_NAME


def smplx_model_file(gender: str = "neutral", ext: str = "npz") -> Path:
    """Return an existing SMPL-X weight file (flat or smplx/ layout)."""
    root = body_models_dir()
    gender_uc = gender.upper()
    candidates = [
        root / "smplx" / f"SMPLX_{gender_uc}.{ext}",
        root / f"SMPLX_{gender_uc}.{ext}",
        root / f"SMPLX_{gender_uc}.pkl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No SMPL-X {gender} model under {root}. Expected SMPLX_{gender_uc}.npz")


def smpl_model_dir() -> Path:
    return smpl_models_dir()
