"""Z-score normalization of motion and trajectory features with training-set statistics."""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from yacs.config import CfgNode

from fom.data import files

_EPS = 1e-6


class Normalizer:
    """Holds ``{motion,traj}_{mean,std}``; near-zero stds are replaced by 1."""

    def __init__(self, stats: dict[str, torch.Tensor]):
        stats = dict(stats)
        for key in ("traj_std", "motion_std"):
            stats[key] = torch.where(stats[key].abs() < 1e-8, torch.ones_like(stats[key]), stats[key])
        self.stats = stats

    @classmethod
    def from_file(cls, path: str | Path) -> "Normalizer":
        logger.info(f"Loading normalization statistics from {path}")
        return cls(torch.load(str(path), map_location="cpu", weights_only=False))

    @classmethod
    def from_config(cls, cfg: CfgNode) -> "Normalizer":
        return cls.from_file(stats_path(cfg))

    def _get(self, key: str, like: torch.Tensor) -> torch.Tensor:
        return self.stats[key].to(device=like.device, dtype=like.dtype)

    def normalize(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        """``kind`` is ``"motion"`` or ``"traj"``."""
        return (x - self._get(f"{kind}_mean", x)) / (self._get(f"{kind}_std", x) + _EPS)

    def denormalize(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        return x * (self._get(f"{kind}_std", x) + _EPS) + self._get(f"{kind}_mean", x)


def stats_path(cfg: CfgNode) -> Path:
    return Path(cfg.DATA.STATS_PATH) if cfg.DATA.STATS_PATH else files.stats_file(cfg.DATA.ROOT)
