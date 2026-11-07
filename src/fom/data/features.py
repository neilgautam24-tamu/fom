"""Temporal alignment of visual features (stored at ``VISUAL_FPS``) with the motion stream.

Visual features are extracted at a low rate and *held* for ``MOTION_FPS // VISUAL_FPS`` motion
frames (a staircase). RGB, depth and goal all follow this same cadence.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from fom.data import files


def hold_to_motion_rate(feat: torch.Tensor, start: int, num_frames: int, motion_fps: int, visual_fps: int) -> torch.Tensor:
    """Sample a per-motion-frame feature on the visual staircase.

    The value used for motion frame ``m`` is ``feat[(m // u) * u]`` with ``u = motion_fps // visual_fps``.
    Returns exactly ``num_frames`` rows starting at motion frame ``start``.
    """
    factor = max(1, motion_fps // visual_fps)
    idx = torch.arange(start, start + num_frames, dtype=torch.long)
    if factor > 1:
        idx = (idx // factor) * factor
    return feat[torch.clamp(idx, max=feat.shape[0] - 1)]


class RgbFeatures:
    """DINOv2 CLS features keyed by take name, indexed in 30 fps frame space."""

    def __init__(self, root: str | Path, split: str, motion_fps: int, visual_fps: int):
        path = files.rgb_feature_file(root, split)
        logger.info(f"Loading DINOv2 features from {path}")
        self.feats = torch.load(path, weights_only=False)["feats"]
        self.motion_fps = motion_fps
        self.visual_fps = visual_fps

    def get(self, seq_name: str, start_idx: int, end_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Features for 30 fps frames ``[start_idx, end_idx]`` relative to the sequence start.

        Sequence names are ``<take>___<start30>___<end30>``.

        Returns:
            ``(T, 1024)`` features and ``(T,)`` validity at the motion rate.
        """
        take_name, seq_start, seq_end = seq_name.split("___")
        start_idx += int(seq_start)
        end_idx += int(seq_start)
        assert end_idx <= int(seq_end)

        fps_ratio = 30 // self.motion_fps
        factor = self.motion_fps // self.visual_fps
        start_m, end_m = start_idx // fps_ratio, end_idx // fps_ratio
        start_v, end_v = start_m // factor, end_m // factor

        embs = self.feats[take_name][start_v : end_v + 1].clone().float()
        embs = embs[:, None].repeat(1, factor, 1, 1).flatten(0, 1)  # hold each visual frame
        offset = start_m % factor
        if offset:
            embs = embs[offset:]
        target_len = end_m - start_m + 1
        embs = embs[:target_len]
        assert embs.shape[0] == target_len

        return embs[:, 0], torch.ones(embs.shape[0], dtype=torch.long)  # CLS token only
