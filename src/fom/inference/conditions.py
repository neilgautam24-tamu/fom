"""Visual conditioning (DINOv2, depth patches, goal heatmap) for arbitrary egocentric video."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from fom.models.fom import GOAL_HEATMAP_DIM
from fom.preprocessing.features.common import N_PATCHES, depth_to_patches

FEAT_SIZE = 512  # square RGB size the training frames were rectified to
GOAL_SIGMA = 1.5

__all__ = ["Dinov2Extractor", "center_goal_heatmap", "depth_to_patches", "make_window_conditions", "resize_square"]


def resize_square(img: np.ndarray, size: int = FEAT_SIZE) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


class Dinov2Extractor:
    """DINOv2 ViT-L/14 (+registers) CLS features, same preprocessing as training."""

    def __init__(self, device: str | torch.device = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def cls_features(self, frames: list[np.ndarray], batch_size: int = 16) -> torch.Tensor:
        """``N`` HxWx3 uint8 frames -> ``(N, 1024)`` CLS features."""
        out = []
        for i in range(0, len(frames), batch_size):
            batch = torch.stack([self.transform(Image.fromarray(f)) for f in frames[i : i + batch_size]])
            out.append(self.model.forward_features(batch.to(self.device))["x_norm_clstoken"].float().cpu())
        return torch.cat(out)


def center_goal_heatmap() -> torch.Tensor:
    """Goal prior at the image centre, used when no interaction target is given.

    Note: peak-normalized (max 1), unlike the sum-normalized training heatmaps.
    """
    center = (N_PATCHES - 1) / 2.0
    ys, xs = np.mgrid[0:N_PATCHES, 0:N_PATCHES].astype(np.float32)
    heat = np.exp(-((xs - center) ** 2 + (ys - center) ** 2) / (2 * GOAL_SIGMA**2))
    return torch.from_numpy((heat / (heat.max() + 1e-8)).reshape(GOAL_HEATMAP_DIM))


def make_window_conditions(
    traj: torch.Tensor,
    rgb_cls: torch.Tensor,
    depth: torch.Tensor,
    goal: torch.Tensor,
    window: int,
) -> dict[str, torch.Tensor]:
    """Assemble an un-batched conditioning dict, truncated / zero-padded to ``window`` frames.

    Args:
        traj: ``(T, 18)`` normalized trajectory.
        rgb_cls: ``(T, 1024)``; depth: ``(T, 5, 1024)``; goal: ``(T, 256)`` (already held to the
            visual staircase).
    """
    length = min(traj.shape[0], window)
    valid = torch.zeros(window, dtype=torch.long)
    valid[:length] = 1

    def pad(x: torch.Tensor) -> torch.Tensor:
        x = x[:length]
        return torch.cat([x, torch.zeros(window - length, *x.shape[1:], dtype=x.dtype)]) if length < window else x

    return {
        "traj": pad(traj.float()),
        "rgb": pad(rgb_cls.float()),
        "rgb_valid": valid.clone(),
        "depth": pad(depth.float()),
        "depth_valid": valid.clone(),
        "goal": pad(goal.float()),
        "valid_frames": valid,
    }
