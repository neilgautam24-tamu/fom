"""GPU depth backends for Nymeria RGB frames."""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch


class DepthEstimator(ABC):
    @abstractmethod
    def predict(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """Return (H,W) float32 metric depth in metres."""

    def predict_batch(self, rgb_uint8_batch: np.ndarray) -> np.ndarray:
        """Return (B,H,W) float32 depth. Default: loop ``predict``."""
        if rgb_uint8_batch.ndim != 4:
            raise ValueError(
                f"Expected (B,H,W,3) uint8 batch, got shape {rgb_uint8_batch.shape}"
            )
        depths = [self.predict(rgb_uint8_batch[i]) for i in range(rgb_uint8_batch.shape[0])]
        return np.stack(depths, axis=0)


class VipeUniDepthEstimator(DepthEstimator):
    """UniDepth via the ViPE package (same backend as the ADT depth extraction)."""

    def __init__(self, vipe_path: Path, model_name: str, device: torch.device) -> None:
        vipe_path = vipe_path.resolve()
        if str(vipe_path) not in sys.path:
            sys.path.insert(0, str(vipe_path))
        from vipe.priors.depth import make_depth_model
        from vipe.priors.depth.base import DepthEstimationInput
        from vipe.priors.depth.unidepth.utils.camera import BatchCamera, Pinhole
        from vipe.utils.cameras import CameraType

        self._input_cls = DepthEstimationInput
        self._camera_type = CameraType.PINHOLE
        self._model = make_depth_model(model_name)
        if hasattr(self._model, "to"):
            self._model.to(device)
        self._device = device
        self._intrinsics = torch.tensor(
            [280.0, 280.0, 256.0, 256.0], device=device, dtype=torch.float32
        )
        # UniDepthV2.infer always wraps cameras with BatchCamera.from_camera, which
        # collapses a Pinhole with K shape (B,3,3) into len(cameras)==1. Expand to
        # B cameras so true batched RGB forward works.
        if not getattr(BatchCamera, "_nymeria_batch_from_camera_patch", False):
            _orig_from_camera = BatchCamera.from_camera

            @classmethod
            def _from_camera_batched(cls, camera):
                if isinstance(camera, BatchCamera):
                    return camera
                if (
                    isinstance(camera, Pinhole)
                    and isinstance(camera.K, torch.Tensor)
                    and camera.K.ndim == 3
                    and camera.K.shape[0] > 1
                ):
                    cams = [Pinhole(K=camera.K[i]) for i in range(camera.K.shape[0])]
                    return BatchCamera(
                        torch.stack([c.params for c in cams], dim=0),
                        torch.stack([c.K for c in cams], dim=0),
                        ["Pinhole"] * len(cams),
                        cams,
                    )
                return _orig_from_camera(camera)

            BatchCamera.from_camera = _from_camera_batched
            BatchCamera._nymeria_batch_from_camera_patch = True

    @torch.no_grad()
    def predict(self, rgb_uint8: np.ndarray) -> np.ndarray:
        return self.predict_batch(rgb_uint8[None])[0]

    @torch.no_grad()
    def predict_batch(self, rgb_uint8_batch: np.ndarray) -> np.ndarray:
        if rgb_uint8_batch.ndim != 4:
            raise ValueError(
                f"Expected (B,H,W,3) uint8 batch, got shape {rgb_uint8_batch.shape}"
            )
        rgb_f32 = (
            torch.from_numpy(np.ascontiguousarray(rgb_uint8_batch))
            .float()
            .div(255.0)
            .to(self._device)
        )
        # Per-batch principal point from actual resolution (ViPE uses intrinsics[0]=fx).
        _, height, width, _ = rgb_f32.shape
        intrinsics = torch.tensor(
            [float(self._intrinsics[0]), float(self._intrinsics[1]), width / 2.0, height / 2.0],
            device=self._device,
            dtype=torch.float32,
        )
        result = self._model.estimate(
            self._input_cls(
                rgb=rgb_f32,
                intrinsics=intrinsics,
                camera_type=self._camera_type,
            )
        )
        depth = result.metric_depth.detach().cpu().numpy().astype(np.float32)
        if depth.ndim == 2:
            depth = depth[None]
        return depth

class MiDaSDepthEstimator(DepthEstimator):
    """MiDaS DPT depth via torch.hub (relative depth; patch layout matches ADT)."""

    def __init__(self, device: torch.device, hub_dir: Path | None = None) -> None:
        if hub_dir is not None:
            hub_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(hub_dir))

        self._device = device
        self._model = torch.hub.load(
            "intel-isl/MiDaS", "DPT_Large", trust_repo=True
        ).to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self._transform = transforms.dpt_transform

    @torch.no_grad()
    def predict(self, rgb_uint8: np.ndarray) -> np.ndarray:
        return self.predict_batch(rgb_uint8[None])[0]

    @torch.no_grad()
    def predict_batch(self, rgb_uint8_batch: np.ndarray) -> np.ndarray:
        if rgb_uint8_batch.ndim != 4:
            raise ValueError(
                f"Expected (B,H,W,3) uint8 batch, got shape {rgb_uint8_batch.shape}"
            )
        h, w = rgb_uint8_batch.shape[1:3]
        # dpt_transform returns (1,C,H',W') per image; stack to (B,C,H',W').
        inp = torch.cat(
            [self._transform(rgb_uint8_batch[i]) for i in range(rgb_uint8_batch.shape[0])],
            dim=0,
        ).to(self._device)
        pred = self._model(inp)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        elif pred.ndim == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        depth = torch.nn.functional.interpolate(
            pred,
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        return np.maximum(depth_np, 0.0)


class TorchHubUniDepthEstimator(DepthEstimator):
    """UniDepth v2 via torch.hub (lpiccinelli/unidepth)."""

    def __init__(
        self,
        backbone: str,
        device: torch.device,
        hub_dir: Path | None = None,
    ) -> None:
        if hub_dir is not None:
            hub_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(hub_dir))

        self._device = device
        self._model = torch.hub.load(
            "lpiccinelli/unidepth",
            "UniDepth",
            version="v2",
            backbone=backbone,
            pretrained=True,
            trust_repo=True,
            skip_validation=True,
        )
        self._model = self._model.to(device).eval()
        self._intrinsics = torch.tensor(
            [[280.0, 0.0, 256.0], [0.0, 280.0, 256.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def predict(self, rgb_uint8: np.ndarray) -> np.ndarray:
        return self.predict_batch(rgb_uint8[None])[0]

    @torch.no_grad()
    def predict_batch(self, rgb_uint8_batch: np.ndarray) -> np.ndarray:
        if rgb_uint8_batch.ndim != 4:
            raise ValueError(
                f"Expected (B,H,W,3) uint8 batch, got shape {rgb_uint8_batch.shape}"
            )
        rgb_f32 = (
            torch.from_numpy(np.ascontiguousarray(rgb_uint8_batch))
            .float()
            .div(255.0)
            .permute(0, 3, 1, 2)
            .to(self._device)
        )
        intrinsics = self._intrinsics.unsqueeze(0).expand(rgb_f32.shape[0], -1, -1)
        if hasattr(self._model, "infer"):
            out = self._model.infer(rgb_f32, intrinsics)
            depth = out["depth"] if isinstance(out, dict) else out
        else:
            out = self._model(rgb_f32, intrinsics)
            depth = out["depth"] if isinstance(out, dict) else out
        if depth.ndim == 4:
            depth = depth.squeeze(1)
        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        if depth_np.ndim == 2:
            depth_np = depth_np[None]
        return depth_np


def build_depth_estimator(
    *,
    backend: str,
    device: torch.device,
    vipe_path: Path | None,
    depth_model: str,
    hub_dir: Path | None = None,
) -> DepthEstimator:
    if backend == "vipe":
        if vipe_path is None or not vipe_path.is_dir():
            raise FileNotFoundError(
                "ViPE backend requires --vipe-path to a checkout of the vipe repo."
            )
        return VipeUniDepthEstimator(vipe_path, depth_model, device)
    if backend == "midas":
        return MiDaSDepthEstimator(device, hub_dir=hub_dir)
    if backend == "torchhub":
        backbone = {"unidepth-l": "vitl14", "unidepth-b": "vitb14", "unidepth-s": "vits14"}[
            depth_model
        ]
        return TorchHubUniDepthEstimator(backbone, device, hub_dir=hub_dir)
    raise ValueError(f"Unknown depth backend: {backend}")


def default_vipe_path() -> Path | None:
    """ViPE checkout from ``FOM_VIPE_ROOT`` (or ``VIPE_PATH``), if set."""
    env = os.environ.get("FOM_VIPE_ROOT") or os.environ.get("VIPE_PATH")
    return Path(env) if env else None
