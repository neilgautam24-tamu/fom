"""Load a trained checkpoint and turn conditioning into decoded SMPL-X motion."""

from __future__ import annotations

from pathlib import Path

import torch

from fom.body.representation import decode_sequence
from fom.body.smplx_model import load_smplx
from fom.config import load_config
from fom.data.normalization import Normalizer
from fom.training.module import load_checkpoint


class MotionPredictor:
    """Wraps model, normalization statistics and the body model of one checkpoint.

    Args:
        config: the YAML the checkpoint was trained with; ``overrides`` must at least set
            ``DATA.ROOT`` or ``DATA.STATS_PATH`` so the training statistics can be found.
    """

    def __init__(
        self,
        config: str | Path,
        ckpt: str | Path,
        overrides: list[str] | tuple[str, ...] = (),
        device: str | torch.device = "cuda",
    ):
        self.cfg = load_config(config, overrides)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.module = load_checkpoint(ckpt, self.cfg).to(self.device).eval()
        self.normalizer = Normalizer.from_config(self.cfg)
        self.body_model = load_smplx()
        self.window = self.cfg.DATA.WINDOW

    def normalize_traj(self, traj: torch.Tensor) -> torch.Tensor:
        return self.normalizer.normalize(traj, "traj")

    @torch.inference_mode()
    def predict(self, y: dict[str, torch.Tensor]) -> torch.Tensor:
        """Un-batched conditioning -> normalized motion for the valid frames ``(T, 243)``."""
        batch = {k: v[None].to(self.device) for k, v in y.items()}
        motion = self.module.sample(batch, batch_size=1)[0]
        return motion[: int(y["valid_frames"].sum())].cpu()

    @torch.no_grad()
    def decode(self, motion: torch.Tensor, traj: torch.Tensor) -> dict:
        """Normalized motion + *un-normalized* trajectory features -> camera poses and SMPL-X params."""
        camera_T, params, joints = decode_sequence(
            self.normalizer.denormalize(motion.float(), "motion"), traj[: motion.shape[0]].float(), self.body_model
        )
        return {"camera_T": camera_T, "smpl_params": params, "body_joints": joints}
