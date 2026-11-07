"""Lightning module wrapping :class:`ForesightToMotion` for the three training stages."""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from loguru import logger
from torch.optim.lr_scheduler import StepLR
from yacs.config import CfgNode

from fom.body.smplx_model import load_smplx
from fom.config import config_to_dict
from fom.data.normalization import Normalizer
from fom.models.fom import ForesightToMotion
from fom.training.losses import headcam_loss, weighted_motion_loss


class FoMModule(pl.LightningModule):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.save_hyperparameters(config_to_dict(cfg))
        self.cfg = cfg
        self.window = cfg.DATA.WINDOW
        self.model = ForesightToMotion(cfg)  # attribute name is part of the checkpoint format

        self.use_headcam_loss = cfg.TRAIN.HEADCAM_LOSS_WEIGHT > 0
        # Not registered as submodules: they must not end up in checkpoints.
        object.__setattr__(self, "_body_model", None)
        object.__setattr__(self, "_normalizer", None)
        if self.use_headcam_loss:
            logger.info(
                f"Head-camera loss enabled (weight={cfg.TRAIN.HEADCAM_LOSS_WEIGHT}, "
                f"trajectory dropout={self.model.mask_prob['traj']})"
            )

    @property
    def body_model(self):
        if self._body_model is None:
            object.__setattr__(self, "_body_model", load_smplx().to(self.device).eval())
        return self._body_model

    @property
    def normalizer(self) -> Normalizer:
        if self._normalizer is None:
            object.__setattr__(self, "_normalizer", Normalizer.from_config(self.cfg))
        return self._normalizer

    def configure_optimizers(self):
        # Fused AdamW breaks gradient clipping under AMP, so only use it in full fp32.
        fused = str(self.cfg.TRAIN.PRECISION).lower() in ("32", "32-true", "fp32")
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
            fused=fused,
        )
        if self.cfg.TRAIN.LR_STEP_EPOCHS <= 0:
            return optimizer
        scheduler = StepLR(optimizer, self.cfg.TRAIN.LR_STEP_EPOCHS, gamma=self.cfg.TRAIN.LR_STEP_GAMMA)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def _step(self, batch: dict, mode: str) -> torch.Tensor:
        target = batch["misc"]["motion"]
        y = batch["y"]
        batch_size, num_frames = target.shape[:2]

        traj_mask = None
        if self.use_headcam_loss:
            # Draw the trajectory dropout here so the loss can skip frames whose camera was hidden.
            y = dict(y)
            if mode == "train":
                drop = torch.rand(batch_size, device=target.device) < self.model.mask_prob["traj"]
                traj_mask = drop[:, None].expand(batch_size, num_frames).float()
            else:
                traj_mask = torch.zeros(batch_size, num_frames, device=target.device)
            y["traj_mask"] = traj_mask

        forcing = self.cfg.TRAIN.FORCING if mode == "train" else "teacher"
        pred = self.model.forward_train(target.clone(), y, forcing=forcing)

        log = dict(on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        loss = weighted_motion_loss(
            pred,
            target,
            y["valid_frames"],
            self.cfg.TRAIN.GLOBAL_ORIENT_WEIGHT,
            self.cfg.TRAIN.GLOBAL_TRANS_WEIGHT,
        )
        self.log(f"{mode}/loss_base", loss, **log)

        if self.use_headcam_loss:
            # Body-model decoding needs matrix inverses, which are unsupported in bf16/fp16.
            with torch.autocast(device_type=self.device.type, enabled=False):
                hc_loss, hc_dist = headcam_loss(
                    self.normalizer.denormalize(pred.float(), "motion"),
                    self.normalizer.denormalize(y["traj"].float(), "traj"),
                    y["valid_frames"].float(),
                    traj_mask,
                    self.body_model,
                )
            loss = loss + self.cfg.TRAIN.HEADCAM_LOSS_WEIGHT * hc_loss
            self.log(f"{mode}/loss_headcam", hc_loss, **log)
            self.log(f"{mode}/headcam_dist_m", hc_dist, **log)

        self.log(f"{mode}/loss", loss, **log)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    @torch.no_grad()
    def sample(self, y: dict, batch_size: int) -> torch.Tensor:
        """Roll out as many frames as the longest valid sequence in the batch."""
        if "valid_frames" in y:
            num_frames = int(y["valid_frames"].sum(dim=-1).max().item())
        elif "traj" in y:
            num_frames = y["traj"].shape[1]
        else:
            num_frames = self.window
        return self.model.sample(y, batch_size, num_frames)


def load_checkpoint(path: str | Path, cfg: CfgNode, map_location: str | torch.device = "cpu") -> FoMModule:
    """Build a :class:`FoMModule` from ``cfg`` and load model weights from a Lightning checkpoint.

    Only ``model.*`` weights are loaded; anything else in the checkpoint (optimizer, EMA state,
    extra buffers) is ignored. Missing model weights raise an error.
    """
    module = FoMModule(cfg)
    state = torch.load(str(path), map_location=map_location, weights_only=False)["state_dict"]
    model_state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = module.model.load_state_dict(model_state, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint {path} is missing model weights: {missing}")
    if unexpected:
        logger.warning(f"Ignoring unexpected checkpoint entries: {unexpected}")
    return module
