"""Train one stage of Foresight to Motion.

    fom-train --config configs/train/stage1_teacher.yaml [KEY VALUE ...]

The run resumes automatically from ``<TRAIN.EXP_DIR>/last.ckpt`` if it exists. Otherwise, if
``TRAIN.INIT_CKPT`` is set, the model weights are initialized from it (previous stage) and the
optimizer / epoch counter start fresh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from fom.config import load_config
from fom.data.dataset import MotionDataModule
from fom.training.ema import EMA
from fom.training.module import FoMModule


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Stage YAML (configs/train/*.yaml).")
    parser.add_argument("--validate-only", action="store_true", help="Run one validation pass and exit.")
    parser.add_argument("overrides", nargs="*", help="KEY VALUE pairs, e.g. TRAIN.LR 1e-4")
    return parser.parse_intermixed_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if not cfg.TRAIN.EXP_DIR:
        raise ValueError("TRAIN.EXP_DIR must be set")
    exp_dir = Path(cfg.TRAIN.EXP_DIR)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(cfg.dump())
    logger.info(f"Config:\n{cfg.dump()}")

    pl.seed_everything(cfg.TRAIN.SEED, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    datamodule = MotionDataModule(cfg)
    module = FoMModule(cfg)

    trainer = pl.Trainer(
        default_root_dir=str(exp_dir),
        logger=TensorBoardLogger(str(exp_dir), name="", version="", default_hp_metric=False, flush_secs=60),
        devices=cfg.TRAIN.NUM_GPUS,
        num_nodes=1,
        accelerator="gpu",
        # Optional conditioning branches may be unused in a step (dropped modalities).
        strategy="ddp_find_unused_parameters_true" if cfg.TRAIN.NUM_GPUS > 1 else "auto",
        precision=cfg.TRAIN.PRECISION,
        num_sanity_val_steps=2,
        log_every_n_steps=cfg.TRAIN.LOG_EVERY_N_STEPS,
        callbacks=[
            ModelCheckpoint(
                dirpath=str(exp_dir), every_n_epochs=cfg.TRAIN.SAVE_EVERY_N_EPOCHS, save_top_k=-1, save_last=True
            ),
            LearningRateMonitor("step"),
            TQDMProgressBar(refresh_rate=1),
            EMA(cfg.TRAIN.EMA_DECAY),
        ],
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        check_val_every_n_epoch=cfg.TRAIN.VAL_EVERY_N_EPOCHS,
        gradient_clip_algorithm="norm",
        gradient_clip_val=cfg.TRAIN.GRAD_CLIP_NORM,
    )

    resume = exp_dir / "last.ckpt"
    ckpt_path = str(resume) if resume.exists() else None
    if ckpt_path:
        logger.info(f"Resuming from {ckpt_path}")
    elif cfg.TRAIN.INIT_CKPT:
        logger.info(f"Initializing weights from {cfg.TRAIN.INIT_CKPT}")
        state = torch.load(cfg.TRAIN.INIT_CKPT, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = module.load_state_dict(state, strict=False)
        logger.info(f"missing keys: {missing}\nunexpected keys: {unexpected}")

    if args.validate_only:
        trainer.validate(module, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
