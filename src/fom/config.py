"""Experiment configuration (yacs).

Every option that the released model depends on lives here with its default.
Stage-specific values are in ``configs/train/*.yaml``; anything can also be overridden
from the command line as ``KEY VALUE`` pairs, e.g. ``TRAIN.LR 1e-4``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
# Directory holding the packed dataset files (see fom.data.files).
_C.DATA.ROOT = ""
# Normalization statistics. Empty -> <DATA.ROOT>/<STATS_FILE>. Point this at the training-set
# statistics when evaluating on another dataset (e.g. ADT with Nymeria statistics).
_C.DATA.STATS_PATH = ""
_C.DATA.WINDOW = 150  # frames per sample (10 s at 15 fps)
_C.DATA.MOTION_FPS = 15
_C.DATA.VISUAL_FPS = 5  # RGB/depth/goal cadence; held for MOTION_FPS // VISUAL_FPS frames
_C.DATA.BATCH_SIZE = 64
_C.DATA.NUM_WORKERS = 2

_C.MODEL = CN()
_C.MODEL.LATENT_DIM = 768
_C.MODEL.NUM_LAYERS = 12
_C.MODEL.NUM_HEADS = 12
_C.MODEL.FF_MULT = 2
_C.MODEL.DROPOUT = 0.1
# Which conditioning streams the architecture has. Removing a stream at inference is done
# with the --drop-* flags of `fom-predict`, not here.
_C.MODEL.COND_RGB = True
_C.MODEL.COND_DEPTH = True
_C.MODEL.COND_GOAL = True
_C.MODEL.COND_TRAJ = True
# Rollout implementation used for self-forcing training and for sampling. This is part of the
# model definition, not just an optimization: the cached step projects self-attention keys and
# values from the layer-normed token (the parallel path uses the raw token) and applies only
# explicit trajectory masks. All released checkpoints were evaluated with the cached rollout.
_C.MODEL.USE_KV_CACHE = False

# Modality dropout during training (probabilities per sequence).
_C.MODEL.MASKING = CN()
_C.MODEL.MASKING.TRAJ = 0.5  # replace the whole trajectory by the mask token
_C.MODEL.MASKING.RGB = 0.1  # replace all RGB frames by the mask token
# Second, independent draw: masks the whole trajectory again and, for RGB, every frame
# except the first (the "first frame + rollout" setting).
_C.MODEL.MASKING.SUBSEQUENT = 0.5
_C.MODEL.MASKING.DEPTH = 0.5
_C.MODEL.MASKING.GOAL = 0.5

_C.TRAIN = CN()
_C.TRAIN.EXP_DIR = ""
# Weights to start from (previous stage). Resuming the same experiment is automatic from
# <EXP_DIR>/last.ckpt and does not need this.
_C.TRAIN.INIT_CKPT = ""
_C.TRAIN.SEED = 62
_C.TRAIN.NUM_EPOCHS = 300
_C.TRAIN.LR = 3.0e-5
_C.TRAIN.WEIGHT_DECAY = 0.01
# StepLR: multiply LR by LR_STEP_GAMMA every LR_STEP_EPOCHS epochs. 0 disables the decay.
_C.TRAIN.LR_STEP_EPOCHS = 0
_C.TRAIN.LR_STEP_GAMMA = 0.1
_C.TRAIN.GRAD_CLIP_NORM = 1.0
_C.TRAIN.EMA_DECAY = 0.999
_C.TRAIN.NUM_GPUS = 1
_C.TRAIN.PRECISION = "32-true"  # Lightning precision string, e.g. "bf16-mixed"
_C.TRAIN.LOG_EVERY_N_STEPS = 10
_C.TRAIN.VAL_EVERY_N_EPOCHS = 10
_C.TRAIN.SAVE_EVERY_N_EPOCHS = 10
# "teacher": ground-truth previous frame as AR input (parallel). "self": the model's own
# (stop-gradient) previous prediction.
_C.TRAIN.FORCING = "teacher"
# Up-weighting of the global root orientation / translation channels in the motion loss.
_C.TRAIN.GLOBAL_ORIENT_WEIGHT = 1.5
_C.TRAIN.GLOBAL_TRANS_WEIGHT = 1.5
# Stage III: weight of |predicted left eye - camera position|^2. 0 disables it.
_C.TRAIN.HEADCAM_LOSS_WEIGHT = 0.0

_FORCING_MODES = ("teacher", "self")


def default_config() -> CN:
    return _C.clone()


def load_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> CN:
    """Defaults <- YAML file <- ``KEY VALUE`` overrides, then validate and freeze."""
    cfg = default_config()
    if path:
        cfg.merge_from_file(str(path))
    if overrides:
        cfg.merge_from_list(list(overrides))
    validate(cfg)
    cfg.freeze()
    return cfg


def validate(cfg: CN) -> None:
    if cfg.TRAIN.FORCING not in _FORCING_MODES:
        raise ValueError(f"TRAIN.FORCING must be one of {_FORCING_MODES}, got {cfg.TRAIN.FORCING!r}")
    if cfg.DATA.MOTION_FPS % cfg.DATA.VISUAL_FPS:
        raise ValueError("DATA.MOTION_FPS must be a multiple of DATA.VISUAL_FPS")
    if cfg.TRAIN.HEADCAM_LOSS_WEIGHT > 0 and not cfg.MODEL.COND_TRAJ:
        raise ValueError("The head-camera loss needs the camera trajectory (MODEL.COND_TRAJ true).")


def config_to_dict(cfg: CN) -> dict:
    return {k: config_to_dict(v) if isinstance(v, CN) else v for k, v in cfg.items()}
