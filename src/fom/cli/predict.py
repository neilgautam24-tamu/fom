"""Run a trained model on a packed validation set and save decoded predictions.

    fom-predict --config configs/train/stage3_headcam.yaml --ckpt <ckpt> --out preds.pkl
    # cross-dataset (ADT) with the Nymeria normalization statistics:
    fom-predict ... DATA.ROOT <adt_dir> DATA.STATS_PATH <nymeria_dir>/v4_beta_ee_train_stats.pt
    # modality ablation (Table 2):
    fom-predict ... --drop-goal --drop-traj
"""

from __future__ import annotations

import argparse

import pytorch_lightning as pl
import torch
from loguru import logger

from fom.config import load_config
from fom.data.dataset import MotionDataset
from fom.evaluation.io import save
from fom.evaluation.runner import Ablation, predict
from fom.training.module import load_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Config the checkpoint was trained with.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True, help="Output .pkl")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--full-length", action="store_true", help="Roll out whole sequences, not the first WINDOW.")
    for name in ("rgb", "depth", "goal", "traj"):
        parser.add_argument(f"--drop-{name}", action="store_true", help=f"Remove {name} conditioning at inference.")
    parser.add_argument("overrides", nargs="*", help="KEY VALUE config overrides")
    return parser.parse_intermixed_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    pl.seed_everything(cfg.TRAIN.SEED, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ablation = Ablation(args.drop_rgb, args.drop_depth, args.drop_goal, args.drop_traj)
    logger.info(f"Checkpoint {args.ckpt} | split {args.split} | dropped modalities: {ablation}")
    module = load_checkpoint(args.ckpt, cfg)
    dataset = MotionDataset(cfg, args.split)
    preds = predict(module, dataset, cfg.DATA.BATCH_SIZE, ablation, args.full_length, device)
    save(preds, args.out)


if __name__ == "__main__":
    main()
