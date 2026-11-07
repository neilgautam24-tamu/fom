"""Normalization statistics of the motion / trajectory representation over a packed train split.

Statistics are computed on whole (non-canonicalized) training sequences, never on validation data.

    python -m fom.preprocessing.packing.stats --train <dir>/ee_train.pt [--out <dir>/v4_beta_ee_train_stats.pt]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from fom.body.representation import MOTION_DIM, TRAJ_DIM, encode_sequence
from fom.body.smplx_model import load_smplx
from fom.data.files import STATS_FILE


def compute_statistics(train_path: str | Path, output_path: str | Path | None = None) -> dict[str, torch.Tensor]:
    train_path = Path(train_path)
    output_path = Path(output_path) if output_path else train_path.parent / STATS_FILE
    data = torch.load(train_path, weights_only=False)
    logger.info(f"Loaded {len(data)} training sequences from {train_path}")
    body_model = load_smplx().cpu()

    motions, trajs, failed = [], [], 0
    for name, seq in tqdm(data.items(), desc="stats"):
        try:
            params = copy.copy({k: v.cpu() for k, v in seq["smpl_params"].items()})
            params["betas"] = params["betas"][:1]  # stored once per sequence
            motion, traj = encode_sequence(
                seq["aria_traj"].cpu(), params, seq["kp3d"].cpu(), seq["floor_height"], body_model
            )
            motions.append(motion)
            trajs.append(traj)
        except Exception as exc:  # keep going; report the first few failures
            failed += 1
            if failed <= 5:
                logger.warning(f"Failed on {name}: {exc}")
    if not motions:
        raise RuntimeError("No sequence could be encoded")
    logger.info(f"Encoded {len(motions)}/{len(data)} sequences ({failed} failed)")

    motion = torch.cat(motions)
    traj = torch.cat(trajs)
    assert motion.shape[1] == MOTION_DIM and traj.shape[1] == TRAJ_DIM
    stats = {
        "motion_mean": motion.mean(dim=0),
        "motion_std": motion.std(dim=0),
        "motion_min": motion.min(dim=0)[0],
        "motion_max": motion.max(dim=0)[0],
        "traj_mean": traj.mean(dim=0),
        "traj_std": traj.std(dim=0),
        "traj_min": traj.min(dim=0)[0],
        "traj_max": traj.max(dim=0)[0],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, output_path)
    logger.info(f"Saved statistics -> {output_path}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True, help="Packed ee_train.pt")
    parser.add_argument("--out", type=Path, default=None, help=f"Default: next to --train as {STATS_FILE}")
    args = parser.parse_args()
    compute_statistics(args.train, args.out)


if __name__ == "__main__":
    main()
