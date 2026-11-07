"""Packed egocentric motion dataset (Nymeria, ADT) and its Lightning data module.

Each item is one clip starting at frame 0, truncated / zero-padded to ``DATA.WINDOW`` frames::

    {
      "y":    conditioning dict consumed by ForesightToMotion (see fom.models.fom),
      "misc": {"seq_name", "start_idx", "end_idx", "motion" (T, 243), "traj" (T, 18)},
    }

``motion`` and ``traj`` are normalized. ``start_idx`` / ``end_idx`` are 30 fps frame indices.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from yacs.config import CfgNode

from fom.body.canonicalization import canonicalized_segment
from fom.body.representation import decode_sequence, encode_sequence
from fom.body.smplx_model import evaluate_smplx, load_smplx
from fom.data import files
from fom.data.features import RgbFeatures, hold_to_motion_rate
from fom.data.normalization import Normalizer, stats_path
from fom.utils.tensors import collate

GOAL_VALID_STATUSES = ("ok", "outside_crop", "outside_image", "partial_outside")
GOAL_DIM = 256


class MotionDataset(Dataset):
    def __init__(
        self,
        cfg: CfgNode,
        split: str,
        *,
        load_rgb: bool | None = None,
        load_depth: bool | None = None,
        load_goal: bool | None = None,
        normalize: bool = True,
    ):
        """
        Args:
            cfg: experiment config; ``DATA.*`` and ``MODEL.COND_*`` are used.
            split: ``"train"`` or ``"val"``.
            load_rgb / load_depth / load_goal: override which feature files are loaded
                (default: follow ``MODEL.COND_*``). Ground-truth-only uses can skip them all.
            normalize: z-score motion / trajectory with the training statistics.
        """
        super().__init__()
        assert split in ("train", "val")
        self.split = split
        self.root = cfg.DATA.ROOT
        self.window = cfg.DATA.WINDOW
        self.motion_fps = cfg.DATA.MOTION_FPS
        self.visual_fps = cfg.DATA.VISUAL_FPS
        self.cond_traj = cfg.MODEL.COND_TRAJ
        self.load_rgb = cfg.MODEL.COND_RGB if load_rgb is None else load_rgb
        self.load_depth = cfg.MODEL.COND_DEPTH if load_depth is None else load_depth
        self.load_goal = cfg.MODEL.COND_GOAL if load_goal is None else load_goal

        self.body_model = load_smplx()
        self.normalizer = Normalizer.from_file(stats_path(cfg)) if normalize else None

        path = files.motion_file(self.root, split)
        logger.info(f"Loading {path}")
        self.motion_data = torch.load(path, weights_only=False)
        self.seq_names = [name for name, seq in self.motion_data.items() if seq["num_frames"] > 0]
        logger.info(f"{split}: {len(self.seq_names)} sequences")

        self.rgb = RgbFeatures(self.root, split, self.motion_fps, self.visual_fps) if self.load_rgb else None
        self.goal_feats = self._load(files.goal_feature_file(self.root, split)) if self.load_goal else None
        self.depth_feats = self._load(files.depth_feature_file(self.root, split)) if self.load_depth else None

    @staticmethod
    def _load(path) -> dict:
        logger.info(f"Loading {path}")
        return torch.load(path, weights_only=False)

    def __len__(self) -> int:
        return len(self.seq_names)

    def __getitem__(self, idx: int) -> dict:
        return self.get_sequence(self.seq_names[idx])

    def get_sequence(self, seq_name: str, full_length: bool = False) -> dict:
        """Clip ``seq_name`` from frame 0.

        Args:
            full_length: return the whole sequence (no truncation / padding) for long rollouts.
        """
        seq = self.motion_data[seq_name]
        num_frames = seq["num_frames"]
        end = num_frames - 1 if full_length else min(self.window - 1, num_frames - 1)

        segment = canonicalized_segment(
            seq["smpl_params"], seq["aria_traj"], seq["kp3d"], self.body_model, 0, end, self.motion_fps
        )
        motion, traj = encode_sequence(
            segment["camera_T"], segment["smpl_params"], segment["joints"], seq["floor_height"], self.body_model
        )
        if self.normalizer is not None:
            motion = self.normalizer.normalize(motion, "motion")
            traj = self.normalizer.normalize(traj, "traj")
        length = motion.shape[0]

        y: dict[str, torch.Tensor] = {}
        if self.cond_traj:
            y["traj"] = traj
        if self.rgb is not None:
            y["rgb"], y["rgb_valid"] = self.rgb.get(seq_name, segment["start_idx"], segment["end_idx"])
        base_name = seq_name.rsplit("___", 2)[0]
        if self.goal_feats is not None:
            y["goal"] = self._goal(seq, base_name, segment["start_idx"], length)
        if self.depth_feats is not None:
            y["depth"], y["depth_valid"] = self._depth(base_name, length)

        item = {
            "y": y,
            "misc": {
                "seq_name": seq_name,
                "start_idx": segment["start_idx"],
                "end_idx": segment["end_idx"],
                "motion": motion,
                "traj": traj,
            },
        }
        if full_length:
            y["valid_frames"] = torch.ones(length, dtype=torch.long)
            return item
        return self._pad_to_window(item)

    def _goal(self, seq: dict, base_name: str, start_30fps: int, length: int) -> torch.Tensor:
        entry = self.goal_feats.get(base_name)
        if entry is None or entry.get("status") not in GOAL_VALID_STATUSES:
            return torch.zeros(length, GOAL_DIM)
        heatmap = entry["heatmap"]
        fps_ratio = 30 // self.motion_fps
        if "stored_fps" not in seq and heatmap.shape[0] >= seq["num_frames"] * fps_ratio:
            # ADT stores goal heatmaps at 30 fps: stride-downsample to the motion rate.
            goal = heatmap[start_30fps::fps_ratio][:length]
        else:
            # Nymeria stores them at the motion rate: follow the RGB staircase cadence.
            goal = hold_to_motion_rate(heatmap, 0, length, self.motion_fps, self.visual_fps)
        if goal.shape[0] < length:
            goal = torch.cat([goal, torch.zeros(length - goal.shape[0], GOAL_DIM, dtype=goal.dtype)], dim=0)
        return goal

    def _depth(self, base_name: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.depth_feats.get(base_name)
        if entry is None:
            return torch.zeros(length, 5, 1024), torch.zeros(length)
        patches = hold_to_motion_rate(entry["patches"], 0, length, self.motion_fps, self.visual_fps).float()
        valid = hold_to_motion_rate(entry["valid"], 0, length, self.motion_fps, self.visual_fps).float()
        return patches, valid

    def _pad_to_window(self, item: dict) -> dict:
        y, misc = item["y"], item["misc"]
        length = misc["motion"].shape[0]
        valid = torch.ones(self.window, dtype=torch.long)
        if length != self.window:
            valid[length:] = 0

            def pad(x: torch.Tensor) -> torch.Tensor:
                return torch.cat([x, torch.zeros(self.window - length, *x.shape[1:], dtype=x.dtype)], dim=0)

            for key in ("traj", "rgb", "rgb_valid", "goal", "depth", "depth_valid"):
                if key in y:
                    y[key] = pad(y[key])
            misc["motion"] = pad(misc["motion"])
            misc["traj"] = pad(misc["traj"])
        y["valid_frames"] = valid
        return item

    def to_body(self, batch: dict, motion: torch.Tensor | None = None) -> list[dict]:
        """Decode a collated batch to world-space SMPL-X sequences (valid frames only).

        Args:
            batch: collated items.
            motion: normalized ``(B, T', 243)`` predictions; defaults to the ground truth.

        Returns:
            One ``{"camera_T", "smpl_params", "joints", "vertices", "full_pose"}`` dict per item.
        """
        motion = batch["misc"]["motion"] if motion is None else motion
        traj = batch["misc"]["traj"]
        out = []
        for i in range(len(batch["misc"]["seq_name"])):
            valid = batch["y"]["valid_frames"][i].bool()
            n_valid = int(valid.sum())
            # Predictions may be padded to the batch-max valid length; valid frames are leading.
            motion_i = motion[i][valid] if motion[i].shape[0] == valid.shape[0] else motion[i][:n_valid]
            traj_i = traj[i][valid]
            if self.normalizer is not None:
                motion_i = self.normalizer.denormalize(motion_i, "motion")
                traj_i = self.normalizer.denormalize(traj_i, "traj")
            camera_T, params, _ = decode_sequence(motion_i, traj_i, self.body_model)
            joints, vertices, full_pose = evaluate_smplx(self.body_model, params)
            out.append(
                {"camera_T": camera_T, "smpl_params": params, "joints": joints, "vertices": vertices, "full_pose": full_pose}
            )
        return out


class MotionDataModule(pl.LightningDataModule):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = MotionDataset(self.cfg, "train")
        self.val_dataset = MotionDataset(self.cfg, "val")

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            shuffle=shuffle,
            collate_fn=collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)
