#!/usr/bin/env python3
"""Extract per-frame depth patch features for Nymeria sequences.

Runs metric depth estimation (GPU) on ``ego_rgb.npz`` frames, then converts
each depth map to five 32x32 patches (ADT-compatible layout).

By default, depth is estimated at ``feature_fps`` (aligned with DINOv2 /
``VISUAL_FPS``) and held across motion frames so the tensor stays length-T at
``motion_fps``. Training already reads depth with the same staircase.

Output:
  ``<output-dir>/depth_{split}.pt``
    {take_name: {patches: (T,5,1024), valid: (T,)}}

Usage:
  python -m fom.preprocessing.features.depth --split train --device cuda
  python -m fom.preprocessing.features.depth --backend vipe --batch-size 8 --feature-fps 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from fom.paths import torch_hub_dir
from fom.preprocessing.features.common import (
    depth_to_patches,
    resolve_bundle_dir,
    resolve_device,
    select_shard,
    shard_output_path,
)
from fom.preprocessing.features.depth_model import (
    DepthEstimator,
    build_depth_estimator,
    default_vipe_path,
)


class NymeriaDepthFeatureExtractor:
    def __init__(
        self,
        *,
        converted_dir: Path,
        output_dir: Path,
        depth_estimator: DepthEstimator,
        device: torch.device,
        bundle_root: Path | None = None,
        single_root: Path | None = None,
        multi_root: Path | None = None,
        shard_id: int = 0,
        num_shards: int = 1,
        names_file: Path | None = None,
        batch_size: int = 8,
        motion_fps: int = 30,
        feature_fps: int = 10,
    ) -> None:
        self.converted_dir = converted_dir
        self.bundle_root = bundle_root
        self.single_root = single_root
        self.multi_root = multi_root
        self.output_dir = output_dir
        self.depth_estimator = depth_estimator
        self.device = device
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.names_file = names_file
        self.batch_size = max(1, int(batch_size))
        self.motion_fps = int(motion_fps)
        self.feature_fps = int(feature_fps)
        if (
            self.motion_fps <= 0
            or self.feature_fps <= 0
            or self.motion_fps % self.feature_fps != 0
        ):
            raise ValueError(
                f"motion_fps must be divisible by feature_fps, got "
                f"{self.motion_fps}/{self.feature_fps}"
            )
        self.feature_step = self.motion_fps // self.feature_fps
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Episode ego_rgb.npz is ~10GB; windows from one episode are contiguous
        # after sorting take names, so keep the last loaded episode in memory.
        self._episode_rgb_cache_key: Path | None = None
        self._episode_rgb_cache: tuple[np.ndarray, np.ndarray] | None = None

    def _load_episode_rgb(self, ep_rgb: Path) -> tuple[np.ndarray, np.ndarray]:
        key = ep_rgb.resolve()
        if self._episode_rgb_cache_key == key and self._episode_rgb_cache is not None:
            return self._episode_rgb_cache
        logger.info(f"Loading episode RGB cache: {key}")
        data = np.load(key)
        idxs = np.asarray(data["frame_indices"], dtype=np.int64)
        order = np.argsort(idxs)
        rgb = np.asarray(data["rgb"][order], dtype=np.uint8)
        idxs = idxs[order]
        self._episode_rgb_cache_key = key
        self._episode_rgb_cache = (rgb, idxs)
        return self._episode_rgb_cache

    def _load_window_rgb(self, bundle_dir: Path, num_frames: int) -> np.ndarray:
        import json

        local = bundle_dir / "ego_rgb.npz"
        if local.is_file():
            rgb_npz = np.load(local)
            order = np.argsort(rgb_npz["frame_indices"].astype(np.int64))
            rgb = np.asarray(rgb_npz["rgb"][order], dtype=np.uint8)
            if rgb.shape[0] < num_frames:
                raise ValueError(
                    f"{bundle_dir.name}: bundle has {rgb.shape[0]} RGB frames, "
                    f"expected {num_frames}"
                )
            return rgb[:num_frames]

        ep_rgb = bundle_dir.parent.parent / "ego_rgb.npz"
        meta_path = bundle_dir / "metadata.json"
        if not ep_rgb.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"missing episode RGB for {bundle_dir}")
        meta = json.loads(meta_path.read_text())
        start = int(meta["global_start_frame_idx"])
        end = int(meta["global_end_frame_idx"])
        rgb, idxs = self._load_episode_rgb(ep_rgb)
        selected = rgb[(idxs >= start) & (idxs <= end)]
        if selected.shape[0] == 0:
            raise ValueError(f"{bundle_dir.name}: empty RGB slice")
        if selected.shape[0] < num_frames:
            pad = np.repeat(selected[-1:], num_frames - selected.shape[0], axis=0)
            selected = np.concatenate([selected, pad], axis=0)
        return selected[:num_frames]

    def process_sequence(
        self,
        take_name: str,
        num_frames: int,
    ) -> dict | None:
        bundle_dir = resolve_bundle_dir(
            take_name,
            bundle_root=self.bundle_root,
            single_root=self.single_root,
            multi_root=self.multi_root,
        )
        if bundle_dir is None:
            return None

        local_rgb = bundle_dir / "ego_rgb.npz"
        episode_rgb = bundle_dir.parent.parent / "ego_rgb.npz"
        if not local_rgb.is_file() and not episode_rgb.is_file():
            return None

        try:
            rgb = self._load_window_rgb(bundle_dir, num_frames)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning(f"{take_name}: {exc}")
            return None

        patches = np.zeros((num_frames, 5, 1024), dtype=np.float32)
        valid = np.zeros(num_frames, dtype=bool)

        # Estimate only at visual/feature fps; hold across motion frames so the
        # stored tensor stays length-T (dataset indexes with motion staircase).
        eval_idx = np.arange(0, num_frames, self.feature_step, dtype=np.int64)
        eval_idx = np.clip(eval_idx, 0, rgb.shape[0] - 1)

        for start in range(0, len(eval_idx), self.batch_size):
            batch_ids = eval_idx[start : start + self.batch_size]
            batch_rgb = np.ascontiguousarray(rgb[batch_ids])
            try:
                depth_batch = self.depth_estimator.predict_batch(batch_rgb)
            except Exception as exc:
                logger.warning(
                    f"{take_name} frames {batch_ids[0]}-{batch_ids[-1]}: "
                    f"depth batch failed ({exc}); falling back per-frame"
                )
                depth_batch = np.stack(
                    [
                        self.depth_estimator.predict(batch_rgb[i])
                        for i in range(len(batch_ids))
                    ],
                    axis=0,
                )

            for j, ti in enumerate(batch_ids):
                depth = depth_batch[j]
                if depth.shape[:2] != rgb.shape[1:3]:
                    import cv2

                    depth = cv2.resize(
                        depth,
                        (rgb.shape[2], rgb.shape[1]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                patch = depth_to_patches(depth)
                is_valid = bool(np.isfinite(depth).any() and (depth > 0).any())
                end = min(int(ti) + self.feature_step, num_frames)
                patches[int(ti) : end] = patch
                valid[int(ti) : end] = is_valid

        return {
            "patches": torch.from_numpy(patches),
            "valid": torch.from_numpy(valid),
        }

    def extract_split(self, split: str, *, no_skip: bool = False) -> None:
        stem = f"depth_{split}.missing" if self.names_file is not None else f"depth_{split}"
        out_path = shard_output_path(
            self.output_dir / f"{stem}.pt",
            self.shard_id,
            self.num_shards,
        )
        if out_path.exists() and not no_skip:
            logger.info(f"[{split}] {out_path} exists - skipping (use --no-skip)")
            return

        pt_path = self.converted_dir / f"ee_{split}.pt"
        if not pt_path.is_file():
            logger.warning(f"Skipping missing {pt_path}")
            return

        data = torch.load(pt_path, weights_only=False)
        base_info: dict[str, int] = {}
        for key, seq in data.items():
            base = key.rsplit("___", 2)[0]
            if base not in base_info:
                base_info[base] = int(seq["num_frames"])
        del data
        import gc

        gc.collect()

        if self.names_file is not None:
            requested = [
                line.strip()
                for line in self.names_file.read_text().splitlines()
                if line.strip()
            ]
            unknown = [name for name in requested if name not in base_info]
            if unknown:
                logger.warning(
                    f"{len(unknown)} names-file entries not in ee_{split}"
                )
            names = sorted(name for name in requested if name in base_info)
            logger.info(
                f"[{split}] depth on {self.device} - {len(names)} names-file "
                f"sequences ({len(base_info)} in split)"
            )
        else:
            names = sorted(base_info.keys())
            logger.info(
                f"[{split}] depth on {self.device} - {len(names)} sequences"
            )
        logger.info(
            f"[{split}] batch_size={self.batch_size} "
            f"motion_fps={self.motion_fps} feature_fps={self.feature_fps} "
            f"(eval every {self.feature_step} motion frames)"
        )
        shard_names = select_shard(names, self.shard_id, self.num_shards)
        if self.num_shards > 1:
            logger.info(
                f"[{split}] shard {self.shard_id + 1}/{self.num_shards}: "
                f"{len(shard_names)} sequences"
            )
        depth_feats: dict[str, dict] = {}
        n_ok = n_miss = 0

        for take_name in tqdm(shard_names, desc=f"depth {split}"):
            num_frames = base_info[take_name]
            result = self.process_sequence(take_name, num_frames)
            if result is None:
                n_miss += 1
                continue
            depth_feats[take_name] = result
            n_ok += 1

        torch.save(depth_feats, out_path)
        logger.info(
            f"Saved {n_ok} sequences -> {out_path} (missing/failed {n_miss})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--converted-dir",
        type=Path,
        required=True,
        help="Packed dataset directory (ee_{train,val}.pt)",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=None,
        help="Episode-level bundles dir (visible_goal_sequence_bundles)",
    )
    parser.add_argument(
        "--single-root",
        type=Path,
        required=True,
        help="Root with visible_goal_sequence_bundles per episode",
    )
    parser.add_argument(
        "--multi-root",
        type=Path,
        required=True,
        help="Root with multi_object_visible_goal_bundles per episode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --converted-dir",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--backend",
        default="midas",
        choices=["midas", "torchhub", "vipe"],
        help="midas = MiDaS DPT (default); torchhub = UniDepth v2; vipe = ADT ViPE backend",
    )
    parser.add_argument(
        "--vipe-path",
        type=Path,
        default=None,
        help="Path to vipe repo (required for --backend vipe)",
    )
    parser.add_argument(
        "--depth-model",
        default="unidepth-l",
        choices=["unidepth-l", "unidepth-b", "unidepth-s"],
    )
    parser.add_argument(
        "--hub-dir",
        type=Path,
        default=torch_hub_dir(),
        help="torch.hub cache directory (use scratch to avoid home quota limits)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of RGB frames per UniDepth/MiDaS forward pass",
    )
    parser.add_argument(
        "--motion-fps",
        type=int,
        default=30,
        help="Native motion / RGB timeline fps (must match ee_*.pt)",
    )
    parser.add_argument(
        "--feature-fps",
        type=int,
        default=10,
        help="Depth estimation rate; held across motion frames (match VISUAL_FPS)",
    )
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--names-file",
        type=Path,
        default=None,
        help="If set, only extract these take names and write depth_{split}.missing*.pt",
    )
    parser.add_argument("--split", choices=["train", "val"], default=None)
    args = parser.parse_args()

    splits = [args.split] if args.split else args.splits

    device = resolve_device(args.device)
    vipe_path = args.vipe_path or default_vipe_path()
    if args.backend == "vipe" and vipe_path is None:
        raise SystemExit(
            "ViPE backend selected but --vipe-path / VIPE_PATH not found."
        )

    logger.info(
        f"Depth backend={args.backend} model={args.depth_model} device={device} "
        f"batch_size={args.batch_size} motion_fps={args.motion_fps} "
        f"feature_fps={args.feature_fps}"
    )
    depth_estimator = build_depth_estimator(
        backend=args.backend,
        device=device,
        vipe_path=vipe_path,
        depth_model=args.depth_model,
        hub_dir=args.hub_dir,
    )

    output_dir = args.output_dir or args.converted_dir
    bundle_root = args.bundle_root.resolve() if args.bundle_root else None
    extractor = NymeriaDepthFeatureExtractor(
        converted_dir=args.converted_dir.resolve(),
        bundle_root=bundle_root,
        single_root=args.single_root.resolve() if bundle_root is None else None,
        multi_root=args.multi_root.resolve() if bundle_root is None else None,
        output_dir=output_dir.resolve(),
        depth_estimator=depth_estimator,
        device=device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        names_file=args.names_file.resolve() if args.names_file else None,
        batch_size=args.batch_size,
        motion_fps=args.motion_fps,
        feature_fps=args.feature_fps,
    )
    for split in splits:
        extractor.extract_split(split, no_skip=args.no_skip)


if __name__ == "__main__":
    main()
