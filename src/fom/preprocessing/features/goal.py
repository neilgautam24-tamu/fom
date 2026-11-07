#!/usr/bin/env python3
"""Extract per-frame goal conditioning features for Nymeria sequences.

Reads ``ee_{train,val}.pt`` and matching ``visible_goal_sequence_bundles/*/``:
  - ``goal_mask.npz``  -> 16x16 Gaussian heatmaps (256-d)
  - ``ego_rgb.npz``     -> DINOv2 patch tokens for Gaussian-weighted pooling (1024-d)

Output:
  ``<output-dir>/goal_cond_perframe_{split}.pt``
    {take_name: {heatmap, patch_pool, n_frames, status}}

Usage:
  python -m fom.preprocessing.features.goal --split train --device cuda
  python -m fom.preprocessing.features.goal --splits train val --batch-size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from loguru import logger
from PIL import Image
from tqdm import tqdm

from fom.preprocessing.features.common import (
    aggregate_status,
    frame_to_heatmap,
    load_bundle_rgb_mask,
    resolve_bundle_dir,
    resolve_device,
    rgb_to_pil_batch,
    select_shard,
    shard_output_path,
)


class NymeriaGoalFeatureExtractor:
    def __init__(
        self,
        *,
        converted_dir: Path,
        output_dir: Path,
        device: torch.device,
        bundle_root: Path | None = None,
        single_root: Path | None = None,
        multi_root: Path | None = None,
        batch_size: int = 32,
        dinov2_model: str = "dinov2_vitl14_reg",
        shard_id: int = 0,
        num_shards: int = 1,
        subshard_id: int = 0,
        num_subshards: int = 1,
        checkpoint_every: int = 50,
    ) -> None:
        self.converted_dir = converted_dir
        self.bundle_root = bundle_root
        self.single_root = single_root
        self.multi_root = multi_root
        self.output_dir = output_dir
        self.device = device
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.subshard_id = subshard_id
        self.num_subshards = num_subshards
        self.checkpoint_every = max(0, int(checkpoint_every))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading DINOv2: {dinov2_model} on {device}")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", dinov2_model, verbose=False
        )
        self.model.to(device).eval()
        self.transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def patch_tokens_batch(self, images: list[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        out = self.model.forward_features(batch)
        return out["x_norm_patchtokens"].cpu()  # (B, 256, 1024)

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
            return {
                "status": "no_bundle",
                "heatmap": torch.zeros(1, 256),
                "patch_pool": torch.zeros(1, 1024),
                "n_frames": 0,
            }

        rgb_path = bundle_dir / "ego_rgb.npz"
        mask_path = bundle_dir / "goal_mask.npz"
        if not rgb_path.is_file() or not mask_path.is_file():
            return {
                "status": "no_mask_npz",
                "heatmap": torch.zeros(1, 256),
                "patch_pool": torch.zeros(1, 1024),
                "n_frames": 0,
            }

        try:
            rgb, mask, goal_uv, goal_visible = load_bundle_rgb_mask(
                bundle_dir, num_frames
            )
        except ValueError as exc:
            logger.warning(f"{take_name}: {exc}")
            return {
                "status": "frame_mismatch",
                "heatmap": torch.zeros(1, 256),
                "patch_pool": torch.zeros(1, 1024),
                "n_frames": 0,
            }

        heatmaps: list[torch.Tensor] = []
        frame_statuses: list[str] = []
        for i in range(num_frames):
            h, status = frame_to_heatmap(mask[i], goal_uv[i], int(goal_visible[i]))
            heatmaps.append(h)
            frame_statuses.append(status)

        heatmap_stack = torch.stack(heatmaps, dim=0)
        pil_frames = rgb_to_pil_batch(rgb)
        patch_pools: list[torch.Tensor] = []

        for start in range(0, num_frames, self.batch_size):
            batch_imgs = pil_frames[start : start + self.batch_size]
            batch_heatmaps = heatmap_stack[start : start + self.batch_size]
            tokens = self.patch_tokens_batch(batch_imgs)
            for j in range(tokens.shape[0]):
                pool = (batch_heatmaps[j].unsqueeze(1) * tokens[j]).sum(dim=0)
                patch_pools.append(pool)

        return {
            "status": aggregate_status(frame_statuses),
            "heatmap": heatmap_stack,
            "patch_pool": torch.stack(patch_pools, dim=0),
            "n_frames": num_frames,
        }

    def _output_path(self, split: str) -> Path:
        out_path = shard_output_path(
            self.output_dir / f"goal_cond_perframe_{split}.pt",
            self.shard_id,
            self.num_shards,
        )
        if self.num_subshards > 1:
            out_path = out_path.with_name(
                f"{out_path.stem}.sub{self.subshard_id:03d}of"
                f"{self.num_subshards:03d}{out_path.suffix}"
            )
        return out_path

    def extract_split(self, split: str) -> None:
        pt_path = self.converted_dir / f"ee_{split}.pt"
        if not pt_path.is_file():
            logger.warning(f"Skipping missing {pt_path}")
            return

        out_path = self._output_path(split)
        if out_path.is_file():
            logger.info(f"[{split}] {out_path} exists - skipping")
            return

        data = torch.load(pt_path, weights_only=False)
        base_info: dict[str, int] = {}
        for key, seq in data.items():
            base = key.rsplit("___", 2)[0]
            if base not in base_info:
                base_info[base] = int(seq["num_frames"])

        logger.info(f"[{split}] {len(base_info)} sequences from {pt_path}")
        shard_names = select_shard(
            sorted(base_info.keys()), self.shard_id, self.num_shards
        )
        if self.num_shards > 1:
            logger.info(
                f"[{split}] shard {self.shard_id + 1}/{self.num_shards}: "
                f"{len(shard_names)} sequences"
            )
        if self.num_subshards > 1:
            shard_names = select_shard(
                shard_names, self.subshard_id, self.num_subshards
            )
            logger.info(
                f"[{split}] subshard {self.subshard_id + 1}/{self.num_subshards}: "
                f"{len(shard_names)} sequences"
            )

        partial_path = out_path.with_suffix(".partial.pt")
        results: dict[str, dict] = {}
        status_counts: dict[str, int] = {}
        if partial_path.is_file():
            results = torch.load(partial_path, weights_only=False)
            for result in results.values():
                status = result.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            logger.info(
                f"[{split}] resumed {len(results)} sequences from {partial_path}"
            )

        pending = [name for name in shard_names if name not in results]
        done_since_ckpt = 0
        for take_name in tqdm(pending, desc=f"goal {split}"):
            num_frames = base_info[take_name]
            result = self.process_sequence(take_name, num_frames)
            if result is None:
                continue
            results[take_name] = result
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            done_since_ckpt += 1
            if (
                self.checkpoint_every > 0
                and done_since_ckpt % self.checkpoint_every == 0
            ):
                torch.save(results, partial_path)
                logger.info(
                    f"[{split}] checkpoint {len(results)}/{len(shard_names)} -> "
                    f"{partial_path}"
                )

        torch.save(results, out_path)
        if partial_path.is_file():
            partial_path.unlink()
        logger.info(f"Saved {len(results)} sequences -> {out_path}")
        for status, count in sorted(status_counts.items()):
            logger.info(f"  {status}: {count}")


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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dinov2-model", default="dinov2_vitl14_reg")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--subshard-id", type=int, default=0)
    parser.add_argument("--num-subshards", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save partial progress every N newly processed sequences (0=off)",
    )
    parser.add_argument("--split", choices=["train", "val"], default=None)
    args = parser.parse_args()

    splits = [args.split] if args.split else args.splits

    device = resolve_device(args.device)
    output_dir = args.output_dir or args.converted_dir
    bundle_root = args.bundle_root.resolve() if args.bundle_root else None
    extractor = NymeriaGoalFeatureExtractor(
        converted_dir=args.converted_dir.resolve(),
        bundle_root=bundle_root,
        single_root=args.single_root.resolve() if bundle_root is None else None,
        multi_root=args.multi_root.resolve() if bundle_root is None else None,
        output_dir=output_dir.resolve(),
        device=device,
        batch_size=args.batch_size,
        dinov2_model=args.dinov2_model,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        subshard_id=args.subshard_id,
        num_subshards=args.num_subshards,
        checkpoint_every=args.checkpoint_every,
    )
    for split in splits:
        extractor.extract_split(split)


if __name__ == "__main__":
    main()
