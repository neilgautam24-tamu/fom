#!/usr/bin/env python3
"""Extract DINOv2 features for packed Nymeria sequences.

Reads ``ee_train.pt`` / ``ee_val.pt`` and corresponding ego RGB from
``visible_goal_sequence_bundles/vgoal_seq_*/ego_rgb.npz``.

Output:
  ``<packed-dir>/egoview_dinov2_{split}.pt``
    {"feats": {take_name: Tensor(T_5fps, 5, 1024)}}

FPS alignment defaults to the training format (15 FPS motion, 5 FPS DINOv2) and can be configured,
for example 30 FPS motion with 10 FPS DINO still uses one image every 3 frames.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

N_DINO_TOKENS = 5


def parse_take_name(take_name: str) -> tuple[str | None, str | None]:
    # mvgoal_seq_ must be checked first: it contains vgoal_seq_ as a substring.
    for marker in ("mvgoal_seq_", "vgoal_seq_", "win_"):
        idx = take_name.find(marker)
        if idx != -1:
            return take_name[:idx].rstrip("_"), take_name[idx:]
    return None, None


def resolve_bundle_dir(
    episode: str,
    bundle_name: str,
    *,
    single_root: Path,
    multi_root: Path,
) -> Path | None:
    if bundle_name.startswith("vgoal_seq_"):
        bundle_dir = single_root / episode / "visible_goal_sequence_bundles" / bundle_name
    elif bundle_name.startswith("mvgoal_seq_"):
        bundle_dir = multi_root / episode / "multi_object_visible_goal_bundles" / bundle_name
    elif bundle_name.startswith("win_"):
        bundle_dir = single_root / episode / "window_bundles" / bundle_name
    else:
        return None
    return bundle_dir if bundle_dir.is_dir() else None


def load_rgb_frames_chronological(rgb_npz: Path) -> list[Image.Image]:
    data = np.load(rgb_npz)
    rgb = data["rgb"]
    if "frame_indices" in data.files:
        order = np.argsort(data["frame_indices"].astype(np.int64))
        rgb = rgb[order]
    return [Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB") for frame in rgb]


def load_window_rgb_frames(
    bundle_dir: Path,
    num_motion_frames: int,
    *,
    episode_rgb_cache: dict[Path, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[Image.Image]:
    """Load RGB for a window; supports per-window or episode-level ego_rgb.npz."""
    local = bundle_dir / "ego_rgb.npz"
    if local.is_file():
        frames = load_rgb_frames_chronological(local)
        if len(frames) >= num_motion_frames:
            return frames[:num_motion_frames]
        return frames

    ep_dir = bundle_dir.parent.parent
    ep_rgb = ep_dir / "ego_rgb.npz"
    if not ep_rgb.is_file():
        return []
    meta_path = bundle_dir / "metadata.json"
    if not meta_path.is_file():
        return []
    meta = json.loads(meta_path.read_text())
    start = int(meta["global_start_frame_idx"])
    end = int(meta["global_end_frame_idx"])

    key = ep_rgb.resolve()
    if episode_rgb_cache is not None and key in episode_rgb_cache:
        rgb, idxs = episode_rgb_cache[key]
    else:
        logger.info(f"Loading episode RGB cache: {key}")
        data = np.load(key)
        idxs = np.asarray(data["frame_indices"], dtype=np.int64)
        order = np.argsort(idxs)
        rgb = np.asarray(data["rgb"][order], dtype=np.uint8)
        idxs = idxs[order]
        if episode_rgb_cache is not None:
            # Keep only the latest episode (~10GB) - takes are sorted by name.
            episode_rgb_cache.clear()
            episode_rgb_cache[key] = (rgb, idxs)

    selected = rgb[(idxs >= start) & (idxs <= end)]
    if selected.shape[0] == 0:
        return []
    # Align to converted motion length (typically every source frame already at 30 FPS).
    if selected.shape[0] > num_motion_frames:
        selected = selected[:num_motion_frames]
    frames = [
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        for frame in selected
    ]
    if len(frames) < num_motion_frames:
        frames.extend([frames[-1]] * (num_motion_frames - len(frames)))
    return frames[:num_motion_frames]


def select_frames_at_feature_fps(
    frames: list[Image.Image],
    num_motion_frames: int,
    *,
    motion_fps: int,
    feature_fps: int,
) -> list[Image.Image]:
    if motion_fps <= 0 or feature_fps <= 0 or motion_fps % feature_fps != 0:
        raise ValueError(
            f"motion_fps must be divisible by feature_fps, got "
            f"{motion_fps}/{feature_fps}"
        )
    feature_step = motion_fps // feature_fps
    num_dino_frames = math.ceil(num_motion_frames / feature_step)
    if not frames:
        blank = Image.new("RGB", (512, 512), (0, 0, 0))
        return [blank] * num_dino_frames

    # RGB and motion are stored on the same timeline. Select exact temporal
    # indices rather than linspace, which can smear alignment at sequence ends.
    indices = np.arange(0, num_motion_frames, feature_step, dtype=np.int64)
    indices = np.clip(indices, 0, len(frames) - 1)
    selected = [frames[i] for i in indices]

    if len(selected) < num_dino_frames:
        selected.extend([selected[-1]] * (num_dino_frames - len(selected)))
    return selected[:num_dino_frames]


class NymeriaDINOFeatureExtractor:
    def __init__(
        self,
        *,
        converted_dir: Path,
        single_root: Path,
        multi_root: Path,
        output_dir: Path,
        split: str,
        dinov2_model: str = "dinov2_vitl14_reg",
        device: str = "cuda",
        batch_size: int = 16,
        shard_id: int = 0,
        num_shards: int = 1,
        motion_fps: int = 15,
        feature_fps: int = 5,
    ) -> None:
        assert split in ("train", "val")
        assert 0 <= shard_id < num_shards
        self.converted_dir = converted_dir
        self.single_root = single_root
        self.multi_root = multi_root
        self.output_dir = output_dir
        self.split = split
        self.device = device
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.dinov2_model_name = dinov2_model
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
        # Episode ego_rgb.npz is ~10GB; reuse across windows of the same episode.
        self._episode_rgb_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

        logger.info(f"Loading DINOv2 model: {dinov2_model}")
        self.dinov2_model = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        self.dinov2_model.to(device).eval()

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            out = self.dinov2_model.forward_features(dummy)
            feat_dim = out["x_norm_clstoken"].shape[-1]
            n_reg = out["x_norm_regtokens"].shape[1] if "x_norm_regtokens" in out else 0
            self.n_tokens = 1 + n_reg

        logger.info(f"Model loaded - feat_dim={feat_dim}, n_tokens={self.n_tokens}")
        self.dinov2_transform = self._get_dinov2_transform()

        pt_path = self.converted_dir / f"ee_{split}.pt"
        logger.info(f"Loading sequences from {pt_path}")
        self.sequences = torch.load(pt_path, weights_only=False)
        logger.info(f"Loaded {len(self.sequences)} sequences")

        self.all_features: dict[str, torch.Tensor] = {}
        self.failed_sequences: list[str] = []
        take_names = sorted({k.split("___")[0] for k in self.sequences})
        self.shard_takes = {
            name
            for i, name in enumerate(take_names)
            if num_shards <= 1 or i % num_shards == shard_id
        }
        if num_shards > 1:
            logger.info(
                f"Shard {shard_id + 1}/{num_shards}: "
                f"{len(self.shard_takes)}/{len(take_names)} take names"
            )
            # Drop non-shard sequences immediately - ee_train.pt is tens of GB.
            before = len(self.sequences)
            self.sequences = {
                k: v
                for k, v in self.sequences.items()
                if k.split("___")[0] in self.shard_takes
            }
            import gc

            gc.collect()
            logger.info(
                f"Pruned sequences for shard: {len(self.sequences)}/{before} kept"
            )

    def _get_dinov2_transform(self):
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def extract_features_batch(self, images: list[Image.Image]) -> torch.Tensor:
        img_tensors = torch.stack([self.dinov2_transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            features = self.dinov2_model.forward_features(img_tensors)
        cls_token = features["x_norm_clstoken"]
        if "x_norm_regtokens" in features:
            feats = torch.cat([cls_token.unsqueeze(1), features["x_norm_regtokens"]], dim=1)
        else:
            feats = cls_token.unsqueeze(1)
        return feats.cpu()

    def process_sequence(self, seq_name: str, seq_data: dict) -> tuple[str, torch.Tensor] | None:
        try:
            take_name = seq_name.split("___")[0]
            episode, bundle_name = parse_take_name(take_name)
            if episode is None or bundle_name is None:
                logger.warning(f"Could not parse take name: {take_name}")
                return None
            bundle_dir = resolve_bundle_dir(
                episode,
                bundle_name,
                single_root=self.single_root,
                multi_root=self.multi_root,
            )
            if bundle_dir is None:
                logger.warning(f"No bundle for {take_name}")
                return None

            rgb_path = bundle_dir / "ego_rgb.npz"
            num_motion_frames = int(seq_data["num_frames"])
            num_dino_frames = math.ceil(num_motion_frames / self.feature_step)

            if rgb_path.is_file() or (bundle_dir.parent.parent / "ego_rgb.npz").is_file():
                rgb_frames = load_window_rgb_frames(
                    bundle_dir,
                    num_motion_frames,
                    episode_rgb_cache=self._episode_rgb_cache,
                )
            else:
                logger.warning(f"Missing ego_rgb.npz for {take_name}")
                return None
            if not rgb_frames:
                logger.warning(f"Empty RGB for {take_name}")
                return None

            selected = select_frames_at_feature_fps(
                rgb_frames,
                num_motion_frames,
                motion_fps=self.motion_fps,
                feature_fps=self.feature_fps,
            )

            all_feats: list[torch.Tensor] = []
            for i in range(0, len(selected), self.batch_size):
                batch_feats = self.extract_features_batch(selected[i : i + self.batch_size])
                all_feats.append(batch_feats)

            features = torch.cat(all_feats, dim=0)
            assert features.shape[0] == num_dino_frames, (
                f"{take_name}: got {features.shape[0]} dino frames, expected {num_dino_frames}"
            )
            assert features.shape[1] == self.n_tokens
            assert features.shape[2] == 1024
            return take_name, features
        except Exception as exc:
            logger.error(f"Failed {seq_name}: {exc}")
            self.failed_sequences.append(seq_name)
            return None

    def extract_all(self) -> None:
        # One DINO tensor per take_name (base window), not per ___segment key.
        take_to_seq: dict[str, tuple[str, dict]] = {}
        for seq_name, seq_data in self.sequences.items():
            take_name = seq_name.split("___")[0]
            if take_name not in self.shard_takes:
                continue
            if take_name not in take_to_seq:
                take_to_seq[take_name] = (seq_name, seq_data)

        for take_name in tqdm(
            sorted(take_to_seq),
            desc=f"DINOv2 {self.split}",
        ):
            seq_name, seq_data = take_to_seq[take_name]
            result = self.process_sequence(seq_name, seq_data)
            if result is None:
                continue
            take_name, features = result
            self.all_features[take_name] = features

        if not self.all_features:
            raise RuntimeError(
                f"No DINO features extracted for split={self.split} "
                f"shard={self.shard_id}/{self.num_shards}"
            )

        out_path = self.output_dir / f"egoview_dinov2_{self.split}.pt"
        if self.num_shards > 1:
            out_path = out_path.with_name(
                f"{out_path.stem}.shard{self.shard_id:03d}of{self.num_shards:03d}"
                f"{out_path.suffix}"
            )
        torch.save({"feats": self.all_features}, out_path)
        logger.info(
            f"Saved {len(self.all_features)} sequences -> {out_path} "
            f"(failed {len(self.failed_sequences)})"
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
        "--bundle-root",
        type=Path,
        default=None,
        help="Deprecated: use --single-root/--multi-root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --converted-dir",
    )
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--dinov2-model", default="dinov2_vitl14_reg")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--motion-fps", type=int, default=15)
    parser.add_argument("--feature-fps", type=int, default=5)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    output_dir = args.output_dir or args.converted_dir
    single_root = args.single_root.resolve()
    if args.bundle_root is not None:
        single_root = args.bundle_root.resolve().parent.parent
    extractor = NymeriaDINOFeatureExtractor(
        converted_dir=args.converted_dir.resolve(),
        single_root=single_root,
        multi_root=args.multi_root.resolve(),
        output_dir=output_dir.resolve(),
        split=args.split,
        dinov2_model=args.dinov2_model,
        device=args.device if torch.cuda.is_available() else "cpu",
        batch_size=args.batch_size,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        motion_fps=args.motion_fps,
        feature_fps=args.feature_fps,
    )
    extractor.extract_all()


if __name__ == "__main__":
    main()
