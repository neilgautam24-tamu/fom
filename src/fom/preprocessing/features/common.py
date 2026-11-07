"""Shared helpers for Nymeria goal / depth feature extraction."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

# DINOv2 preprocessing (shared with the ADT feature extraction)
ORIG_SIZE = 512
RESIZE_SIZE = 256
CROP_SIZE = 224
PATCH_SIZE = 14
N_PATCHES = CROP_SIZE // PATCH_SIZE  # 16
BORDER = (RESIZE_SIZE - CROP_SIZE) // 2
GAUSS_SIGMA = 1.5

# Nymeria rectified head RGB (512x512, rotated CW90) - same as ADT ViPE pipeline
FOCAL = 280.0
CX = CY = 256.0

SMPL_FPS = 15


def resolve_device(device: str = "cuda") -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_take_name(take_name: str) -> tuple[str | None, str | None]:
    # mvgoal_seq_ must be checked first: it contains vgoal_seq_ as a substring.
    for marker in ("mvgoal_seq_", "vgoal_seq_", "win_"):
        idx = take_name.find(marker)
        if idx != -1:
            return take_name[:idx].rstrip("_"), take_name[idx:]
    return None, None


def resolve_bundle_dir(
    take_name: str,
    *,
    bundle_root: Path | None = None,
    single_root: Path | None = None,
    multi_root: Path | None = None,
) -> Path | None:
    episode, bundle_name = parse_take_name(take_name)
    if bundle_name is None:
        return None

    if bundle_root is not None:
        bundle_dir = bundle_root / bundle_name
    elif bundle_name.startswith("vgoal_seq_") and single_root is not None and episode:
        bundle_dir = (
            single_root / episode / "visible_goal_sequence_bundles" / bundle_name
        )
    elif bundle_name.startswith("mvgoal_seq_") and multi_root is not None and episode:
        bundle_dir = (
            multi_root / episode / "multi_object_visible_goal_bundles" / bundle_name
        )
    elif bundle_name.startswith("win_") and single_root is not None and episode:
        bundle_dir = single_root / episode / "window_bundles" / bundle_name
    else:
        return None

    return bundle_dir if bundle_dir.is_dir() else None


def base_names_from_split(data: dict) -> list[str]:
    return sorted({key.rsplit("___", 2)[0] for key in data.keys()})


def shard_output_path(path: Path, shard_id: int, num_shards: int) -> Path:
    if num_shards <= 1:
        return path
    return path.with_name(
        f"{path.stem}.shard{shard_id:03d}of{num_shards:03d}{path.suffix}"
    )


def select_shard(items: list[str], shard_id: int, num_shards: int) -> list[str]:
    if num_shards <= 1:
        return items
    return [name for i, name in enumerate(items) if i % num_shards == shard_id]


def load_window_rgb_array(bundle_dir: Path, num_frames: int) -> np.ndarray:
    """RGB (T,H,W,3) for a window from local or episode-level ego_rgb.npz."""
    local = bundle_dir / "ego_rgb.npz"
    if local.is_file():
        rgb_npz = np.load(local)
        order = np.argsort(rgb_npz["frame_indices"].astype(np.int64))
        rgb = np.asarray(rgb_npz["rgb"][order], dtype=np.uint8)
        if rgb.shape[0] < num_frames:
            raise ValueError(
                f"{bundle_dir.name}: bundle has {rgb.shape[0]} RGB frames, expected {num_frames}"
            )
        return rgb[:num_frames]

    ep_rgb = bundle_dir.parent.parent / "ego_rgb.npz"
    meta_path = bundle_dir / "metadata.json"
    if not ep_rgb.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"missing episode RGB for {bundle_dir}")
    meta = json.loads(meta_path.read_text())
    start = int(meta["global_start_frame_idx"])
    end = int(meta["global_end_frame_idx"])
    data = np.load(ep_rgb)
    idxs = np.asarray(data["frame_indices"], dtype=np.int64)
    order = np.argsort(idxs)
    rgb = np.asarray(data["rgb"][order], dtype=np.uint8)
    idxs = idxs[order]
    selected = rgb[(idxs >= start) & (idxs <= end)]
    if selected.shape[0] == 0:
        raise ValueError(f"{bundle_dir.name}: empty RGB slice")
    if selected.shape[0] < num_frames:
        pad = np.repeat(selected[-1:], num_frames - selected.shape[0], axis=0)
        selected = np.concatenate([selected, pad], axis=0)
    return selected[:num_frames]


def load_bundle_rgb_mask(
    bundle_dir: Path,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return rgb (T,H,W,3), mask (T,H,W), goal_uv (T,2), goal_visible (T,).

    Goal artifacts are optional (30s no-goal dataset uses zeros).
    """
    rgb = load_window_rgb_array(bundle_dir, num_frames)
    mask_path = bundle_dir / "goal_mask.npz"
    if mask_path.is_file():
        mask_npz = np.load(mask_path)
        order = np.argsort(mask_npz["frame_indices"].astype(np.int64))
        mask = np.asarray(mask_npz["mask"][order], dtype=np.uint8)
        goal_uv = np.asarray(mask_npz["goal_uv"][order], dtype=np.float32)
        goal_visible = np.asarray(mask_npz["goal_visible"][order], dtype=np.int8)
        if mask.shape[0] > num_frames:
            mask = mask[:num_frames]
            goal_uv = goal_uv[:num_frames]
            goal_visible = goal_visible[:num_frames]
        if mask.shape[0] < num_frames:
            raise ValueError(
                f"{bundle_dir.name}: goal mask has {mask.shape[0]} frames, expected {num_frames}"
            )
    else:
        h, w = rgb.shape[1], rgb.shape[2]
        mask = np.zeros((num_frames, h, w), dtype=np.uint8)
        goal_uv = np.full((num_frames, 2), -1.0, dtype=np.float32)
        goal_visible = np.zeros((num_frames,), dtype=np.int8)

    return rgb, mask, goal_uv, goal_visible


def rgb_to_pil_batch(rgb: np.ndarray) -> list[Image.Image]:
    return [Image.fromarray(frame).convert("RGB") for frame in rgb]


def orig_to_dino_px(cx: float, cy: float) -> tuple[float, float]:
    """Map full-frame pixel to DINOv2 center-crop pixel space (224x224)."""
    sx = cx * RESIZE_SIZE / ORIG_SIZE - BORDER
    sy = cy * RESIZE_SIZE / ORIG_SIZE - BORDER
    return sx, sy


def orig_to_patch_coords(cx: float, cy: float) -> tuple[float, float]:
    """Map full-frame pixel (512x512) to fractional 16x16 patch coordinates."""
    col_f = cx * N_PATCHES / ORIG_SIZE
    row_f = cy * N_PATCHES / ORIG_SIZE
    return col_f, row_f


def dino_to_patch(px: float, py: float) -> tuple[float, float]:
    return px / PATCH_SIZE, py / PATCH_SIZE


def heatmap_to_orig_space(heatmap_flat: np.ndarray) -> np.ndarray:
    """Upsample flattened 16x16 heatmap to full 512x512 image space."""
    h16 = np.asarray(heatmap_flat, dtype=np.float32).reshape(N_PATCHES, N_PATCHES)
    h512 = cv2.resize(h16, (ORIG_SIZE, ORIG_SIZE), interpolation=cv2.INTER_LINEAR)
    if h512.max() > 0:
        h512 /= h512.max()
    return h512


def make_gaussian_heatmap(
    col_f: float,
    row_f: float,
    n: int = N_PATCHES,
    sigma: float = GAUSS_SIGMA,
) -> torch.Tensor:
    grid = np.zeros((n, n), dtype=np.float32)
    ci = max(0, min(n - 1, int(round(col_f))))
    ri = max(0, min(n - 1, int(round(row_f))))
    grid[ri, ci] = 1.0
    grid = gaussian_filter(grid, sigma=sigma)
    total = grid.sum()
    if total > 0:
        grid /= total
    return torch.from_numpy(grid.flatten())


def frame_to_heatmap(
    mask_frame: np.ndarray,
    goal_uv: np.ndarray,
    goal_visible: int,
) -> tuple[torch.Tensor, str]:
    """Build a (256,) normalized Gaussian heatmap for one frame."""
    cx = cy = None
    if goal_visible and goal_uv[0] >= 0 and goal_uv[1] >= 0:
        cx, cy = float(goal_uv[0]), float(goal_uv[1])
    else:
        ys, xs = np.where(mask_frame > 127)
        if len(xs) == 0:
            return torch.zeros(N_PATCHES * N_PATCHES), "blank"
        cx, cy = float(xs.mean()), float(ys.mean())

    col_f, row_f = orig_to_patch_coords(cx, cy)
    outside_image = not (0 <= cx < ORIG_SIZE and 0 <= cy < ORIG_SIZE)

    col_c = max(0.0, min(N_PATCHES - 1, col_f))
    row_c = max(0.0, min(N_PATCHES - 1, row_f))
    heatmap = make_gaussian_heatmap(col_c, row_c)

    if outside_image:
        return heatmap, "outside_image"
    return heatmap, "ok"


def aggregate_status(statuses: list[str]) -> str:
    if not statuses or all(s == "blank" for s in statuses):
        return "blank_mask"
    valid = [s for s in statuses if s != "blank"]
    outside = {"outside_crop", "outside_image"}
    if all(s in outside for s in valid):
        return "outside_image"
    if any(s in outside for s in valid):
        return "partial_outside"
    return "ok"


def depth_to_patches(depth: np.ndarray) -> np.ndarray:
    """(H,W) metric depth -> (5,1024) float32 patches (ADT-compatible)."""
    d = np.log1p(np.maximum(depth, 0.0).astype(np.float32))
    d64 = cv2.resize(d, (64, 64), interpolation=cv2.INTER_LINEAR)
    p0 = cv2.resize(d64, (32, 32), interpolation=cv2.INTER_AREA)
    patches = np.stack(
        [
            p0.ravel(),
            d64[:32, :32].ravel(),
            d64[:32, 32:].ravel(),
            d64[32:, :32].ravel(),
            d64[32:, 32:].ravel(),
        ],
        axis=0,
    )
    return patches.astype(np.float32)


def nymeria_intrinsics_tensor(device: torch.device) -> torch.Tensor:
    return torch.tensor([FOCAL, FOCAL, CX, CY], device=device, dtype=torch.float32)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Convert (H,W) float depth to (H,W,3) uint8 viridis-like colormap."""
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    lo, hi = np.percentile(valid, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    norm = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def make_triptych(
    rgb: np.ndarray,
    depth_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    gap: int = 4,
) -> np.ndarray:
    """Side-by-side RGB | depth | goal mask with labels."""
    h, w = rgb.shape[:2]
    mask_rgb = np.stack([mask, mask, mask], axis=-1)
    panel = np.full((h, w * 3 + gap * 2, 3), 255, dtype=np.uint8)
    panel[:, :w] = rgb
    panel[:, w + gap : 2 * w + gap] = depth_rgb
    panel[:, 2 * w + 2 * gap : 3 * w + 2 * gap] = mask_rgb
    for i, label in enumerate(["RGB", "Depth", "Goal mask"]):
        x = i * (w + gap) + 8
        cv2.putText(
            panel,
            label,
            (x, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            label,
            (x, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return panel

