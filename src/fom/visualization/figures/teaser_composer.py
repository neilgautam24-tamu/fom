#!/usr/bin/env python3
"""Interactive teaser composer: scrub frame, set ghosts/POV, then screenshot.

Open the Viser UI, dial in:
  - sequence + frame
  - GT / pred body + independent ghost trails
  - camera POV (orbit sliders or free mouse orbit)
  - PCD source (VIPE RGB / MPS / MPS+ego-RGB)

Then hit **Capture screenshot** to save a PNG of the current view
(``client.get_render`` - WYSIWYG).

Run::

    python -m fom.visualization.figures.teaser_composer --preds preds.pkl --gt gt.pkl \\
        --eval-list keys.json --port 8095
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import viser

from fom.body.smplx_model import load_smplx
from fom.evaluation.io import load_sequences, sequence_key
from fom.visualization.common import find_bundle, host_ip, rotmat_to_wxyz, to_np, viz_cache_dir

VIZ_DATA = viz_cache_dir("teaser").parent
MPS_PCD_DIR = viz_cache_dir("mps_pcd")
VIPE_PCD_DIR = viz_cache_dir("vipe_pcd")
SHOTS_DIR = viz_cache_dir("teaser") / "shots"
MOTION_FPS = 15

# Saturated paper-teaser palette (same as fom.visualization.figures.teaser_render).
GT_COLOR = (51, 199, 102)  # green
PRED_COLOR = (51, 128, 250)  # blue
GT_TRAJ = (26, 158, 71)
PRED_TRAJ = (38, 140, 242)
FRUSTUM_COLOR = (255, 176, 48)
FRUSTUM_DIM = (210, 160, 70)
PRED_GHOST_BASE = (40, 100, 210)
GT_GHOST_BASE = (30, 150, 75)
# Soft scene backdrop - pure white kills pale PCD / traj contrast.
SCENE_BG = (228, 232, 236)


def lerp_rgb(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(a[i] * (1.0 - t) + b[i] * t)) for i in range(3))


def same_hue_fade(base: tuple[int, int, int], ramp: float) -> tuple[int, int, int]:
    """Older (ramp->0) = darker same-hue; newer (ramp->1) = full base.

    Do **not** wash toward white - that erased GT-green vs pred-blue identity.
    """
    ramp = float(np.clip(ramp, 0.0, 1.0))
    dark = tuple(max(0, int(round(c * 0.45))) for c in base)
    return lerp_rgb(dark, base, 0.35 + 0.65 * ramp)


def time_ramp(i: int, n: int) -> float:
    """0 = oldest ghost, 1 = newest (closest to current frame)."""
    if n <= 1:
        return 1.0
    return float(i) / float(n - 1)


def poses_wxyz_pos(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,4) wxyz + (N,3) position from 4x4 c2w stack."""
    traj = np.asarray(traj)
    n = len(traj)
    wxyz = np.stack([rotmat_to_wxyz(traj[i, :3, :3]) for i in range(n)])
    pos = traj[:, :3, 3].astype(np.float32)
    return wxyz, pos


def scrub_pcd_colors(xyz: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop pure-black / near-black holes that read as artifacts on light bg."""
    if xyz.shape[0] == 0:
        return xyz, rgb
    lum = rgb.astype(np.float32).mean(axis=1)
    keep = lum >= 12.0
    if int(keep.sum()) < max(500, xyz.shape[0] // 10):
        return xyz, rgb
    return xyz[keep], rgb[keep]








def add_traj(
    client,
    name: str,
    traj: np.ndarray,
    color: tuple[int, int, int],
    *,
    line_width: float = 5.0,
):
    """Draw trajectory in *same hue* (dark->bright along time). Never fade to yellow."""
    if traj is None or len(traj) < 2:
        return None
    pts = traj[:, :3, 3].astype(np.float32)
    if len(pts) > 160:
        idx = np.linspace(0, len(pts) - 1, 160).round().astype(int)
        pts = pts[idx]
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    n = len(seg)
    cols = np.zeros_like(seg, dtype=np.uint8)
    c_dark = np.array(same_hue_fade(color, 0.0), dtype=np.float32)
    c_bright = np.array(color, dtype=np.float32)
    for i in range(n):
        t = i / max(n - 1, 1)
        cols[i] = (c_dark * (1.0 - t) + c_bright * t).astype(np.uint8)
    return client.scene.add_line_segments(
        name, points=seg, colors=cols, line_width=float(line_width)
    )


def ghost_indices_temporal(frame: int, count: int, stride: int) -> list[int]:
    """Pick up to ``count`` frame indices at a fixed time stride before ``frame``.

    Ghosts land at ``frame - stride, frame - 2*stride, ...`` (oldest first,
    matching the ``same_hue_fade``/``time_ramp`` ordering callers expect),
    clipped to frames that exist. A fixed cadence in time - not walked
    distance - so a paused/slow-moving stretch still gets a ghost trail
    instead of collapsing to nothing.
    """
    if count <= 0 or frame <= 0 or stride <= 0:
        return []
    idxs = [frame - i * stride for i in range(count, 0, -1)]
    return [i for i in idxs if 0 <= i < frame]


def pcd_extent_ratio(xyz: np.ndarray, traj_xyz: np.ndarray) -> float:
    """Spread of ``xyz`` (p90 radius) relative to the camera trajectory's own spread.

    A well-aligned, room-scale scene cloud should sit within a few multiples
    of how far the wearer actually walked. A much larger ratio is the
    signature of a misaligned or outlier-polluted cloud - the exact failure
    mode that motivated ``mps_obs_pcd.CACHE_VERSION`` (bad Umeyama
    correspondence + unfiltered SLAM points could scatter points tens of
    meters from the real scene). Used only as a soft warning, never to drop
    points, so it stays cheap and side-effect-free.
    """
    if xyz.shape[0] < 50 or traj_xyz.shape[0] < 2:
        return 1.0
    center = np.median(traj_xyz, axis=0)
    pcd_r = float(np.percentile(np.linalg.norm(xyz - center, axis=1), 90))
    traj_r = float(np.percentile(np.linalg.norm(traj_xyz - center, axis=1), 90))
    return pcd_r / max(traj_r, 1.0)


def load_pcd_for_key(
    key: str,
    *,
    mode: str,
    mps_dir: Path,
    vipe_dir: Path,
    gt_aria: np.ndarray,
    max_points: int = 150_000,
    force_mps: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    stem = key.replace("/", "_")
    vipe_path = vipe_dir / f"{stem}.npz"
    mps_path = mps_dir / f"{stem}.npz"

    def _load(path: Path, tag: str):
        z = np.load(path)
        xyz = np.asarray(z["xyz"], np.float32)
        rgb = np.asarray(z["rgb"], np.uint8)
        xyz, rgb = scrub_pcd_colors(xyz, rgb)
        if xyz.shape[0] > max_points:
            rng = np.random.default_rng(0)
            sel = rng.choice(xyz.shape[0], max_points, replace=False)
            xyz, rgb = xyz[sel], rgb[sel]
        return xyz, rgb, tag

    def _finalize(result: tuple[np.ndarray, np.ndarray, str]):
        xyz, rgb, tag = result
        ratio = pcd_extent_ratio(xyz, gt_aria[:, :3, 3])
        if ratio > 4.0:
            print(
                f"[pcd] WARNING: {tag} cloud for {key[:60]} spans {ratio:.1f}x "
                "the walked trajectory - likely misaligned/outlier-polluted "
                "(try Reload PCD / Rebuild MPS-RGB)",
                flush=True,
            )
            tag = f"{tag} (wide)"
        return xyz, rgb, tag

    if mode == "vipe":
        if not vipe_path.is_file():
            raise FileNotFoundError(f"No VIPE PCD: {vipe_path.name}")
        return _finalize(_load(vipe_path, "vipe"))

    if mode in ("mps", "mps_rgb"):
        if not mps_path.is_file():
            raise FileNotFoundError(f"No MPS PCD: {mps_path.name}")
        xyz, rgb, tag = _load(mps_path, "mps")
        if mode == "mps_rgb":
            from fom.visualization.pointcloud.mps_rgb import load_or_build_mps_rgb

            cache_dir = VIZ_DATA / "teaser" / "_mps_rgb_viz_cache"
            xyz, rgb, tag = load_or_build_mps_rgb(
                key, xyz, rgb, gt_aria, cache_dir=cache_dir, force=force_mps
            )
            xyz, rgb = scrub_pcd_colors(xyz, rgb)
            if xyz.shape[0] > max_points:
                rng = np.random.default_rng(0)
                sel = rng.choice(xyz.shape[0], max_points, replace=False)
                xyz, rgb = xyz[sel], rgb[sel]
        return _finalize((xyz, rgb, tag))

    # auto -> VIPE then MPS
    if vipe_path.is_file():
        return _finalize(_load(vipe_path, "vipe"))
    if mps_path.is_file():
        return _finalize(_load(mps_path, "mps"))
    return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), "none"


def prepare_motion(key: str, preds: dict, gt: dict, smpl) -> dict:
    from fom.body.smplx_model import evaluate_smplx

    pred = preds[key]
    g = gt[key]
    pred_smpl = {k: torch.as_tensor(to_np(v)) for k, v in pred["smpl_params"].items()}
    gt_smpl = {k: torch.as_tensor(to_np(v)) for k, v in g["smpl_params"].items()}
    with torch.no_grad():
        _, pred_verts, _ = evaluate_smplx(smpl, pred_smpl)
        _, gt_verts, _ = evaluate_smplx(smpl, gt_smpl)
    pred_verts = pred_verts.cpu().numpy()
    gt_verts = gt_verts.cpu().numpy()
    n = min(len(pred_verts), len(gt_verts))
    pred_aria = to_np(pred["camera_T"])[:n]
    gt_aria = to_np(g["camera_T"])[:n]
    # Frustum RGB is recorded at GT Aria poses - keep GT as default cam for
    # textured frustums so they sit on the same frame as the scene PCD.
    gt_wxyz, gt_pos = poses_wxyz_pos(gt_aria)
    pred_wxyz, pred_pos = poses_wxyz_pos(pred_aria)
    panels, thumbs, fov_y = load_ego_rgb_aligned(key, n)
    return {
        "key": key,
        "n": n,
        "pred_verts": pred_verts[:n],
        "gt_verts": gt_verts[:n],
        "pred_aria": pred_aria,
        "gt_aria": gt_aria,
        "gt_cam_pos": gt_pos,
        "gt_cam_wxyz": gt_wxyz,
        "pred_cam_pos": pred_pos,
        "pred_cam_wxyz": pred_wxyz,
        "faces": np.asarray(smpl.faces, dtype=np.int32),
        "panels": panels,
        "thumbs": thumbs,
        "fov_y": fov_y,
    }


def load_ego_rgb_aligned(
    key: str, n_frames: int, *, max_edge: int = 240, max_src_mb: float = 800.0
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Ego RGB panels (2D strip) + frustum thumbs (viser texture), aligned to motion frames.

    Bundle stores CW90 training orientation -> upright needs a 180 deg turn
    (``rot90(k=2)``), the same ``rot_k=2`` used by
    :func:`fom.visualization.pointcloud.mps_rgb.project_ego_rgb_hq`. ``upright`` is
    correct for viser's ``add_camera_frustum(image=...)`` texture. The 2D strip
    needs one more 180 deg turn, because viser's frustum UV mapping and PIL's
    ``Image.fromarray`` compositing differ in vertical orientation by a half turn.
    """
    from PIL import Image

    take = key.split("___")[0]
    bundle = find_bundle(take)
    fov_y = 1.0
    if bundle is None:
        return [], [], fov_y
    tr = np.load(bundle / "trajectory.npz")
    if "rgb_fov_y_rad" in tr.files:
        fov_y = float(tr["rgb_fov_y_rad"][0])
    rgb_path = bundle / "ego_rgb.npz"
    if not rgb_path.is_file():
        return [], [], fov_y
    if rgb_path.stat().st_size / 1e6 > max_src_mb:
        print(f"[rgb] skip large {rgb_path.name}", flush=True)
        return [], [], fov_y
    z = np.load(rgb_path, mmap_mode="r")
    rgb = z["rgb"]
    n_src = int(rgb.shape[0])
    if n_src <= 0 or n_frames <= 0:
        return [], [], fov_y
    idx = np.linspace(0, n_src - 1, n_frames).round().astype(int)
    rotated_cw90 = True
    if "rgb_rotated_cw90" in z.files:
        rotated_cw90 = bool(int(np.asarray(z["rgb_rotated_cw90"]).reshape(-1)[0]))

    def _resize(im: np.ndarray) -> np.ndarray:
        h, w = im.shape[:2]
        if max(h, w) <= max_edge:
            return im
        scale = max_edge / float(max(h, w))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return np.asarray(Image.fromarray(im).resize((nw, nh), Image.Resampling.BILINEAR))

    panels, thumbs = [], []
    for i in idx:
        fr = np.asarray(rgb[int(i)], dtype=np.uint8)
        upright = (
            np.ascontiguousarray(np.rot90(fr, k=2)) if rotated_cw90 else np.ascontiguousarray(fr)
        )
        strip_frame = np.ascontiguousarray(np.rot90(upright, k=2))
        panels.append(_resize(strip_frame))
        thumbs.append(_resize(upright))
    return panels, thumbs, fov_y


def composite_rgb_strip(
    scene_rgb: np.ndarray,
    panels: list[np.ndarray],
    frame_ids: list[int],
    *,
    strip_frac: float = 0.22,
    pad: int = 10,
    label_prefix: str = "t",
    accent: tuple[int, int, int] = FRUSTUM_COLOR,
) -> np.ndarray:
    """Paste a row of ego-RGB frames under the 3D screenshot."""
    from PIL import Image, ImageDraw

    if not panels or not frame_ids:
        return scene_rgb
    # viser PNG transport may return RGBA - drop alpha for compositing.
    scene = np.asarray(scene_rgb)
    if scene.ndim == 3 and scene.shape[2] == 4:
        scene = scene[:, :, :3]
    scene = np.ascontiguousarray(scene, dtype=np.uint8)

    H, W = scene.shape[:2]
    strip_h = max(88, int(round(H * strip_frac)))
    canvas_img = Image.new("RGB", (W, H + strip_h), (236, 238, 242))
    canvas_img.paste(Image.fromarray(scene), (0, 0))

    n = len(frame_ids)
    cell_w = max(1, (W - pad * (n + 1)) // n)
    cell_h = strip_h - 2 * pad - 20
    draw = ImageDraw.Draw(canvas_img)
    # Separator line between 3D view and strip.
    draw.rectangle([0, H, W, H + 2], fill=accent)
    for i, fi in enumerate(frame_ids):
        src = panels[int(np.clip(fi, 0, len(panels) - 1))]
        src = np.asarray(src)
        if src.ndim == 3 and src.shape[2] == 4:
            src = src[:, :, :3]
        im = Image.fromarray(np.ascontiguousarray(src, dtype=np.uint8))
        im.thumbnail((cell_w - 4, cell_h - 4), Image.Resampling.BILINEAR)
        tw, th = im.size
        x0 = pad + i * (cell_w + pad) + (cell_w - tw) // 2
        y0 = H + pad + (cell_h - th) // 2
        # Current (last) frame gets a thicker accent border.
        border = 3 if i == n - 1 else 1
        draw.rectangle(
            [x0 - border - 1, y0 - border - 1, x0 + tw + border, y0 + th + border],
            outline=accent if i == n - 1 else (90, 90, 95),
            width=border,
        )
        canvas_img.paste(im, (x0, y0))
        draw.text(
            (x0, H + strip_h - 16),
            f"{label_prefix}={int(fi)}",
            fill=(25, 25, 28),
        )
    return np.asarray(canvas_img, dtype=np.uint8)


def orbit_eye(
    center: np.ndarray,
    *,
    radius: float,
    elev_deg: float,
    azim_deg: float,
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    up = up / max(np.linalg.norm(up), 1e-9)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    east = np.cross(up, tmp)
    east /= max(np.linalg.norm(east), 1e-9)
    north = np.cross(up, east)
    direction = (
        np.cos(elev) * (np.cos(azim) * east + np.sin(azim) * north) + np.sin(elev) * up
    )
    return center + radius * direction


class TeaserComposer:
    def __init__(
        self,
        client: viser.ClientHandle,
        keys: list[str],
        preds: dict,
        gt: dict,
        smpl,
        *,
        mps_dir: Path,
        vipe_dir: Path,
        shots_dir: Path,
        start_idx: int = 0,
        shared_cache: dict | None = None,
    ) -> None:
        self.client = client
        self.keys = keys
        self.preds = preds
        self.gt = gt
        self.smpl = smpl
        self.mps_dir = mps_dir
        self.vipe_dir = vipe_dir
        self.shots_dir = shots_dir
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        self.faces = np.asarray(smpl.faces, dtype=np.int32)
        self.cache = shared_cache if shared_cache is not None else {}
        self.idx = min(start_idx, len(keys) - 1)
        self.t = 0
        self.playing = False
        self.fps = 12.0
        self._handles: dict = {}
        self._ghost_gt: list = []
        self._ghost_pred: list = []
        self._ghost_frustums: list = []
        self._last_ghost_frame_ids: list[int] = []
        self._pending_idx: int | None = None
        self._switching = False
        self._ready = False
        self._pcd_mode = "auto"
        self._last_shot = ""

        light = np.full((8, 8, 3), SCENE_BG, dtype=np.uint8)
        client.scene.set_background_image(light, format="jpeg")
        client.scene.set_up_direction((0.0, 0.0, 1.0))

        with client.gui.add_folder("Sequence"):
            self.gui_seq = client.gui.add_slider(
                "Index", min=0, max=max(len(keys) - 1, 0), step=1, initial_value=self.idx
            )
            self.gui_name = client.gui.add_text("Name", keys[self.idx] if keys else "")
            self.gui_info = client.gui.add_markdown("Loading...")
            self.gui_pcd_mode = client.gui.add_dropdown(
                "PCD source",
                options=["auto", "vipe", "mps", "mps_rgb"],
                initial_value="vipe",
            )
            client.gui.add_markdown(
                "_**vipe** = dense RGB-D | **mps** = sparse SLAM | "
                "**mps_rgb** = MPS + ego-RGB | **auto** = VIPE else MPS_"
            )
            self.gui_force_mps = client.gui.add_checkbox(
                "Force rebuild MPS-RGB cache", initial_value=False
            )
            self.gui_reload_pcd = client.gui.add_button("Reload PCD")
            self.gui_focus = client.gui.add_button("Focus scene")

        with client.gui.add_folder("Frame"):
            self.gui_t = client.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            self.gui_play = client.gui.add_checkbox("Playing", initial_value=False)
            self.gui_fps = client.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=12)

        with client.gui.add_folder("Bodies"):
            self.gui_gt = client.gui.add_checkbox("Show GT body", initial_value=True)
            self.gui_pred = client.gui.add_checkbox("Show pred body", initial_value=True)
            self.gui_gt_op = client.gui.add_slider(
                "GT opacity", min=0.15, max=1.0, step=0.05, initial_value=0.95
            )
            self.gui_pred_op = client.gui.add_slider(
                "Pred opacity", min=0.15, max=1.0, step=0.05, initial_value=0.95
            )
            self.gui_gt_traj = client.gui.add_checkbox("Show GT traj", initial_value=True)
            self.gui_pred_traj = client.gui.add_checkbox("Show pred traj", initial_value=True)
            self.gui_pcd = client.gui.add_checkbox("Show PCD", initial_value=True)
            self.gui_pcd_size = client.gui.add_slider(
                "Point size", min=0.003, max=0.05, step=0.001, initial_value=0.016
            )
            self.gui_frustum = client.gui.add_checkbox(
                "Show RGB frustums", initial_value=True
            )
            self.gui_frustum_src = client.gui.add_dropdown(
                "Frustum pose",
                options=["gt", "pred"],
                initial_value="gt",
            )
            client.gui.add_markdown(
                "_Frustum pose **gt** (default) matches PCD + recorded ego RGB. "
                "Use **pred** only to visualize estimated camera drift._"
            )
            self.gui_frustum_scale = client.gui.add_slider(
                "Frustum scale", min=0.10, max=0.45, step=0.02, initial_value=0.22
            )
            self.gui_rebuild_mps_rgb = client.gui.add_button("Rebuild MPS-RGB (force)")

        with client.gui.add_folder("Ghosts - GT"):
            self.gui_gt_ghost = client.gui.add_checkbox("Show GT ghosts", initial_value=True)
            self.gui_gt_ghost_n = client.gui.add_slider(
                "GT ghost count", min=0, max=10, step=1, initial_value=4
            )
            self.gui_gt_ghost_op = client.gui.add_slider(
                "GT ghost opacity", min=0.1, max=0.95, step=0.05, initial_value=0.40
            )
            self.gui_gt_ghost_interval = client.gui.add_slider(
                "GT ghost interval (s)", min=0.1, max=3.0, step=0.1, initial_value=0.5
            )

        with client.gui.add_folder("Ghosts - Pred"):
            self.gui_pred_ghost = client.gui.add_checkbox(
                "Show pred ghosts", initial_value=True
            )
            self.gui_pred_ghost_n = client.gui.add_slider(
                "Pred ghost count", min=0, max=12, step=1, initial_value=6
            )
            self.gui_pred_ghost_op = client.gui.add_slider(
                "Pred ghost opacity", min=0.15, max=1.0, step=0.05, initial_value=0.55
            )
            self.gui_pred_ghost_interval = client.gui.add_slider(
                "Pred ghost interval (s)", min=0.1, max=3.0, step=0.1, initial_value=0.4
            )
            client.gui.add_markdown(
                "_Ghosts: **same hue**, darker->brighter over time (not white wash)._ "
                "Spaced at a fixed time interval, not distance."
            )

        with client.gui.add_folder("POV"):
            self.gui_elev = client.gui.add_slider(
                "Elevation (deg)", min=-10.0, max=70.0, step=1.0, initial_value=22.0
            )
            self.gui_azim = client.gui.add_slider(
                "Azimuth (deg)", min=-180.0, max=180.0, step=5.0, initial_value=40.0
            )
            self.gui_radius = client.gui.add_slider(
                "Radius scale", min=0.6, max=3.0, step=0.05, initial_value=1.4
            )
            self.gui_apply_pov = client.gui.add_button("Apply POV (orbit)")
            self.gui_flip_side = client.gui.add_button("Flip side (+180 deg azim)")
            client.gui.add_markdown(
                "_Or orbit freely with the mouse, then capture - screenshot "
                "uses the **current** camera._"
            )

        with client.gui.add_folder("Screenshot"):
            self.gui_w = client.gui.add_slider(
                "Width", min=640, max=1920, step=16, initial_value=1600
            )
            self.gui_h = client.gui.add_slider(
                "Height", min=480, max=1200, step=16, initial_value=1000
            )
            self.gui_fmt = client.gui.add_dropdown(
                "Format", options=["png", "jpeg"], initial_value="png"
            )
            self.gui_rgb_strip = client.gui.add_checkbox(
                "RGB strip in screenshot", initial_value=True
            )
            self.gui_capture = client.gui.add_button("Capture screenshot")
            self.gui_shot_info = client.gui.add_markdown("_No screenshot yet._")
            client.gui.add_markdown(
                "_Bottom strip = ego RGB at ghost + current frames._"
            )

        @self.gui_seq.on_update
        def _(_) -> None:
            self._pending_idx = int(self.gui_seq.value)
            self.gui_info.content = f"Loading sequence {self.gui_seq.value}..."

        @self.gui_t.on_update
        def _(_) -> None:
            if self._ready and not self._switching:
                self.set_frame(int(self.gui_t.value))

        @self.gui_play.on_update
        def _(_) -> None:
            self.playing = bool(self.gui_play.value)

        @self.gui_fps.on_update
        def _(_) -> None:
            self.fps = float(self.gui_fps.value)

        @self.gui_focus.on_click
        def _(_) -> None:
            if self._ready:
                self.focus(self.get_data(self.keys[self.idx]))

        @self.gui_apply_pov.on_click
        def _(_) -> None:
            if self._ready:
                self.apply_pov()

        @self.gui_flip_side.on_click
        def _(_) -> None:
            self.gui_azim.value = float((self.gui_azim.value + 180.0 + 180.0) % 360.0 - 180.0)
            if self._ready:
                self.apply_pov()

        @self.gui_reload_pcd.on_click
        def _(_) -> None:
            key = self.keys[self.idx]
            if key in self.cache:
                self.cache[key].pop("pcd_xyz", None)
                self.cache[key].pop("pcd_rgb", None)
                self.cache[key].pop("pcd_tag", None)
            self._pending_idx = self.idx

        @self.gui_rebuild_mps_rgb.on_click
        def _(_) -> None:
            key = self.keys[self.idx]
            cache = (
                VIZ_DATA
                / "teaser"
                / "_mps_rgb_viz_cache"
                / f"{key.replace('/', '_')}_mpsrgb_v4.npz"
            )
            if cache.is_file():
                cache.unlink()
            if key in self.cache:
                self.cache[key].pop("pcd_xyz", None)
                self.cache[key].pop("pcd_rgb", None)
                self.cache[key].pop("pcd_tag", None)
            self.gui_pcd_mode.value = "mps_rgb"
            self.gui_force_mps.value = True
            self._pending_idx = self.idx

        @self.gui_force_mps.on_update
        def _(_) -> None:
            key = self.keys[self.idx]
            if key in self.cache:
                self.cache[key].pop("pcd_xyz", None)
                self.cache[key].pop("pcd_rgb", None)
                self.cache[key].pop("pcd_tag", None)
            self._pending_idx = self.idx

        @self.gui_pcd_mode.on_update
        def _(_) -> None:
            self._pcd_mode = str(self.gui_pcd_mode.value)
            if self._pcd_mode in ("mps", "mps_rgb") and float(self.gui_pcd_size.value) < 0.014:
                self.gui_pcd_size.value = 0.020
            if self._pcd_mode in ("vipe", "auto") and float(self.gui_pcd_size.value) > 0.018:
                self.gui_pcd_size.value = 0.014
            self.gui_info.content = f"Loading PCD `{self._pcd_mode}`..."
            print(f"[pcd] switch -> {self._pcd_mode}", flush=True)
            key = self.keys[self.idx]
            if key in self.cache:
                self.cache[key].pop("pcd_xyz", None)
                self.cache[key].pop("pcd_rgb", None)
                self.cache[key].pop("pcd_tag", None)
            self._pending_idx = self.idx

        # Frustum toggle / pose source / scale -> rebuild textured frustums.
        @self.gui_frustum.on_update
        def _(_) -> None:
            if self._ready:
                self.refresh_ghosts()
                self.apply_vis()

        @self.gui_frustum_src.on_update
        def _(_) -> None:
            if self._ready:
                self.refresh_ghosts()

        @self.gui_frustum_scale.on_update
        def _(_) -> None:
            if self._ready:
                self.refresh_ghosts()

        @self.gui_capture.on_click
        def _(_) -> None:
            if self._ready:
                self.capture_screenshot()

        for g in (
            self.gui_gt,
            self.gui_pred,
            self.gui_gt_traj,
            self.gui_pred_traj,
            self.gui_pcd,
            self.gui_pcd_size,
            self.gui_gt_op,
            self.gui_pred_op,
        ):
            g.on_update(lambda _: self.apply_vis())

        for g in (
            self.gui_gt_ghost,
            self.gui_gt_ghost_n,
            self.gui_gt_ghost_op,
            self.gui_gt_ghost_interval,
            self.gui_pred_ghost,
            self.gui_pred_ghost_n,
            self.gui_pred_ghost_op,
            self.gui_pred_ghost_interval,
        ):
            g.on_update(lambda _: self.refresh_ghosts() if self._ready else None)

    def get_data(self, key: str) -> dict:
        if key not in self.cache or "gt_verts" not in self.cache[key]:
            print(f"[prep] {key[:80]}...", flush=True)
            motion = prepare_motion(key, self.preds, self.gt, self.smpl)
            self.cache[key] = motion
        data = self.cache[key]
        if "pcd_xyz" not in data:
            data["pcd_xyz"] = np.zeros((0, 3), np.float32)
            data["pcd_rgb"] = np.zeros((0, 3), np.uint8)
            data["pcd_tag"] = "pending"
        return data

    async def process_pending(self) -> None:
        if self._pending_idx is None:
            return
        if self._switching:
            return
        self._switching = True
        try:
            while self._pending_idx is not None:
                idx = self._pending_idx
                self._pending_idx = None
                idx = int(np.clip(idx, 0, len(self.keys) - 1))
                key = self.keys[idx]
                same_seq = (idx == self.idx) and (key in self.cache) and ("gt_verts" in self.cache.get(key, {}))
                mode = str(self.gui_pcd_mode.value)
                force_mps = bool(self.gui_force_mps.value)

                if not same_seq:
                    self._ready = False
                    self.gui_info.content = f"Preparing `{key[:70]}`..."
                    if key not in self.cache or "gt_verts" not in self.cache.get(key, {}):
                        self.cache[key] = await asyncio.to_thread(
                            lambda k=key: prepare_motion(k, self.preds, self.gt, self.smpl)
                        )
                    if self._pending_idx is not None:
                        continue

                self.gui_info.content = f"Loading PCD `{mode}`..."
                try:
                    xyz, rgb, tag = await asyncio.to_thread(
                        lambda k=key, m=mode, fm=force_mps: load_pcd_for_key(
                            k,
                            mode=m,
                            mps_dir=self.mps_dir,
                            vipe_dir=self.vipe_dir,
                            gt_aria=self.cache[k]["gt_aria"],
                            force_mps=fm,
                        )
                    )
                except FileNotFoundError as exc:
                    xyz = np.zeros((0, 3), np.float32)
                    rgb = np.zeros((0, 3), np.uint8)
                    tag = f"missing:{mode}"
                    print(f"[pcd] {exc}", flush=True)

                if self._pending_idx is not None:
                    continue

                self.cache[key]["pcd_xyz"] = xyz
                self.cache[key]["pcd_rgb"] = rgb
                self.cache[key]["pcd_tag"] = tag

                if same_seq:
                    self._swap_pcd(xyz, rgb, tag)
                else:
                    self.rebuild(idx)
        except Exception:
            msg = traceback.format_exc()
            print(msg)
            self.gui_info.content = f"**Failed**\n```\n{msg[-500:]}\n```"
        finally:
            self._switching = False

    def _swap_pcd(self, xyz: np.ndarray, rgb: np.ndarray, tag: str) -> None:
        """Hot-swap just the point cloud without tearing down the scene."""
        old = self._handles.pop("pcd", None)
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
        n_pcd = xyz.shape[0]
        if n_pcd > 0:
            self._handles["pcd"] = self.client.scene.add_point_cloud(
                "/pcd",
                points=xyz,
                colors=rgb,
                point_size=float(self.gui_pcd_size.value),
                point_shape="circle",
                visible=bool(self.gui_pcd.value),
            )
        self.gui_info.content = (
            f"**{self.idx}/{len(self.keys)-1}** | frames **{self.cache[self.keys[self.idx]]['n']}** | "
            f"t=**{self.t}**  \n"
            f"GT (green) | Pred (blue) | PCD `{tag}` (**{n_pcd:,}** pts)  \n"
            f"Dial ghosts/POV, then **Capture screenshot**."
        )
        print(f"[pcd] swapped -> {tag} ({n_pcd:,} pts)", flush=True)

    def _clear_ghosts(self, *, remove: bool = True) -> None:
        if remove:
            for h in self._ghost_gt + self._ghost_pred + getattr(self, "_ghost_frustums", []):
                try:
                    h.remove()
                except Exception:
                    pass
        self._ghost_gt = []
        self._ghost_pred = []
        self._ghost_frustums = []

    def _snapshot_frame_ids(self, data: dict) -> list[int]:
        """Frames for frustums + RGB strip: ghost samples + current.

        Prefer pred ghost times when pred ghosts are on (pred teaser mode);
        else GT ghost times; always include the current scrub frame last.
        """
        frame = self.t
        ids: list[int] = []
        if self.gui_pred_ghost.value and int(self.gui_pred_ghost_n.value) > 0:
            ids.extend(
                ghost_indices_temporal(
                    frame,
                    int(self.gui_pred_ghost_n.value),
                    self._ghost_stride(self.gui_pred_ghost_interval.value),
                )
            )
        if self.gui_gt_ghost.value and int(self.gui_gt_ghost_n.value) > 0:
            # Only add GT ghost times when pred ghosts are off - otherwise the
            # strip/frustum set balloons and overlaps.
            if not (self.gui_pred_ghost.value and int(self.gui_pred_ghost_n.value) > 0):
                ids.extend(
                    ghost_indices_temporal(
                        frame,
                        int(self.gui_gt_ghost_n.value),
                        self._ghost_stride(self.gui_gt_ghost_interval.value),
                    )
                )
        ids.append(frame)
        out: list[int] = []
        for i in ids:
            if i not in out:
                out.append(int(i))
        return out

    def _cam_arrays(self, data: dict) -> tuple[np.ndarray, np.ndarray]:
        """Frustum / ego-RGB pose arrays. Default GT to match PCD + recorded RGB."""
        src = str(getattr(self, "gui_frustum_src", None) and self.gui_frustum_src.value or "gt")
        if src == "pred" and "pred_cam_pos" in data:
            return data["pred_cam_wxyz"], data["pred_cam_pos"]
        return data["gt_cam_wxyz"], data["gt_cam_pos"]

    @staticmethod
    def _ghost_stride(interval_s: float) -> int:
        """Convert a GUI time interval (s) to a frame stride at MOTION_FPS."""
        return max(1, int(round(float(interval_s) * MOTION_FPS)))

    def refresh_ghosts(self) -> None:
        if not self.keys or not self._ready:
            return
        data = self.get_data(self.keys[self.idx])
        self._clear_ghosts(remove=True)
        frame = self.t
        snap_ids = self._snapshot_frame_ids(data)
        self._last_ghost_frame_ids = snap_ids

        if self.gui_gt_ghost.value and int(self.gui_gt_ghost_n.value) > 0:
            idxs = ghost_indices_temporal(
                frame,
                int(self.gui_gt_ghost_n.value),
                self._ghost_stride(self.gui_gt_ghost_interval.value),
            )
            base_op = float(self.gui_gt_ghost_op.value)
            n = max(len(idxs), 1)
            for i, gi in enumerate(idxs):
                ramp = time_ramp(i, n)
                color = same_hue_fade(GT_GHOST_BASE, ramp)
                op = base_op * (0.45 + 0.55 * ramp)
                h = self.client.scene.add_mesh_simple(
                    f"/ghost_gt/{gi}",
                    vertices=data["gt_verts"][gi],
                    faces=self.faces,
                    color=color,
                    opacity=float(np.clip(op, 0.10, 0.90)),
                    cast_shadow=False,
                    receive_shadow=False,
                )
                self._ghost_gt.append(h)

        if self.gui_pred_ghost.value and int(self.gui_pred_ghost_n.value) > 0:
            idxs = ghost_indices_temporal(
                frame,
                int(self.gui_pred_ghost_n.value),
                self._ghost_stride(self.gui_pred_ghost_interval.value),
            )
            base_op = float(self.gui_pred_ghost_op.value)
            n = max(len(idxs), 1)
            for i, gi in enumerate(idxs):
                ramp = time_ramp(i, n)
                color = same_hue_fade(PRED_GHOST_BASE, ramp)
                op = base_op * (0.45 + 0.55 * ramp)
                h = self.client.scene.add_mesh_simple(
                    f"/ghost_pred/{gi}",
                    vertices=data["pred_verts"][gi],
                    faces=self.faces,
                    color=color,
                    opacity=float(np.clip(op, 0.12, 0.92)),
                    cast_shadow=False,
                    receive_shadow=False,
                )
                self._ghost_pred.append(h)

        if bool(self.gui_frustum.value) and data.get("thumbs"):
            thumbs = data["thumbs"]
            aspect = thumbs[0].shape[1] / max(thumbs[0].shape[0], 1)
            cam_wxyz, cam_pos = self._cam_arrays(data)
            base_scale = float(self.gui_frustum_scale.value)
            n_snap = max(len(snap_ids), 1)
            for si, gi in enumerate(snap_ids):
                gi = int(np.clip(gi, 0, data["n"] - 1))
                is_cur = gi == frame
                ramp = time_ramp(si, n_snap)
                scale = base_scale * (1.25 if is_cur else 0.72 + 0.20 * ramp)
                color = FRUSTUM_COLOR if is_cur else same_hue_fade(FRUSTUM_DIM, ramp)
                fr = self.client.scene.add_camera_frustum(
                    f"/frustum/{gi}",
                    fov=float(data.get("fov_y", 1.0)),
                    aspect=float(aspect),
                    scale=float(scale),
                    line_width=2.4 if is_cur else 1.4,
                    color=color,
                    image=thumbs[gi],
                    wxyz=cam_wxyz[gi],
                    position=cam_pos[gi],
                )
                self._ghost_frustums.append(fr)

    def _remove_handle(self, name: str) -> None:
        h = self._handles.pop(name, None)
        if h is None:
            return
        try:
            h.remove()
        except Exception:
            pass

    def _sync_body_mesh(
        self,
        name: str,
        path: str,
        verts: np.ndarray,
        color: tuple[int, int, int],
        *,
        show: bool,
        opacity: float,
    ) -> None:
        """Show/hide body mesh. Unchecked -> remove (not just hide) so it cannot linger."""
        if not show:
            self._remove_handle(name)
            return
        verts = np.asarray(verts, dtype=np.float32)
        existing = self._handles.get(name)
        if existing is not None:
            try:
                existing.vertices = verts
                existing.color = color
                existing.opacity = float(opacity)
                existing.visible = True
                return
            except Exception:
                self._remove_handle(name)
        self._handles[name] = self.client.scene.add_mesh_simple(
            path,
            vertices=verts,
            faces=self.faces,
            color=color,
            opacity=float(opacity),
            visible=True,
            side="double",
            cast_shadow=False,
            receive_shadow=False,
        )

    def apply_vis(self) -> None:
        if not self._ready:
            return
        h = self._handles
        data = self.get_data(self.keys[self.idx]) if self.keys else None
        t = min(self.t, data["n"] - 1) if data and data["n"] > 0 else 0

        def _set(handle, **kwargs):
            if handle is None:
                return
            try:
                for k, v in kwargs.items():
                    setattr(handle, k, v)
            except Exception:
                pass

        if data is not None and data["n"] > 0:
            self._sync_body_mesh(
                "gt_mesh",
                "/gt_body",
                data["gt_verts"][t],
                GT_COLOR,
                show=bool(self.gui_gt.value),
                opacity=float(self.gui_gt_op.value),
            )
            self._sync_body_mesh(
                "pred_mesh",
                "/pred_body",
                data["pred_verts"][t],
                PRED_COLOR,
                show=bool(self.gui_pred.value),
                opacity=float(self.gui_pred_op.value),
            )
        _set(h.get("gt_traj"), visible=bool(self.gui_gt_traj.value))
        _set(h.get("pred_traj"), visible=bool(self.gui_pred_traj.value))
        _set(
            h.get("pcd"),
            visible=bool(self.gui_pcd.value),
            point_size=float(self.gui_pcd_size.value),
        )
        show_f = bool(getattr(self, "gui_frustum", None) and self.gui_frustum.value)
        for fr in getattr(self, "_ghost_frustums", []):
            _set(fr, visible=show_f)

    def focus(self, data: dict) -> None:
        t = min(self.t, data["n"] - 1)
        pts = np.concatenate(
            [data["gt_verts"][t], data["pred_verts"][t], data["gt_aria"][:, :3, 3]],
            axis=0,
        )
        center = np.median(pts, axis=0)
        extent = float(np.percentile(np.linalg.norm(pts - center, axis=1), 90))
        extent = max(extent, 1.5)
        self.client.camera.position = center + np.array([0.0, -1.8 * extent, 0.7 * extent])
        self.client.camera.look_at = center
        self.client.camera.up_direction = (0.0, 0.0, 1.0)

    def apply_pov(self) -> None:
        data = self.get_data(self.keys[self.idx])
        t = min(self.t, data["n"] - 1)
        content = np.concatenate(
            [
                data["gt_verts"][t],
                data["pred_verts"][t],
                data["gt_aria"][:, :3, 3],
                data["pred_aria"][:, :3, 3],
            ],
            axis=0,
        )
        center = 0.55 * np.median(content, axis=0) + 0.45 * data["gt_aria"][:, :3, 3].mean(0)
        extent = float(np.percentile(np.linalg.norm(content - center, axis=1), 92))
        radius = max(extent, 1.5) * float(self.gui_radius.value)
        eye = orbit_eye(
            center,
            radius=radius,
            elev_deg=float(self.gui_elev.value),
            azim_deg=float(self.gui_azim.value),
        )
        self.client.camera.position = eye
        self.client.camera.look_at = center
        self.client.camera.up_direction = (0.0, 0.0, 1.0)

    def rebuild(self, idx: int) -> None:
        try:
            self._rebuild_inner(idx)
            self._ready = True
            print(f"[rebuild] idx={idx} ready, handles={list(self._handles.keys())}", flush=True)
        except Exception:
            msg = traceback.format_exc()
            print(msg, flush=True)
            self.gui_info.content = f"**Rebuild failed**\n```\n{msg[-500:]}\n```"

    def _rebuild_inner(self, idx: int) -> None:
        self.idx = int(np.clip(idx, 0, len(self.keys) - 1))
        key = self.keys[self.idx]
        data = self.get_data(key)
        self.t = min(self.t, data["n"] - 1)
        # Drop handle refs before reset - reset removes all scene nodes.
        self._handles = {}
        self._clear_ghosts(remove=False)
        self.client.scene.reset()
        light = np.full((8, 8, 3), SCENE_BG, dtype=np.uint8)
        self.client.scene.set_background_image(light, format="jpeg")
        self.client.scene.set_up_direction((0.0, 0.0, 1.0))

        self.gui_name.value = key
        self.gui_t.max = max(data["n"] - 1, 0)
        # Avoid cascading on_update during rebuild.
        prev_ready = self._ready
        self._ready = False
        self.gui_t.value = self.t
        self._ready = prev_ready
        tag = data.get("pcd_tag", "?")
        n_pcd = int(data.get("pcd_xyz", np.zeros((0, 3))).shape[0])
        self.gui_info.content = (
            f"**{self.idx}/{len(self.keys)-1}** | frames **{data['n']}** | "
            f"t={self.t}  \n"
            f"GT (green) | Pred (blue) | PCD `{tag}` (**{n_pcd:,}** pts)  \n"
            f"Dial ghosts/POV, then **Capture screenshot**."
        )

        if n_pcd > 0:
            self._handles["pcd"] = self.client.scene.add_point_cloud(
                "/pcd",
                points=data["pcd_xyz"],
                colors=data["pcd_rgb"],
                point_size=float(self.gui_pcd_size.value),
                point_shape="circle",
            )

        self._handles["gt_traj"] = add_traj(
            self.client, "/gt_traj", data["gt_aria"], GT_TRAJ, line_width=5.5
        )
        self._handles["pred_traj"] = add_traj(
            self.client, "/pred_traj", data["pred_aria"], PRED_TRAJ, line_width=4.5
        )
        # Create body meshes directly (apply_vis will update/hide later).
        t = min(self.t, data["n"] - 1) if data["n"] > 0 else 0
        if data["n"] > 0:
            if bool(self.gui_gt.value):
                self._handles["gt_mesh"] = self.client.scene.add_mesh_simple(
                    "/gt_body",
                    vertices=data["gt_verts"][t],
                    faces=self.faces,
                    color=GT_COLOR,
                    opacity=float(self.gui_gt_op.value),
                    side="double",
                    cast_shadow=False,
                    receive_shadow=False,
                )
            if bool(self.gui_pred.value):
                self._handles["pred_mesh"] = self.client.scene.add_mesh_simple(
                    "/pred_body",
                    vertices=data["pred_verts"][t],
                    faces=self.faces,
                    color=PRED_COLOR,
                    opacity=float(self.gui_pred_op.value),
                    side="double",
                    cast_shadow=False,
                    receive_shadow=False,
                )
        self._ready = True
        self.refresh_ghosts()
        self.apply_vis()
        self.apply_pov()

    def set_frame(self, ft: int) -> None:
        data = self.get_data(self.keys[self.idx])
        if data["n"] == 0:
            return
        self.t = int(np.clip(ft, 0, data["n"] - 1))
        # Update body verts in place (or remove if unchecked) - no remove+readd race.
        self._sync_body_mesh(
            "gt_mesh",
            "/gt_body",
            data["gt_verts"][self.t],
            GT_COLOR,
            show=bool(self.gui_gt.value),
            opacity=float(self.gui_gt_op.value),
        )
        self._sync_body_mesh(
            "pred_mesh",
            "/pred_body",
            data["pred_verts"][self.t],
            PRED_COLOR,
            show=bool(self.gui_pred.value),
            opacity=float(self.gui_pred_op.value),
        )
        self.refresh_ghosts()
        self.apply_vis()
        tag = data.get("pcd_tag", "?")
        self.gui_info.content = (
            f"**{self.idx}/{len(self.keys)-1}** | frames **{data['n']}** | "
            f"t=**{self.t}**  \n"
            f"GT (green) | Pred (blue) | PCD `{tag}`  \n"
            f"Dial ghosts/POV, then **Capture screenshot**."
        )

    def capture_screenshot(self) -> None:
        key = self.keys[self.idx]
        data = self.get_data(key)
        w = int(self.gui_w.value)
        h = int(self.gui_h.value)
        fmt = str(self.gui_fmt.value)
        self.gui_shot_info.content = f"_Capturing {w}x{h}..._"
        try:
            img = self.client.get_render(
                height=h,
                width=w,
                transport_format="png" if fmt == "png" else "jpeg",
            )
        except Exception as exc:
            self.gui_shot_info.content = f"**Capture failed:** `{exc}`"
            print(f"[shot] failed: {exc}", flush=True)
            return

        # Composite ego-RGB strip for ghost + current snapshot times.
        if bool(getattr(self, "gui_rgb_strip", None) and self.gui_rgb_strip.value):
            try:
                panels = data.get("panels") or []
                frame_ids = list(self._last_ghost_frame_ids) or [self.t]
                if not panels:
                    panels, thumbs, fov_y = load_ego_rgb_aligned(key, data["n"])
                    data["panels"] = panels
                    data["thumbs"] = thumbs
                    data["fov_y"] = fov_y
                if panels:
                    img = composite_rgb_strip(img, panels, frame_ids)
            except Exception as exc:
                print(f"[shot] rgb strip failed (saving scene only): {exc}", flush=True)

        # Ensure writable RGB uint8 for PIL.
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img = np.ascontiguousarray(img, dtype=np.uint8)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{key.replace('/', '_')}_t{self.t}_{ts}"
        out = self.shots_dir / f"{stem}.{fmt if fmt == 'png' else 'jpg'}"
        from PIL import Image

        try:
            Image.fromarray(img).save(out)
        except Exception as exc:
            self.gui_shot_info.content = f"**Save failed:** `{exc}`"
            print(f"[shot] save failed: {exc}", flush=True)
            return
        meta = {
            "key": key,
            "frame": self.t,
            "n_frames": data["n"],
            "pcd_tag": data.get("pcd_tag"),
            "snapshot_frames": list(self._last_ghost_frame_ids),
            "gt_ghosts": bool(self.gui_gt_ghost.value),
            "gt_ghost_count": int(self.gui_gt_ghost_n.value),
            "pred_ghosts": bool(self.gui_pred_ghost.value),
            "pred_ghost_count": int(self.gui_pred_ghost_n.value),
            "elev": float(self.gui_elev.value),
            "azim": float(self.gui_azim.value),
            "radius_scale": float(self.gui_radius.value),
            "camera_position": np.asarray(self.client.camera.position).tolist(),
            "camera_wxyz": np.asarray(self.client.camera.wxyz).tolist(),
            "camera_fov": float(self.client.camera.fov),
            "rgb_strip": bool(getattr(self, "gui_rgb_strip", None) and self.gui_rgb_strip.value),
            "frustum_pose": str(self.gui_frustum_src.value),
            "frustum_scale": float(self.gui_frustum_scale.value),
            "out": str(out),
            "size": [int(img.shape[1]), int(img.shape[0])],
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        self._last_shot = str(out)
        self.gui_shot_info.content = (
            f"**Saved** `{out.name}`  \n`{out.parent}/`  \n"
            f"{img.shape[1]}x{img.shape[0]}"
        )
        print(f"[shot] {out}", flush=True)

    def tick(self) -> None:
        if self.playing and self._ready and self.keys:
            data = self.get_data(self.keys[self.idx])
            if data["n"] > 0:
                self.gui_t.value = (self.t + 1) % data["n"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--mps-pcd-dir", type=Path, default=MPS_PCD_DIR)
    parser.add_argument("--vipe-pcd-dir", type=Path, default=VIPE_PCD_DIR)
    parser.add_argument("--eval-list", type=Path, required=True, help='JSON list of {"pred_key": ...}')
    parser.add_argument("--shots-dir", type=Path, default=SHOTS_DIR)
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--seq", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--preload", type=int, default=1, help="Preload N sequences")
    args = parser.parse_args()


    items = json.loads(Path(args.eval_list).read_text())
    keys = [sequence_key(x["pred_key"]) for x in items]
    print(f"Loading preds...", flush=True)
    preds = load_sequences(args.preds)
    print(f"Loading GT...", flush=True)
    gt = load_sequences(args.gt)
    keys = [k for k in keys if k in preds and k in gt]
    print(f"Sequences: {len(keys)}", flush=True)

    print("Loading SMPL-X...", flush=True)
    smpl = load_smplx().to(torch.device(args.device))

    shared: dict = {}
    preload_n = max(1, min(args.preload, len(keys)))
    start = min(args.seq, len(keys) - 1)
    for i in range(preload_n):
        k = keys[(start + i) % len(keys)]
        t0 = time.time()
        shared[k] = prepare_motion(k, preds, gt, smpl)
        print(f"  preload {k[:55]}... {time.time()-t0:.1f}s", flush=True)

    server = viser.ViserServer(host=args.host, port=args.port, verbose=False)
    lan = host_ip()
    print(f"Teaser composer -> http://127.0.0.1:{args.port}", flush=True)
    print(f"                 http://{lan}:{args.port}", flush=True)
    print(f"Shots dir: {args.shots_dir}", flush=True)
    viewers: dict[int, TeaserComposer] = {}

    @server.on_client_connect
    async def _(client: viser.ClientHandle) -> None:
        viewers.pop(client.client_id, None)
        await asyncio.sleep(0.05)
        viewer = TeaserComposer(
            client,
            keys,
            preds,
            gt,
            smpl,
            mps_dir=args.mps_pcd_dir,
            vipe_dir=args.vipe_pcd_dir,
            shots_dir=args.shots_dir,
            start_idx=start,
            shared_cache=shared,
        )
        viewers[client.client_id] = viewer
        viewer._pending_idx = start
        while True:
            await viewer.process_pending()
            viewer.tick()
            await asyncio.sleep(1.0 / max(viewer.fps, 1.0))

    while True:
        time.sleep(3600.0)


if __name__ == "__main__":
    main()
