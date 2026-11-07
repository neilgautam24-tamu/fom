#!/usr/bin/env python3
"""Viser viewer: ground-truth vs predicted body, camera trajectories, frustum and scene point cloud.

Loads:
  - a prediction file from ``fom-predict`` and the matching ground truth from ``fom-gt``
  - ego RGB for the frustum / RGB panel (RGB cache, exported MP4 or the sequence bundle)
  - optional fused scene point clouds ``<pcd-dir>/<key>.npz`` from
    :mod:`fom.visualization.pointcloud.vipe_fusion`

    python -m fom.visualization.viewers.gt_vs_pred --preds preds.pkl --gt gt.pkl --port 8095
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import viser
from PIL import Image

from fom.body.smplx_model import evaluate_smplx, load_smplx
from fom.evaluation.io import load_sequences
from fom.visualization.common import (
    find_bundle,
    host_ip,
    resize_max_edge,
    rot180,
    rotmat_to_wxyz,
    to_np,
    viz_cache_dir,
)

GT_COLOR = (70, 210, 110)
PRED_COLOR = (70, 150, 255)
GT_TRAJ = (40, 180, 90)
PRED_TRAJ = (80, 180, 255)
FRUSTUM_COLOR = (255, 200, 80)














def thumb(rgb: np.ndarray, size: int = 96) -> np.ndarray:
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    im.thumbnail((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im)


def add_traj(client, name: str, traj: np.ndarray, color: tuple[int, int, int], *, width: float = 8.0):
    """Draw a camera path: thick spline + segment lines + waypoint dots."""
    if traj is None or len(traj) < 2:
        return None
    pts = np.asarray(traj, dtype=np.float32)[:, :3, 3].copy()
    pts[:, 2] += 0.04  # lift above floor points so the path isn't buried
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    n = segs.shape[0]
    cols = np.zeros((n, 2, 3), dtype=np.uint8)
    cols[...] = np.asarray(color, np.uint8)
    handles = []
    handles.append(
        client.scene.add_line_segments(name, points=segs, colors=cols, line_width=width)
    )
    try:
        handles.append(
            client.scene.add_spline_catmull_rom(
                f"{name}_spline",
                points=pts,
                line_width=width,
                color=color,
                segments=max(4 * len(pts), 32),
            )
        )
    except Exception:
        pass
    handles.append(
        client.scene.add_point_cloud(
            f"{name}_pts",
            points=pts,
            colors=np.tile(np.asarray(color, np.uint8), (len(pts), 1)),
            point_size=0.045,
            point_shape="circle",
        )
    )
    return handles


def load_rgb_from_mp4(mp4: Path, n_frames: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read the already-exported 15fps MP4 (fast) instead of the 90MB ego_rgb.npz."""
    import cv2

    cap = cv2.VideoCapture(str(mp4))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return [], []
    rgb = np.stack(frames, axis=0)
    if len(rgb) > n_frames:
        idx = np.linspace(0, len(rgb) - 1, n_frames).round().astype(int)
        rgb = rgb[idx]
    elif len(rgb) < n_frames:
        rgb = np.concatenate([rgb, np.repeat(rgb[-1:], n_frames - len(rgb), axis=0)], axis=0)
    panels = [resize_max_edge(f, 360) for f in rgb]
    thumbs = [rot180(thumb(f, 80)) for f in rgb]
    return panels, thumbs


def load_rgb_window(bundle: Path, n_frames: int) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Fallback: ego_rgb.npz if no MP4 is available."""
    ego = np.load(bundle / "ego_rgb.npz", mmap_mode="r")
    rgb = np.asarray(ego["rgb"], dtype=np.uint8)
    if len(rgb) > n_frames:
        idx = np.linspace(0, len(rgb) - 1, n_frames).round().astype(int)
        rgb = rgb[idx]
    else:
        rgb = rgb[:n_frames]
    while len(rgb) < n_frames:
        rgb = np.concatenate([rgb, rgb[-1:]], axis=0)

    fov_y = 1.0
    traj_path = bundle / "trajectory.npz"
    if traj_path.is_file():
        tr = np.load(traj_path)
        if "rgb_fov_y_rad" in tr.files:
            fov_y = float(tr["rgb_fov_y_rad"][0])

    panels = [resize_max_edge(f, 360) for f in rgb]
    thumbs = [rot180(thumb(f, 80)) for f in rgb]
    return panels, thumbs, fov_y


class ReconViewer:
    def __init__(
        self,
        client: viser.ClientHandle,
        keys: list[str],
        preds: dict,
        gt: dict,
        smpl,
        pcd_dir: Path,
        start_idx: int = 0,
        shared_cache: dict[str, dict] | None = None,
        defer_rebuild: bool = False,
        video_manifest: Path | None = None,
    ) -> None:
        self.client = client
        self.keys = keys
        self.preds = preds
        self.gt = gt
        self.smpl = smpl
        self.pcd_dir = pcd_dir
        self.faces = np.asarray(smpl.faces)
        self.cache = shared_cache if shared_cache is not None else {}
        self.idx = min(start_idx, len(keys) - 1)
        self.t = 0
        self.playhead = 0.0
        self.last_tick_time = time.monotonic()
        self.source_fps = 15.0
        self.playing = False
        self.fps = 10.0
        self._handles = {}
        self._mp4 = {}
        man = video_manifest
        if man is not None and man.is_file():
            for row in json.loads(man.read_text()):
                self._mp4[row["pred_key"]] = Path(row["mp4"])

        with client.gui.add_folder("Sequence"):
            self.gui_seq = client.gui.add_slider(
                "Index", min=0, max=len(keys) - 1, step=1, initial_value=self.idx
            )
            self.gui_name = client.gui.add_text("Name", keys[self.idx][:80])
            self.gui_info = client.gui.add_markdown("")
            self.gui_pcd_only = client.gui.add_checkbox(
                "Only sequences with VIPE PCD", initial_value=False
            )
        with client.gui.add_folder("Playback"):
            self.gui_t = client.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            self.gui_play = client.gui.add_checkbox("Playing", initial_value=False)
            self.gui_fps = client.gui.add_slider(
                "Render FPS", min=1, max=15, step=1, initial_value=10
            )
        with client.gui.add_folder("Display"):
            self.gui_gt = client.gui.add_checkbox("Show GT body", initial_value=True)
            self.gui_pred = client.gui.add_checkbox("Show pred body", initial_value=True)
            self.gui_frustum = client.gui.add_checkbox("Show frustum", initial_value=True)
            self.gui_pcd = client.gui.add_checkbox("Show VIPE PCD", initial_value=True)
            self.gui_pcd_size = client.gui.add_slider(
                "Point size", min=0.005, max=0.05, step=0.005, initial_value=0.015
            )
            self.gui_rgb_panel = client.gui.add_checkbox("Show RGB panel", initial_value=True)

        placeholder = np.zeros((64, 64, 3), np.uint8)
        self.rgb_panel = client.gui.add_image(placeholder, label="Ego RGB", order=-100.0)

        @self.gui_seq.on_update
        async def _(_) -> None:
            idx = int(self.gui_seq.value)
            key = self._active_keys[int(np.clip(idx, 0, len(self._active_keys) - 1))]
            data = await asyncio.to_thread(self.get_data, key)
            await asyncio.to_thread(self._load_rgb, data)
            self.rebuild(idx)

        @self.gui_t.on_update
        def _(event) -> None:
            if event.client_id is not None:
                self.playhead = float(self.gui_t.value)
            self.set_frame(int(self.gui_t.value))

        @self.gui_play.on_update
        def _(_) -> None:
            self.playing = bool(self.gui_play.value)
            self.last_tick_time = time.monotonic()

        @self.gui_fps.on_update
        def _(_) -> None:
            self.fps = float(self.gui_fps.value)

        @self.gui_pcd_only.on_update
        def _(_) -> None:
            self._refresh_seq_list()

        for g in (
            self.gui_gt,
            self.gui_pred,
            self.gui_frustum,
            self.gui_pcd,
            self.gui_pcd_size,
            self.gui_rgb_panel,
        ):
            g.on_update(lambda _: self.apply_vis())

        self._active_keys = list(keys)
        if not defer_rebuild:
            self.rebuild(self.idx)

    def _pcd_path(self, key: str) -> Path:
        return self.pcd_dir / f"{key.replace('/', '_')}.npz"

    def _refresh_seq_list(self) -> None:
        if self.gui_pcd_only.value:
            self._active_keys = [k for k in self.keys if self._pcd_path(k).is_file()]
            if not self._active_keys:
                self.gui_info.content = "No PCD caches yet - wait for VIPE job / fuse."
                return
        else:
            self._active_keys = list(self.keys)
        self.gui_seq.max = max(len(self._active_keys) - 1, 0)
        self.gui_seq.value = 0
        self.rebuild(0)

    def get_data(self, key: str) -> dict:
        if key in self.cache:
            return self.cache[key]


        pred = self.preds[key]
        gt = self.gt[key]
        pred_smpl = {k: torch.as_tensor(to_np(v)) for k, v in pred["smpl_params"].items()}
        gt_smpl = {k: torch.as_tensor(to_np(v)) for k, v in gt["smpl_params"].items()}
        # Overlay GT shape so pose/traj errors are visible (COND_BETAS=false).
        if "betas" in gt_smpl and "betas" in pred_smpl:
            pred_smpl = dict(pred_smpl)
            pred_smpl["betas"] = gt_smpl["betas"]
        with torch.no_grad():
            _, pred_verts, _ = evaluate_smplx(self.smpl, pred_smpl)
            _, gt_verts, _ = evaluate_smplx(self.smpl, gt_smpl)
        pred_verts = pred_verts.cpu().numpy()
        gt_verts = gt_verts.cpu().numpy()
        n = min(len(pred_verts), len(gt_verts))
        pred_verts, gt_verts = pred_verts[:n], gt_verts[:n]

        pred_aria = to_np(pred["camera_T"])[:n]
        gt_aria = to_np(gt["camera_T"])[:n]

        take = key.split("___")[0]
        panels, thumbs, fov_y = [], [], 1.0
        # RGB is loaded after the first scene paint so the websocket can open quickly.

        cam_wxyz = np.stack([rotmat_to_wxyz(pred_aria[i, :3, :3]) for i in range(n)])
        cam_pos = pred_aria[:, :3, 3].astype(np.float32)

        pcd_xyz = np.zeros((0, 3), np.float32)
        pcd_rgb = np.zeros((0, 3), np.uint8)
        pcd_path = self._pcd_path(key)
        if pcd_path.is_file():
            z = np.load(pcd_path)
            pcd_xyz = np.asarray(z["xyz"], np.float32)
            pcd_rgb = np.asarray(z["rgb"], np.uint8)
            max_pts = 80_000
            if pcd_xyz.shape[0] > max_pts:
                rng = np.random.default_rng(0)
                idx = rng.choice(pcd_xyz.shape[0], max_pts, replace=False)
                pcd_xyz, pcd_rgb = pcd_xyz[idx], pcd_rgb[idx]

        data = {
            "key": key,
            "n": n,
            "pred_verts": pred_verts,
            "gt_verts": gt_verts,
            "pred_aria": pred_aria,
            "gt_aria": gt_aria,
            "cam_pos": cam_pos,
            "cam_wxyz": cam_wxyz,
            "panels": panels,
            "thumbs": thumbs,
            "fov_y": fov_y,
            "pcd_xyz": pcd_xyz,
            "pcd_rgb": pcd_rgb,
            "has_pcd": pcd_xyz.shape[0] > 0,
            "_need_rgb": True,
            "take": take,
        }
        self.cache[key] = data
        return data

    def _load_rgb(self, data: dict) -> None:
        if not data.get("_need_rgb"):
            return
        data["_need_rgb"] = False
        n = data["n"]
        key = data["key"]
        panels, thumbs, fov_y = [], [], float(data.get("fov_y") or 1.0)
        mp4 = self._mp4.get(key)
        try:
            cache = viz_cache_dir("rgb") / f"{key.replace('/', '_')}.npz"
            if cache.is_file():
                z = np.load(cache)
                panels_arr = np.asarray(z["panels"], dtype=np.uint8)
                if len(panels_arr) > n:
                    panels_arr = panels_arr[:n]
                panels = [panels_arr[i] for i in range(len(panels_arr))]
                thumbs = [rot180(thumb(p, 80)) for p in panels]
                if "fov_y" in z.files:
                    fov_y = float(np.asarray(z["fov_y"]).reshape(-1)[0])
            elif mp4 is not None and mp4.is_file():
                panels, thumbs = load_rgb_from_mp4(mp4, n)
            elif (bundle := find_bundle(key)) is not None:
                panels, thumbs, fov_y = load_rgb_window(bundle, n)
            else:
                print(f"[rgb] {key}: no RGB found (cache, video manifest or bundle)", flush=True)
        except Exception as exc:
            print(f"[rgb] {key}: {exc}", flush=True)
            traceback.print_exc()
            panels, thumbs = [], []
        data["panels"] = panels
        data["thumbs"] = thumbs
        data["fov_y"] = fov_y

    def _set_traj_visible(self, name: str, visible: bool) -> None:
        h = self._handles.get(name)
        if h is None:
            return
        items = h if isinstance(h, list) else [h]
        for item in items:
            if item is not None:
                item.visible = visible

    def apply_vis(self) -> None:
        h = self._handles
        if h.get("gt_mesh") is not None:
            h["gt_mesh"].visible = bool(self.gui_gt.value)
        if h.get("pred_mesh") is not None:
            h["pred_mesh"].visible = bool(self.gui_pred.value)
        if h.get("frustum") is not None:
            h["frustum"].visible = bool(self.gui_frustum.value)
        if h.get("frame") is not None:
            h["frame"].visible = bool(self.gui_frustum.value)
        if h.get("pcd") is not None:
            h["pcd"].visible = bool(self.gui_pcd.value)
            h["pcd"].point_size = float(self.gui_pcd_size.value)
        if self.rgb_panel is not None:
            self.rgb_panel.visible = bool(self.gui_rgb_panel.value)

    def focus(self, data: dict) -> None:
        body_c = data["pred_verts"][0].mean(axis=0)
        # Third-person view of the human so GT/pred bodies are in frame with the room.
        self.client.camera.look_at = body_c
        self.client.camera.position = body_c + np.array([2.2, -2.8, 1.6])
        self.client.camera.up_direction = (0.0, 0.0, 1.0)

    def rebuild(self, idx: int) -> None:
        if not self._active_keys:
            return
        self.idx = int(np.clip(idx, 0, len(self._active_keys) - 1))
        key = self._active_keys[self.idx]
        data = self.get_data(key)
        self.t = 0
        self.playhead = 0.0
        self.last_tick_time = time.monotonic()
        self.client.scene.reset()
        self._handles = {}

        self.gui_name.value = key[:100]
        self.gui_t.max = max(data["n"] - 1, 0)
        self.gui_t.value = 0
        pcd_note = f"VIPE PCD **{data['pcd_xyz'].shape[0]:,}** pts" if data["has_pcd"] else "VIPE PCD: _not cached yet_"
        self.gui_info.content = (
            f"**{self.idx}/{len(self._active_keys)-1}**  \n"
            f"frames: {data['n']}  \n"
            f"GT (green) | Pred (blue) | frustum (yellow)  \n"
            f"{pcd_note}"
        )

        if data["has_pcd"]:
            self._handles["pcd"] = self.client.scene.add_point_cloud(
                "/vipe_pcd",
                points=data["pcd_xyz"],
                colors=data["pcd_rgb"],
                point_size=float(self.gui_pcd_size.value),
                point_shape="circle",
            )

        # Bodies first so a traj/frustum failure still leaves the human visible.
        self._handles["gt_mesh"] = self.client.scene.add_mesh_simple(
            "/gt_body",
            vertices=data["gt_verts"][0],
            faces=self.faces,
            color=GT_COLOR,
            opacity=0.55,
        )
        self._handles["pred_mesh"] = self.client.scene.add_mesh_simple(
            "/pred_body",
            vertices=data["pred_verts"][0],
            faces=self.faces,
            color=PRED_COLOR,
            opacity=0.85,
        )

        thumb0 = data["thumbs"][0] if data["thumbs"] else None
        aspect = (thumb0.shape[1] / max(thumb0.shape[0], 1)) if thumb0 is not None else 1.0
        self._handles["frame"] = self.client.scene.add_frame(
            "/cam_axes",
            wxyz=data["cam_wxyz"][0],
            position=data["cam_pos"][0],
            axes_length=0.18,
            axes_radius=0.008,
        )
        self._handles["frustum"] = self.client.scene.add_camera_frustum(
            "/cam_frustum",
            fov=float(data["fov_y"]),
            aspect=aspect,
            scale=0.30,
            line_width=2.5,
            color=FRUSTUM_COLOR,
            image=thumb0,
            wxyz=data["cam_wxyz"][0],
            position=data["cam_pos"][0],
        )
        if data["panels"]:
            self.rgb_panel.image = data["panels"][0]

        self.apply_vis()
        self.focus(data)
        self._load_rgb(data)
        thumb0 = data["thumbs"][0] if data["thumbs"] else None
        fr = self._handles.get("frustum")
        if fr is not None and thumb0 is not None:
            fr.aspect = float(thumb0.shape[1] / max(thumb0.shape[0], 1))
            fr.fov = float(data["fov_y"])
            fr.image = thumb0
        if data["panels"]:
            self.rgb_panel.image = data["panels"][0]

    def set_frame(self, ft: int) -> None:
        if not self._active_keys:
            return
        key = self._active_keys[self.idx]
        data = self.get_data(key)
        if data["n"] == 0:
            return
        self.t = int(np.clip(ft, 0, data["n"] - 1))
        h = self._handles
        if h.get("gt_mesh") is not None:
            h["gt_mesh"].vertices = np.ascontiguousarray(data["gt_verts"][self.t], dtype=np.float32)
        if h.get("pred_mesh") is not None:
            h["pred_mesh"].vertices = np.ascontiguousarray(data["pred_verts"][self.t], dtype=np.float32)
        if h.get("frame") is not None:
            h["frame"].position = data["cam_pos"][self.t]
            h["frame"].wxyz = data["cam_wxyz"][self.t]
        if h.get("frustum") is not None:
            h["frustum"].position = data["cam_pos"][self.t]
            h["frustum"].wxyz = data["cam_wxyz"][self.t]
            if data["thumbs"]:
                h["frustum"].image = data["thumbs"][self.t]
        if data["panels"]:
            self.rgb_panel.image = data["panels"][self.t]

    def tick(self) -> None:
        now = time.monotonic()
        elapsed = min(now - self.last_tick_time, 0.5)
        self.last_tick_time = now
        if not self.playing or not self._active_keys:
            return
        data = self.get_data(self._active_keys[self.idx])
        if data["n"] > 0:
            # Advance according to the source timeline, independently of the
            # lower network/render update rate.
            self.playhead = (self.playhead + elapsed * self.source_fps) % data["n"]
            frame = int(self.playhead)
            if frame != self.t:
                self.gui_t.value = frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path, required=True, help="fom-predict output")
    parser.add_argument("--gt", type=Path, required=True, help="fom-gt output")
    parser.add_argument("--pcd-dir", type=Path, default=viz_cache_dir("vipe_pcd"))
    parser.add_argument("--video-manifest", type=Path, default=None,
                        help="Optional JSON list of {pred_key, mp4} for fast RGB loading")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--seq", type=int, default=0)
    parser.add_argument("--max-seqs", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--pcd-only",
        action="store_true",
        help="Only list sequences that have a fused scene point cloud in --pcd-dir.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public viser.studio share URL (expires in 24h).",
    )
    args = parser.parse_args()

    print(f"Loading preds: {args.preds}")
    preds = load_sequences(args.preds)
    print(f"Loading GT: {args.gt}")
    gt = load_sequences(args.gt)
    keys = sorted(set(preds) & set(gt))
    pcd_only = bool(args.pcd_only)
    if pcd_only:
        keys = [k for k in keys if (args.pcd_dir / f"{k.replace('/', '_')}.npz").is_file()]
    keys = keys[: args.max_seqs]
    if not keys:
        raise SystemExit("No sequences to show (check --preds/--gt, or drop --pcd-only).")
    print(f"Sequences: {len(keys)} (capped at {args.max_seqs}, pcd_only={pcd_only})")
    n_pcd = sum(1 for k in keys if (args.pcd_dir / f"{k.replace('/', '_')}.npz").is_file())
    print(f"VIPE PCD caches available: {n_pcd}")

    print("Loading SMPL-X...")
    smpl = load_smplx()
    # warm one sequence on device
    device = torch.device(args.device)
    smpl = smpl.to(device)

    bind = args.host or "0.0.0.0"
    server = viser.ViserServer(host=bind, port=args.port, verbose=False)
    print(f"Viser -> http://{host_ip()}:{args.port}  (bound {bind}:{args.port})", flush=True)
    print(f"If Cursor is SSH'd here, also try http://127.0.0.1:{args.port}", flush=True)
    if args.share:
        print("Requesting public share URL via share.viser.studio ...", flush=True)
        try:
            share_url = server.request_share_url(verbose=True)
        except Exception as exc:
            share_url = None
            print(f"Share URL failed: {exc}", flush=True)
        print(f"PUBLIC_SHARE_URL={share_url}", flush=True)
        share_path = viz_cache_dir("logs") / "viser_share.url"
        share_path.parent.mkdir(parents=True, exist_ok=True)
        share_path.write_text("" if share_url is None else f"{share_url}\n")
    viewers: dict[int, ReconViewer] = {}
    viewer_tasks: dict[int, asyncio.Task] = {}
    shared_cache: dict[str, dict] = {}
    prepare_lock = asyncio.Lock()

    async def run_viewer(viewer: ReconViewer) -> None:
        try:
            key = viewer._active_keys[viewer.idx]
            print(f"[client {viewer.client.client_id}] preparing {key}", flush=True)
            # SMPL evaluation and MP4 decoding take several seconds. Keep them off
            # viser's websocket loop so tunneled browser connections don't time out.
            async with prepare_lock:
                data = await asyncio.to_thread(viewer.get_data, key)
                await asyncio.to_thread(viewer._load_rgb, data)
            print(f"[client {viewer.client.client_id}] publishing scene", flush=True)
            viewer.rebuild(viewer.idx)
            while viewer.client.client_id in viewers:
                viewer.tick()
                await asyncio.sleep(1.0 / max(viewer.fps, 1.0))
        except asyncio.CancelledError:
            raise
        except Exception:
            traceback.print_exc()

    @server.on_client_connect
    async def _(client: viser.ClientHandle) -> None:
        print(f"[client {client.client_id}] connected", flush=True)
        viewers.pop(client.client_id, None)
        viewer = ReconViewer(
            client,
            keys,
            preds,
            gt,
            smpl,
            args.pcd_dir,
            start_idx=args.seq,
            shared_cache=shared_cache,
            defer_rebuild=True,
            video_manifest=args.video_manifest,
        )
        viewers[client.client_id] = viewer
        # The connect callback must return immediately: viser doesn't start its
        # websocket producers until all connect callbacks have returned.
        task = asyncio.create_task(run_viewer(viewer))
        viewer_tasks[client.client_id] = task
        task.add_done_callback(lambda _: viewer_tasks.pop(client.client_id, None))
        print(f"[client {client.client_id}] websocket ready", flush=True)

    @server.on_client_disconnect
    def _(client: viser.ClientHandle) -> None:
        viewers.pop(client.client_id, None)
        task = viewer_tasks.pop(client.client_id, None)
        if task is not None:
            task.cancel()

    while True:
        time.sleep(3600.0)


if __name__ == "__main__":
    main()
