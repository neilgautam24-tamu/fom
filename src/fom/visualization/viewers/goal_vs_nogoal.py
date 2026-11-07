#!/usr/bin/env python3
"""Viser: GT (green) + goal-conditioned (blue) + no-goal (orange) predictions in one scene (Fig. S3).

Same layout as :mod:`fom.visualization.viewers.gt_vs_pred`. Inputs come from
:mod:`fom.visualization.select_goal_examples`:

    python -m fom.visualization.viewers.goal_vs_nogoal --data-dir goal_vs_nogoal/ --port 8110
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
GOAL_COLOR = (70, 150, 255)
NOGOAL_COLOR = (255, 140, 60)
FRUSTUM_COLOR = (255, 200, 80)















def thumb(rgb: np.ndarray, size: int = 80) -> np.ndarray:
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    im.thumbnail((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im)


def load_rgb_window(bundle: Path, n_frames: int):
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


class CompareViewer:
    def __init__(
        self,
        client,
        keys,
        preds_goal,
        preds_nogoal,
        gt,
        smpl,
        ranking: list[dict],
        pcd_dir: Path,
        start_idx: int = 0,
        shared_cache: dict | None = None,
        defer_rebuild: bool = False,
    ) -> None:
        self.client = client
        self.keys = keys
        self.preds_goal = preds_goal
        self.preds_nogoal = preds_nogoal
        self.gt = gt
        self.smpl = smpl
        self.ranking = {r["pred_key"]: r for r in ranking}
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
        self._handles: dict = {}
        self._active_keys = list(keys)

        with client.gui.add_folder("Sequence"):
            self.gui_seq = client.gui.add_slider(
                "Index", min=0, max=len(keys) - 1, step=1, initial_value=self.idx
            )
            self.gui_name = client.gui.add_text("Name", keys[self.idx][:80])
            self.gui_info = client.gui.add_markdown("")
        with client.gui.add_folder("Playback"):
            self.gui_t = client.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            self.gui_play = client.gui.add_checkbox("Playing", initial_value=False)
            self.gui_fps = client.gui.add_slider(
                "Render FPS", min=1, max=15, step=1, initial_value=10
            )
        with client.gui.add_folder("Display"):
            self.gui_gt = client.gui.add_checkbox("Show GT", initial_value=True)
            self.gui_goal = client.gui.add_checkbox("Show WITH goal", initial_value=True)
            self.gui_nogoal = client.gui.add_checkbox("Show NO goal", initial_value=True)
            self.gui_frustum = client.gui.add_checkbox("Show frustum", initial_value=True)
            self.gui_pcd = client.gui.add_checkbox("Show GT PCD", initial_value=True)
            self.gui_pcd_size = client.gui.add_slider(
                "Point size", min=0.002, max=0.03, step=0.001, initial_value=0.008
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

        for g in (
            self.gui_gt,
            self.gui_goal,
            self.gui_nogoal,
            self.gui_frustum,
            self.gui_pcd,
            self.gui_pcd_size,
            self.gui_rgb_panel,
        ):
            g.on_update(lambda _: self.apply_vis())

        if not defer_rebuild:
            self.rebuild(self.idx)

    def _pcd_path(self, key: str) -> Path:
        return self.pcd_dir / f"{key.replace('/', '_')}.npz"

    def get_data(self, key: str) -> dict:
        if key in self.cache:
            return self.cache[key]

        goal = self.preds_goal[key]
        nogoal = self.preds_nogoal[key]
        gt = self.gt[key]
        goal_smpl = {k: torch.as_tensor(to_np(v)) for k, v in goal["smpl_params"].items()}
        nogoal_smpl = {k: torch.as_tensor(to_np(v)) for k, v in nogoal["smpl_params"].items()}
        gt_smpl = {k: torch.as_tensor(to_np(v)) for k, v in gt["smpl_params"].items()}
        with torch.no_grad():
            _, goal_verts, _ = evaluate_smplx(self.smpl, goal_smpl)
            _, nogoal_verts, _ = evaluate_smplx(self.smpl, nogoal_smpl)
            _, gt_verts, _ = evaluate_smplx(self.smpl, gt_smpl)
        goal_verts = goal_verts.cpu().numpy()
        nogoal_verts = nogoal_verts.cpu().numpy()
        gt_verts = gt_verts.cpu().numpy()
        n = min(len(goal_verts), len(nogoal_verts), len(gt_verts))
        goal_verts, nogoal_verts, gt_verts = goal_verts[:n], nogoal_verts[:n], gt_verts[:n]

        aria = to_np(goal["camera_T"] if goal.get("camera_T") is not None else gt["camera_T"])[:n]
        cam_wxyz = np.stack([rotmat_to_wxyz(aria[i, :3, :3]) for i in range(n)])
        cam_pos = aria[:, :3, 3].astype(np.float32)

        pcd_xyz = np.zeros((0, 3), np.float32)
        pcd_rgb = np.zeros((0, 3), np.uint8)
        pcd_path = self._pcd_path(key)
        if pcd_path.is_file():
            z = np.load(pcd_path)
            pcd_xyz = np.asarray(z["xyz"], np.float32)
            pcd_rgb = np.asarray(z["rgb"], np.uint8)
            max_pts = 120_000
            if pcd_xyz.shape[0] > max_pts:
                rng = np.random.default_rng(0)
                idx = rng.choice(pcd_xyz.shape[0], max_pts, replace=False)
                pcd_xyz, pcd_rgb = pcd_xyz[idx], pcd_rgb[idx]

        data = {
            "key": key,
            "n": n,
            "goal_verts": goal_verts,
            "nogoal_verts": nogoal_verts,
            "gt_verts": gt_verts,
            "cam_pos": cam_pos,
            "cam_wxyz": cam_wxyz,
            "panels": [],
            "thumbs": [],
            "fov_y": 1.0,
            "pcd_xyz": pcd_xyz,
            "pcd_rgb": pcd_rgb,
            "has_pcd": pcd_xyz.shape[0] > 0,
            "_need_rgb": True,
            "take": key.split("___")[0],
            "metrics": self.ranking.get(key, {}),
        }
        self.cache[key] = data
        return data

    def _load_rgb(self, data: dict) -> None:
        if not data.get("_need_rgb"):
            return
        data["_need_rgb"] = False
        try:
            bundle = find_bundle(data["take"])
            if bundle is not None:
                panels, thumbs, fov_y = load_rgb_window(bundle, data["n"])
                data["panels"] = panels
                data["thumbs"] = thumbs
                data["fov_y"] = fov_y
        except Exception as exc:
            print(f"[rgb] {data['key']}: {exc}", flush=True)

    def apply_vis(self) -> None:
        h = self._handles
        if h.get("gt_mesh") is not None:
            h["gt_mesh"].visible = bool(self.gui_gt.value)
        if h.get("goal_mesh") is not None:
            h["goal_mesh"].visible = bool(self.gui_goal.value)
        if h.get("nogoal_mesh") is not None:
            h["nogoal_mesh"].visible = bool(self.gui_nogoal.value)
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
        body_c = data["gt_verts"][0].mean(axis=0)
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
        m = data["metrics"]
        delta = m.get("delta_mm")
        g = m.get("mpjpe_goal_mm")
        ng = m.get("mpjpe_nogoal_mm")
        metric_line = (
            f"MPJPE mm - **goal {g:.1f}** | no-goal {ng:.1f} | delta={delta:.1f}  \n"
            if delta is not None
            else ""
        )
        pcd_note = (
            f"GT PCD **{data['pcd_xyz'].shape[0]:,}** pts"
            if data["has_pcd"]
            else "GT PCD: _not cached_"
        )
        self.gui_info.content = (
            f"**{self.idx}/{len(self._active_keys)-1}**  \n"
            f"frames: {data['n']}  \n"
            f"{metric_line}"
            f"GT (green) | WITH goal (blue) | NO goal (orange) | frustum (yellow)  \n"
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

        self._handles["gt_mesh"] = self.client.scene.add_mesh_simple(
            "/gt_body",
            vertices=data["gt_verts"][0],
            faces=self.faces,
            color=GT_COLOR,
            opacity=0.50,
        )
        self._handles["goal_mesh"] = self.client.scene.add_mesh_simple(
            "/goal_body",
            vertices=data["goal_verts"][0],
            faces=self.faces,
            color=GOAL_COLOR,
            opacity=0.85,
        )
        self._handles["nogoal_mesh"] = self.client.scene.add_mesh_simple(
            "/nogoal_body",
            vertices=data["nogoal_verts"][0],
            faces=self.faces,
            color=NOGOAL_COLOR,
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
        if h.get("goal_mesh") is not None:
            h["goal_mesh"].vertices = np.ascontiguousarray(
                data["goal_verts"][self.t], dtype=np.float32
            )
        if h.get("nogoal_mesh") is not None:
            h["nogoal_mesh"].vertices = np.ascontiguousarray(
                data["nogoal_verts"][self.t], dtype=np.float32
            )
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
            self.playhead = (self.playhead + elapsed * self.source_fps) % data["n"]
            frame = int(self.playhead)
            if frame != self.t:
                self.gui_t.value = frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True, help="Output dir of select_goal_examples")
    ap.add_argument("--pcd-dir", type=Path, default=viz_cache_dir("vipe_pcd"))
    ap.add_argument("--port", type=int, default=8110)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()


    ranking_path = args.data_dir / "goal_advantage_ranking.json"
    ranking = json.loads(ranking_path.read_text()) if ranking_path.is_file() else []
    print(f"Loading selection from {args.data_dir}")
    preds_g = load_sequences(args.data_dir / "preds_goal_viz.pkl")
    preds_ng = load_sequences(args.data_dir / "preds_nogoal_viz.pkl")
    gt = load_sequences(args.data_dir / "gt_viz.pkl")
    if ranking:
        keys = [r["pred_key"] for r in ranking if r["pred_key"] in preds_g and r["pred_key"] in preds_ng]
    else:
        keys = sorted(set(preds_g) & set(preds_ng) & set(gt))
    print(f"Sequences: {len(keys)}")

    smpl = load_smplx().to(torch.device(args.device))
    server = viser.ViserServer(host=args.host, port=args.port, verbose=False)
    print(f"Viser -> http://{host_ip()}:{args.port}", flush=True)
    if args.share:
        try:
            share_url = server.request_share_url(verbose=True)
            print(f"PUBLIC_SHARE_URL={share_url}", flush=True)
            (viz_cache_dir("logs") / "viser_share.url").write_text(f"{share_url}\n")
        except Exception as exc:
            print(f"Share failed: {exc}", flush=True)

    viewers: dict = {}
    viewer_tasks: dict = {}
    shared_cache: dict = {}
    prepare_lock = asyncio.Lock()

    async def run_viewer(viewer: CompareViewer) -> None:
        try:
            key = viewer._active_keys[viewer.idx]
            async with prepare_lock:
                data = await asyncio.to_thread(viewer.get_data, key)
                await asyncio.to_thread(viewer._load_rgb, data)
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
        viewer = CompareViewer(
            client,
            keys,
            preds_g,
            preds_ng,
            gt,
            smpl,
            ranking,
            args.pcd_dir,
            start_idx=args.seq,
            shared_cache=shared_cache,
            defer_rebuild=True,
        )
        viewers[client.client_id] = viewer
        task = asyncio.create_task(run_viewer(viewer))
        viewer_tasks[client.client_id] = task
        task.add_done_callback(lambda _: viewer_tasks.pop(client.client_id, None))

    @server.on_client_disconnect
    def _(client: viser.ClientHandle) -> None:
        viewers.pop(client.client_id, None)
        task = viewer_tasks.pop(client.client_id, None)
        if task is not None:
            task.cancel()

    print("Ready - open the URL above.", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
