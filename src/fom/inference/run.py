"""Step 2 of video inference: conditions + autoregressive prediction + stitching (GPU).

Works for generated clips (shorter than one window) and long real videos alike:

  * the ViPE trajectory (optionally floor-levelled by ``fom.inference.prepare --level``) is
    converted to the training convention and resampled to the motion rate;
  * the clip is split into ``WINDOW``-frame windows overlapping by ``--overlap`` frames, each
    canonicalized at its own first frame (as in training) with one shared floor height;
  * window predictions are mapped back to the clip frame and overlaps are cross-faded.

Without levelling, the vertical axis may be flipped for generated clips whose name implies
ascending / descending / sitting (see :func:`fom.inference.camera.infer_gravity_flip`).

Outputs (in ``<out_dir>``):
  motion_smplx_canonical.npz   SMPL-X in the clip frame (Z-up metres, floor z=0, frame-0 camera at the
                               XY origin), with AMASS-style ``poses`` / ``trans``.
  motion_smplx_vipe_world.npz  The same sequence in ViPE world (OpenCV, Y down) for overlay on ViPE maps.
  windows/                     Raw per-window predictions.

Usage:
    python -m fom.inference.run <prep_dir> <out_dir> --config configs/train/stage3_headcam.yaml \\
        --ckpt <ckpt> DATA.ROOT <packed training data with the normalization statistics>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fom.body.canonicalization import canonicalize_camera
from fom.inference.camera import (
    HEAD_HEIGHT_M,
    RZ180,
    floor_shift,
    infer_gravity_flip,
    opencv_to_zup,
    resample_poses,
    trajectory_features,
    zup_world_matrix,
)
from fom.inference.conditions import Dinov2Extractor, center_goal_heatmap, depth_to_patches, make_window_conditions
from fom.inference.predictor import MotionPredictor
from fom.inference.stitching import stitch_windows, transform_smplx
from fom.utils import rotation_conversions as rc


def _camera_path(prep_dir: Path, meta: dict, motion_fps: int, flip: str) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Clip camera poses (Z-up, Aria axes) at the motion rate, and the world transform applied."""
    c2w = torch.from_numpy(np.load(prep_dir / "pose.npz")["data"].astype(np.float32))
    level = torch.tensor(meta["R_level_vipe"], dtype=torch.float32) if meta.get("R_level_vipe") else None

    flip_gravity = False
    if level is None and flip != "no":
        zup = opencv_to_zup(c2w)
        flip_gravity = flip == "yes" or infer_gravity_flip(meta["clip_id"], float(zup[-1, 2, 3] - zup[0, 2, 3]))

    zup = opencv_to_zup(c2w, flip_gravity=flip_gravity, level=level)
    world = zup_world_matrix(flip_gravity, level)
    return resample_poses(zup, meta["src_fps"], motion_fps), world, flip_gravity


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prep_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--config", required=True, help="Config the checkpoint was trained with")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--overlap", type=int, default=30, help="Frames shared by consecutive windows")
    ap.add_argument("--flip-gravity", choices=["auto", "yes", "no"], default="auto",
                    help="Only for un-levelled clips; 'auto' uses the clip-name heuristic")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("overrides", nargs="*", help="KEY VALUE config overrides (e.g. DATA.ROOT ...)")
    return ap.parse_intermixed_args(argv)


@torch.inference_mode()
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out = args.out_dir
    (out / "windows").mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.prep_dir / "meta.json").read_text())
    predictor = MotionPredictor(args.config, args.ckpt, args.overrides, args.device)
    cfg = predictor.cfg
    window, motion_fps = cfg.DATA.WINDOW, cfg.DATA.MOTION_FPS
    visual_step = motion_fps // cfg.DATA.VISUAL_FPS
    body = predictor.body_model

    # Camera trajectory
    c2w, world_vipe_to_zup, flipped = _camera_path(args.prep_dir, meta, motion_fps, args.flip_gravity)
    num_frames = c2w.shape[0]
    src_idx = np.linspace(0, meta["n_frames"] - 1, num_frames).round().astype(int)
    floor_height = float(c2w[0, 2, 3]) - HEAD_HEIGHT_M  # one floor for the whole clip
    _, clip_transform = canonicalize_camera(c2w)

    starts = [0]
    while starts[-1] + window < num_frames:
        starts.append(starts[-1] + window - args.overlap)
    assert all(s % visual_step == 0 for s in starts), "window starts must align with the visual staircase"
    print(f"{meta['clip_id']}: {num_frames} frames @ {motion_fps} fps, windows "
          f"{[(s, min(s + window, num_frames)) for s in starts]}, gravity flipped: {flipped}", flush=True)

    # Visual conditions on the whole timeline (held on the visual staircase, as in training)
    rgb = np.load(args.prep_dir / "rgb_512.npy", mmap_mode="r")
    depth = np.load(args.prep_dir / "depth_512.npy", mmap_mode="r")
    key_frames = list(range(0, num_frames, visual_step))
    extractor = Dinov2Extractor(predictor.device)
    rgb_cls = extractor.cls_features([np.ascontiguousarray(rgb[src_idx[j]]) for j in key_frames])
    depth_key = torch.from_numpy(np.stack([depth_to_patches(np.asarray(depth[src_idx[j]])) for j in key_frames]))
    held = torch.arange(num_frames) // visual_step
    rgb_cls, depth_patches = rgb_cls[held], depth_key[held].float()
    goal = center_goal_heatmap()[None].repeat(num_frames, 1)
    del extractor
    torch.cuda.empty_cache()

    # Windowed prediction
    windows = []
    for k, start in enumerate(starts):
        end = min(start + window, num_frames)
        canonical, window_transform = canonicalize_camera(c2w[start:end])
        traj = trajectory_features(canonical, floor_height)
        y = make_window_conditions(
            predictor.normalize_traj(traj), rgb_cls[start:end], depth_patches[start:end], goal[start:end], window
        )
        motion = predictor.predict(y)
        decoded = predictor.decode(motion, traj)
        torch.save({"y": y, "traj": traj, "motion_norm": motion, "decoded": decoded, "start": start, "end": end},
                   out / "windows" / f"window_{k}.pt")

        # window frame (canonical at window start, floored) -> clip frame (canonical at clip start, floored)
        to_clip = floor_shift(floor_height) @ clip_transform @ torch.linalg.inv(window_transform) @ torch.linalg.inv(
            floor_shift(floor_height))
        params = {key: v.float() for key, v in decoded["smpl_params"].items()}
        for key in ("jaw_pose", "leye_pose", "reye_pose"):  # the model predicts body + hands only
            params.setdefault(key, torch.eye(3).repeat(end - start, 1, 1))
        pelvis = body.forward(betas=params["betas"][:1]).joints[0, 0]
        windows.append({
            "start": start,
            "end": end,
            "smpl": transform_smplx(params, to_clip, pelvis),
            "camera": to_clip[None] @ decoded["camera_T"].float(),
        })
        print(f"  window {k}: frames {start}-{end}", flush=True)

    # One body shape per clip: length-weighted mean of the window betas; keep each pelvis in place.
    lengths = torch.tensor([w["end"] - w["start"] for w in windows], dtype=torch.float32)
    betas = (torch.stack([w["smpl"]["betas"][0] for w in windows]) * lengths[:, None]).sum(0) / lengths.sum()
    pelvis_clip = body.forward(betas=betas[None]).joints[0, 0]
    for w in windows:
        pelvis_w = body.forward(betas=w["smpl"]["betas"][:1]).joints[0, 0]
        w["smpl"]["transl"] = w["smpl"]["transl"] + pelvis_w - pelvis_clip
        w["smpl"]["betas"] = betas[None].repeat(w["end"] - w["start"], 1)

    stitched = stitch_windows(windows, num_frames)
    smpl_seq, camera = stitched["smpl"], stitched["camera"]
    joints, verts = [], []
    for a in range(0, num_frames, 64):
        o = body.forward(**{key: v[a : a + 64] for key, v in smpl_seq.items()})
        joints.append(o.joints[:, :55])
        verts.append(o.vertices)
    joints, verts = torch.cat(joints), torch.cat(verts)

    def axis_angle(rot: torch.Tensor) -> torch.Tensor:
        return rc.matrix_to_axis_angle(rot).reshape(num_frames, -1)

    poses = torch.cat([axis_angle(smpl_seq["global_orient"][:, None]), axis_angle(smpl_seq["body_pose"]),
                       axis_angle(smpl_seq["jaw_pose"][:, None]), axis_angle(smpl_seq["leye_pose"][:, None]),
                       axis_angle(smpl_seq["reye_pose"][:, None]), axis_angle(smpl_seq["left_hand_pose"]),
                       axis_angle(smpl_seq["right_hand_pose"])], 1)  # (N, 165)

    common = {
        "fps": motion_fps,
        "time_s": np.arange(num_frames) / motion_fps,
        "src_frame_index": src_idx,
        "faces": body.faces.astype(np.int32),
        "gender": "neutral",
    }
    np.savez(
        out / "motion_smplx_canonical.npz",
        **common,
        frame="clip canonical: Z-up metres, floor z=0, frame-0 camera at the XY origin",
        **{key: v.numpy() for key, v in smpl_seq.items()},
        poses=poses.numpy(),
        trans=smpl_seq["transl"].numpy(),
        mocap_framerate=motion_fps,
        joints=joints.numpy(),
        vertices=verts.numpy().astype(np.float32),
        camera_c2w=camera.numpy(),
    )

    # clip frame -> ViPE world
    to_vipe = torch.linalg.inv(world_vipe_to_zup) @ torch.linalg.inv(clip_transform) @ torch.linalg.inv(
        floor_shift(floor_height))
    smpl_vipe = transform_smplx(smpl_seq, to_vipe, body.forward(betas=smpl_seq["betas"][:1]).joints[0, 0])
    camera_axes = torch.eye(4) if flipped else RZ180
    np.savez(
        out / "motion_smplx_vipe_world.npz",
        **common,
        frame="ViPE world (OpenCV: X right, Y down, Z forward of the first camera), metres",
        **{key: v.numpy() for key, v in smpl_vipe.items()},
        joints=(joints @ to_vipe[:3, :3].T + to_vipe[:3, 3]).numpy(),
        vertices=(verts @ to_vipe[:3, :3].T + to_vipe[:3, 3]).numpy().astype(np.float32),
        camera_c2w=(to_vipe[None] @ camera @ camera_axes[None]).numpy(),
    )
    (out / "meta.json").write_text(json.dumps({
        "clip_id": meta["clip_id"],
        "ckpt": str(args.ckpt),
        "n_frames": num_frames,
        "fps": motion_fps,
        "windows": [[w["start"], w["end"]] for w in windows],
        "overlap": args.overlap,
        "floor_height": floor_height,
        "head_height_assumed_m": HEAD_HEIGHT_M,
        "gravity_flipped": flipped,
        "levelling": meta.get("levelling"),
    }, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
