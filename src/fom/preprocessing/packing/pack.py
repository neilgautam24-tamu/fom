#!/usr/bin/env python3
"""Pack Nymeria visible-goal sequence bundles into the ``ee_*.pt`` training format.

Nymeria bundles (``visible_goal_sequence_bundles/vgoal_seq_*``) already contain
SMPL-X motion and the Aria head trajectory. Compared with the ADT packer:

  - Source motion is ~60 FPS (240 Hz body stream, stride 4), not ADT's 30 FPS.
  - World coordinates are already Z-up (no ADT Y-up -> Z-up Rx(+90 deg) transform).
  - Camera poses come from ``trajectory.npz`` (device pose + ``t_device_camera``)
    instead of ADT ``camera_poses.csv`` + ``device_calibration.json``.

Output (in ``<output-dir>``):

  - ``ee_train.pt`` / ``ee_val.pt`` - dicts keyed by
    ``{episode}_{vgoal_seq_name}___0___{T_source-1}``
  - ``split_log.json`` - sequence-level train/val assignment

After packing, compute normalization statistics with
``python -m fom.preprocessing.packing.stats --train <output-dir>/ee_train.pt``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from fom.body.smplx_model import evaluate_smplx, load_smplx
from fom.utils import rotation_conversions as rc
from fom.utils.pca import matrix_to_pca

DEFAULT_TARGET_FPS = 15

# Rotate 90 deg CCW locally to align the Aria sensor frame with the OpenCV camera convention.
R_CAMERA_FIX = torch.tensor(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)


def infer_source_fps(timestamps_us: np.ndarray) -> float:
    ts = np.asarray(timestamps_us, dtype=np.int64)
    if ts.shape[0] < 2:
        return 60.0
    dt_s = np.median(np.diff(ts)) * 1e-6
    return float(1.0 / dt_s) if dt_s > 0 else 60.0


def load_smplx_params(smplx_path: Path, device: str) -> dict[str, torch.Tensor]:
    data = np.load(smplx_path)
    t = data["body_pose"].shape[0]

    body_pose = torch.tensor(data["body_pose"], dtype=torch.float32, device=device)
    body_pose = rc.axis_angle_to_matrix(body_pose.reshape(-1, 3)).reshape(t, 21, 3, 3)

    left_hand = torch.tensor(data["left_hand_pose"], dtype=torch.float32, device=device)
    left_hand = rc.axis_angle_to_matrix(left_hand.reshape(-1, 3)).reshape(t, 15, 3, 3)

    right_hand = torch.tensor(data["right_hand_pose"], dtype=torch.float32, device=device)
    right_hand = rc.axis_angle_to_matrix(right_hand.reshape(-1, 3)).reshape(t, 15, 3, 3)

    global_orient = torch.tensor(data["global_orient"], dtype=torch.float32, device=device)
    global_orient = rc.axis_angle_to_matrix(global_orient)

    transl = torch.tensor(data["transl"], dtype=torch.float32, device=device)

    betas = data["betas"].astype(np.float32)
    if betas.ndim == 2:
        betas = betas[0]
    betas = torch.tensor(betas, dtype=torch.float32, device=device)

    return {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "left_hand_pose": left_hand,
        "right_hand_pose": right_hand,
        "transl": transl,
        "betas": betas,
    }


def interpolate_smplx_params(
    smpl_params: dict[str, torch.Tensor],
    target_frames: int,
    *,
    counter: list[int],
) -> dict[str, torch.Tensor]:
    source_frames = smpl_params["global_orient"].shape[0]
    if source_frames == target_frames:
        return smpl_params

    device = smpl_params["global_orient"].device
    src_idx = np.arange(source_frames)
    tgt_idx = np.linspace(0, source_frames - 1, target_frames)
    out: dict[str, torch.Tensor] = {}

    for key, param in smpl_params.items():
        if key == "betas":
            out[key] = param
            continue
        p = param.detach().cpu().numpy()
        if p.ndim == 2:
            new_p = np.zeros((target_frames, p.shape[1]))
            for i in range(p.shape[1]):
                new_p[:, i] = interp1d(
                    src_idx, p[:, i], kind="linear", fill_value="extrapolate"
                )(tgt_idx)
        elif p.ndim == 3:
            new_p = np.zeros((target_frames, p.shape[1], p.shape[2]))
            for j in range(p.shape[1]):
                for i in range(p.shape[2]):
                    new_p[:, j, i] = interp1d(
                        src_idx, p[:, j, i], kind="linear", fill_value="extrapolate"
                    )(tgt_idx)
        elif p.ndim == 4:
            interp_joints = []
            for j in range(p.shape[1]):
                quats = np.array(
                    [Rotation.from_matrix(p[t, j]).as_quat() for t in range(source_frames)]
                )
                iq = np.zeros((target_frames, 4))
                for i in range(4):
                    iq[:, i] = interp1d(
                        src_idx, quats[:, i], kind="linear", fill_value="extrapolate"
                    )(tgt_idx)
                mats = np.array(
                    [Rotation.from_quat(q / np.linalg.norm(q)).as_matrix() for q in iq]
                )
                interp_joints.append(mats)
            new_p = np.stack(interp_joints, axis=1)
        else:
            raise ValueError(f"Unexpected shape for '{key}': {p.shape}")
        out[key] = torch.tensor(new_p, dtype=torch.float32, device=device)

    counter[0] += 1
    return out


def load_camera_poses_from_trajectory(traj_path: Path, device: str) -> torch.Tensor:
    traj = np.load(traj_path)
    t = len(traj["device_position"])

    translation = torch.tensor(traj["device_position"], dtype=torch.float32, device=device)
    quat_wxyz = traj["device_quat_wxyz"]
    rot_mats = torch.tensor(
        np.array(
            [
                Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                for q in quat_wxyz
            ]
        ),
        dtype=torch.float32,
        device=device,
    )

    t_world_device = torch.eye(4, device=device).unsqueeze(0).repeat(t, 1, 1)
    t_world_device[:, :3, :3] = rot_mats
    t_world_device[:, :3, 3] = translation

    t_device_sensor = torch.tensor(traj["t_device_camera"], dtype=torch.float32, device=device)
    r_cam_fix = R_CAMERA_FIX.to(device)

    aria_traj_t = torch.einsum("tij,jk->tik", t_world_device, t_device_sensor)
    aria_traj_t = torch.einsum("tij,jk->tik", aria_traj_t, r_cam_fix)
    return aria_traj_t


def compute_aria_trajectory_6d(aria_traj_t: torch.Tensor) -> torch.Tensor:
    rotation = aria_traj_t[:, :3, :3]
    translation = aria_traj_t[:, :3, 3]
    return torch.cat([rc.matrix_to_rotation_6d(rotation), translation], dim=1)


def compute_floor_height(kp3d: torch.Tensor) -> float:
    return kp3d[:, [10, 11], 2].mean().item() - 0.02


def downsample_smpl_params(
    smpl_params: dict[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {k: (v if k == "betas" else v[indices]) for k, v in smpl_params.items()}


def recenter_sequence_inplace(sequence_data: dict) -> dict:
    """Remove the arbitrary world-frame origin from a converted sequence.

    Nymeria SLAM world frames can place the activity tens of meters from the
    origin (~44-50 m for some single-object episodes). The v4_beta
    representation stores the first-frame absolute position, so this offset
    leaks into training targets and normalization statistics. We apply a rigid
    XY translation so the frame-0 pelvis sits at the world XY origin.

    Z (hence ``floor_height``) and all relative geometry are preserved.
    ``body_root_offset`` is a difference of two equally-shifted points, so it is
    unchanged. Image/heatmap features (DINOv2, goal heatmaps) are not in the
    world frame and need no adjustment.
    """
    if sequence_data.get("recentered"):
        return sequence_data

    kp3d = sequence_data["kp3d"]
    origin_xy = kp3d[0, 0, :2].clone()  # frame-0 pelvis world XY

    sequence_data["smpl_params"]["transl"][:, :2] -= origin_xy
    kp3d[..., :2] -= origin_xy
    sequence_data["aria_traj"][:, 6:8] -= origin_xy

    meta = sequence_data.get("metadata")
    if isinstance(meta, dict):
        ox, oy = float(origin_xy[0]), float(origin_xy[1])
        gw = meta.get("goal_world")
        if isinstance(gw, (list, tuple)) and len(gw) >= 2:
            gw = list(gw)
            gw[0] -= ox
            gw[1] -= oy
            meta["goal_world"] = gw
        if "goal_world_x" in meta:
            meta["goal_world_x"] -= ox
        if "goal_world_y" in meta:
            meta["goal_world_y"] -= oy

    sequence_data["recentered"] = True
    sequence_data["recenter_origin_xy"] = origin_xy.cpu()
    return sequence_data


def find_sequences(
    bundle_root: Path,
    episode_name: str,
    *,
    prefixes: tuple[str, ...] = ("vgoal_seq_",),
    min_duration_s: float = 0.0,
) -> list[dict]:
    sequences = []
    for seq_dir in sorted(bundle_root.iterdir()):
        if not seq_dir.is_dir() or not any(seq_dir.name.startswith(p) for p in prefixes):
            continue
        smplx_path = seq_dir / "smplx_motion.npz"
        traj_path = seq_dir / "trajectory.npz"
        metadata_path = seq_dir / "metadata.json"
        if not smplx_path.is_file() or not traj_path.is_file():
            continue
        if min_duration_s > 0.0 and metadata_path.is_file():
            with open(metadata_path) as handle:
                meta = json.load(handle)
            # Window bundles store duration_sec / timing.duration_s, contact sequences total_duration_s.
            duration = float(
                meta.get("total_duration_s")
                or meta.get("duration_sec")
                or meta.get("duration_s")
                or (meta.get("timing") or {}).get("duration_s")
                or 0.0
            )
            if duration < min_duration_s:
                continue
        sequences.append(
            {
                "episode_name": episode_name,
                "seq_name": seq_dir.name,
                "smplx_path": smplx_path,
                "traj_path": traj_path,
                "metadata_path": metadata_path,
                "sequence_dir": seq_dir,
            }
        )
    return sequences


class NymeriaSequencePacker:
    def __init__(
        self,
        bundle_root: Path,
        episode_name: str,
        output_dir: Path,
        *,
        device: str = "cuda",
        min_frames: int = 15,
        train_ratio: float = 0.86,
        recenter: bool = True,
        target_fps: int = DEFAULT_TARGET_FPS,
    ) -> None:
        self.bundle_root = bundle_root
        self.episode_name = episode_name
        self.output_dir = output_dir
        self.packed_dir = output_dir
        self.device = device
        self.min_frames = min_frames
        self.train_ratio = train_ratio
        self.recenter = recenter
        self.target_fps = int(target_fps)
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")

        self.packed_dir.mkdir(parents=True, exist_ok=True)
        self.smpl = load_smplx().to(device)
        self.r_cam_fix = R_CAMERA_FIX.to(device)

        self.train_sequences: dict[str, dict] = {}
        self.val_sequences: dict[str, dict] = {}
        self.failed_sequences: list[str] = []
        self.frame_mismatch_count = 0
        self.interpolated_count = [0]

    def process_sequence(self, seq_info: dict) -> tuple[str, dict] | None:
        take_name = f"{seq_info['episode_name']}_{seq_info['seq_name']}"
        try:
            smpl_params = load_smplx_params(seq_info["smplx_path"], self.device)
            t_smpl = smpl_params["global_orient"].shape[0]

            aria_traj_t = load_camera_poses_from_trajectory(seq_info["traj_path"], self.device)
            t_camera = aria_traj_t.shape[0]

            motion_raw = np.load(seq_info["smplx_path"])
            source_fps = infer_source_fps(motion_raw["timestamps"])
            smpl_step = max(1, int(round(source_fps / self.target_fps)))

            if t_smpl != t_camera:
                self.frame_mismatch_count += 1
                t_target = min(t_smpl, t_camera)
                if t_smpl != t_target:
                    smpl_params = interpolate_smplx_params(
                        smpl_params, t_target, counter=self.interpolated_count
                    )
                if t_camera != t_target:
                    aria_traj_t = aria_traj_t[:t_target]
                t_source = t_target
            else:
                t_source = t_smpl

            indices_target = torch.arange(0, t_source, smpl_step, device=self.device)
            t_target_fps = len(indices_target)
            smpl_params = downsample_smpl_params(smpl_params, indices_target)
            aria_traj_t = aria_traj_t[indices_target]
            t = t_target_fps

            if t < self.min_frames:
                return None

            betas_expanded = smpl_params["betas"].unsqueeze(0).expand(t, -1)
            smpl_params_eval = {**smpl_params, "betas": betas_expanded}
            kp3d, _, _ = evaluate_smplx(self.smpl, smpl_params_eval)
            kp3d = kp3d[:, :76]

            floor_height = compute_floor_height(kp3d)

            head_kp_0 = kp3d[0, 15]
            aria_pos_0 = aria_traj_t[0, :3, 3]
            xy_offset = head_kp_0[:2] - aria_pos_0[:2]
            aria_traj_t[:, 0, 3] += xy_offset[0]
            aria_traj_t[:, 1, 3] += xy_offset[1]

            leye_kp_0 = kp3d[0, 23]
            z_offset = float(leye_kp_0[2] - aria_traj_t[0, 2, 3])
            aria_traj_t[:, 2, 3] += z_offset

            aria_traj = compute_aria_trajectory_6d(aria_traj_t)
            body_root_offset = kp3d[0, 0] - smpl_params["transl"][0]

            smpl_params_saved = {
                "global_orient": rc.matrix_to_rotation_6d(smpl_params["global_orient"]).cpu(),
                "body_pose": rc.matrix_to_rotation_6d(
                    smpl_params["body_pose"].reshape(t * 21, 3, 3)
                )
                .reshape(t, 21, 6)
                .cpu(),
                "left_hand_pose": matrix_to_pca(
                    smpl_params["left_hand_pose"], self.smpl.left_hand_components
                ).cpu(),
                "right_hand_pose": matrix_to_pca(
                    smpl_params["right_hand_pose"], self.smpl.right_hand_components
                ).cpu(),
                "transl": smpl_params["transl"].cpu(),
                "betas": smpl_params["betas"].unsqueeze(0).expand(t, -1).cpu(),
            }

            sequence_name = f"{take_name}___0___{t_source - 1}"
            sequence_data = {
                "smpl_params": smpl_params_saved,
                "aria_traj": aria_traj.cpu(),
                "kp3d": kp3d.cpu(),
                "num_frames": t,
                "start_idx": 0,
                "end_idx": t_source - 1,
                "floor_height": floor_height,
                "body_root_offset": body_root_offset.cpu(),
                "source_fps": source_fps,
                "stored_fps": self.target_fps,
                "coord_system": "Z-up",
                "aria_pose_type": "T_world_sensor_xyz_aligned_nymeria_v1",
            }

            if seq_info["metadata_path"].is_file():
                with open(seq_info["metadata_path"]) as handle:
                    sequence_data["metadata"] = json.load(handle)

            if self.recenter:
                recenter_sequence_inplace(sequence_data)

            return sequence_name, sequence_data

        except Exception as exc:
            print(f"[error] {take_name}: {exc}", flush=True)
            self.failed_sequences.append(take_name)
            return None

    @staticmethod
    def build_split_map(seq_names: list[str], train_ratio: float) -> dict[str, str]:
        n = len(seq_names)
        if n <= 1:
            return {name: "train" for name in seq_names}
        n_val = max(1, min(n - 1, round(n * (1.0 - train_ratio))))
        val_names = set(seq_names[-n_val:])
        return {name: ("val" if name in val_names else "train") for name in seq_names}

    def convert_all(self) -> None:
        sequences = find_sequences(self.bundle_root, self.episode_name)
        if not sequences:
            raise FileNotFoundError(f"No vgoal_seq_* bundles under {self.bundle_root}")

        print(f"[convert] found {len(sequences)} sequences in {self.bundle_root}")

        processed: list[tuple[str, dict]] = []
        for seq_info in tqdm(sequences, desc="Converting"):
            result = self.process_sequence(seq_info)
            if result is not None:
                processed.append(result)

        if not processed:
            raise RuntimeError("No sequences converted successfully.")

        split_map = self.build_split_map([name for name, _ in processed], self.train_ratio)
        split_log = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script": "fom.preprocessing.packing.pack",
            "episode_name": self.episode_name,
            "bundle_root": str(self.bundle_root),
            "train_ratio_target": self.train_ratio,
            "total_sequences": len(processed),
            "train_sequences": sum(v == "train" for v in split_map.values()),
            "val_sequences": sum(v == "val" for v in split_map.values()),
            "failed_sequences": self.failed_sequences,
            "frame_mismatch_count": self.frame_mismatch_count,
            "interpolated_count": self.interpolated_count[0],
            "assignments": split_map,
        }
        with open(self.output_dir / "split_log.json", "w") as handle:
            json.dump(split_log, handle, indent=2)

        for seq_name, seq_data in processed:
            split = split_map[seq_name]
            if split == "train":
                self.train_sequences[seq_name] = seq_data
            else:
                self.val_sequences[seq_name] = seq_data

        train_path = self.packed_dir / "ee_train.pt"
        val_path = self.packed_dir / "ee_val.pt"
        torch.save(self.train_sequences, train_path)
        torch.save(self.val_sequences, val_path)

        print(f"[convert] saved {len(self.train_sequences)} train -> {train_path}")
        print(f"[convert] saved {len(self.val_sequences)} val   -> {val_path}")
        print(f"[convert] failed {len(self.failed_sequences)} sequences")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Episode output directory containing visible_goal_sequence_bundles/",
    )
    parser.add_argument(
        "--bundle-subdir",
        type=str,
        default="visible_goal_sequence_bundles",
        help="Subdirectory with exported vgoal_seq_* bundles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --input-dir",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--min-frames", type=int, default=15)
    parser.add_argument(
        "--target-fps",
        type=int,
        default=DEFAULT_TARGET_FPS,
        help="Stored motion/head trajectory FPS (source bundles are approximately 60 FPS).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.86)
    parser.add_argument(
        "--recenter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recenter each sequence so frame-0 pelvis is at the world XY origin "
        "(removes arbitrary SLAM world-frame offsets). Use --no-recenter to keep world coordinates.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    bundle_root = input_dir / args.bundle_subdir
    episode_name = input_dir.name

    converter = NymeriaSequencePacker(
        bundle_root=bundle_root,
        episode_name=episode_name,
        output_dir=output_dir,
        device=args.device,
        min_frames=args.min_frames,
        train_ratio=args.train_ratio,
        recenter=args.recenter,
        target_fps=args.target_fps,
    )
    converter.convert_all()


if __name__ == "__main__":
    main()
