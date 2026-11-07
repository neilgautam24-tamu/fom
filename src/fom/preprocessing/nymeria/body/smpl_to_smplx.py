#!/usr/bin/env python3
"""Convert Nymeria-Plus SMPL motion (xdata_smpl_neutral.npz) to SMPL-X.

Modes
-----
fast
    Vectorized joint remap + one-time shape fit. ~15 s for a 240 k-frame clip.
balanced
    Remap init + 3 Adam steps per batch. ~6 min (1 GPU) / ~2 min (4 GPUs).
quality
    Remap init + 12 Adam steps per batch. ~25 min (1 GPU) / ~7 min (4 GPUs).
quality-h100
    Like quality but batch_size=256 for large GPUs (best with --gpus).
ultra-h100
    Weighted all-vertex loss, two-stage optim, multi-frame shape fit.
    Best visual quality; ~10-12 min on 4x H100 for 240 k frames.

Example (multi-GPU)
-------------------
    python -m fom.preprocessing.nymeria.body.smpl_to_smplx \\
        --input <episode>/body/xdata_smpl_neutral.npz --mode quality --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from fom.preprocessing.nymeria.body.smpl_correspondence import (
    build_vertex_weights,
    load_or_build_correspondence,
    transfer_smpl_to_smplx_vertices,
    weighted_vertex_loss,
)
from fom.preprocessing.nymeria.body.smpl_models import (
    ensure_clean_smpl_model,
    load_smpl_motion,
    smpl_body_pose_to_smplx,
)
from fom.preprocessing.nymeria.io.paths import (
    default_smplx_npz,
    smplx_model_file,
)

MODE_PRESETS: dict[str, dict] = {
    "fast": {"fit_iters": 0, "shape_iters": 30, "batch_size": 512,
             "weighted": False, "two_stage": False, "shape_frames": 1},
    "balanced": {"fit_iters": 3, "shape_iters": 40, "batch_size": 128,
                 "weighted": False, "two_stage": False, "shape_frames": 1},
    "quality": {"fit_iters": 12, "shape_iters": 80, "batch_size": 64,
                "weighted": False, "two_stage": False, "shape_frames": 1},
    "quality-h100": {"fit_iters": 12, "shape_iters": 100, "batch_size": 256,
                     "weighted": False, "two_stage": False, "shape_frames": 4},
    "ultra-h100": {"fit_iters": 24, "shape_iters": 150, "batch_size": 256,
                   "weighted": True, "two_stage": True, "shape_frames": 4,
                   "transl_iters": 8},
}


def _zero_hand_face(
    batch: int, device: torch.device, dtype: torch.dtype, smplx_model,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z3 = torch.zeros(batch, 3, device=device, dtype=dtype)
    return (
        torch.zeros(batch, 45, device=device, dtype=dtype),
        torch.zeros(batch, 45, device=device, dtype=dtype),
        z3,
        torch.zeros(batch, smplx_model.num_expression_coeffs, device=device, dtype=dtype),
    )


def _vertex_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    vert_mask: torch.Tensor,
    vert_weights: torch.Tensor | None,
    use_weighted: bool,
) -> torch.Tensor:
    if use_weighted and vert_weights is not None:
        return weighted_vertex_loss(pred, target, vert_weights)
    return (pred[:, vert_mask] - target[:, vert_mask]).pow(2).mean()


def _smplx_forward(
    smplx_model,
    betas: torch.Tensor,
    go: torch.Tensor,
    bp: torch.Tensor,
    tr: torch.Tensor,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
):
    lh, rh, jaw, expr = _zero_hand_face(batch, device, dtype, smplx_model)
    return smplx_model(
        betas=betas, global_orient=go, body_pose=bp, transl=tr,
        left_hand_pose=lh, right_hand_pose=rh,
        jaw_pose=jaw, leye_pose=jaw, reye_pose=jaw, expression=expr,
    )


def _fit_smplx_batch(
    smplx_model,
    target_verts: torch.Tensor,
    vert_mask: torch.Tensor,
    vert_weights: torch.Tensor | None,
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    transl: torch.Tensor,
    betas: torch.Tensor,
    fit_iters: int,
    *,
    use_weighted: bool = False,
    two_stage: bool = False,
    transl_iters: int = 5,
    lr: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if fit_iters <= 0:
        return global_orient.detach(), body_pose.detach(), transl.detach()

    b = target_verts.shape[0]
    device = target_verts.device
    go = global_orient.detach().clone().requires_grad_(True)
    bp = body_pose.detach().clone().requires_grad_(True)
    tr = transl.detach().clone().requires_grad_(True)
    be = betas.detach().clone()

    if two_stage and fit_iters > transl_iters:
        opt_tr = torch.optim.Adam([tr], lr=lr * 2.0)
        for _ in range(transl_iters):
            opt_tr.zero_grad(set_to_none=True)
            out = _smplx_forward(smplx_model, be, go.detach(), bp.detach(), tr, b, device, go.dtype)
            _vertex_loss(out.vertices, target_verts, vert_mask, vert_weights, use_weighted).backward()
            opt_tr.step()

    opt = torch.optim.Adam([go, bp, tr], lr=lr)
    pose_iters = fit_iters if not two_stage else max(1, fit_iters - transl_iters)
    for _ in range(pose_iters):
        opt.zero_grad(set_to_none=True)
        out = _smplx_forward(smplx_model, be, go, bp, tr, b, device, go.dtype)
        _vertex_loss(out.vertices, target_verts, vert_mask, vert_weights, use_weighted).backward()
        opt.step()

    return go.detach(), bp.detach(), tr.detach()


def _fit_shape_betas(
    smplx_model,
    target_verts: torch.Tensor,
    vert_mask: torch.Tensor,
    vert_weights: torch.Tensor | None,
    init_body_pose: torch.Tensor,
    init_global_orient: torch.Tensor,
    init_transl: torch.Tensor,
    iters: int = 30,
    *,
    use_weighted: bool = False,
    lr: float = 0.05,
) -> torch.Tensor:
    if iters <= 0:
        return torch.zeros(1, smplx_model.num_betas, device=target_verts.device)

    betas = torch.zeros(
        1, smplx_model.num_betas, device=target_verts.device, requires_grad=True,
    )
    go = init_global_orient.detach()
    bp = init_body_pose.detach()
    tr = init_transl.detach()
    nf = target_verts.shape[0]

    opt = torch.optim.Adam([betas], lr=lr)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        out = _smplx_forward(
            smplx_model, betas.expand(nf, -1), go, bp, tr, nf, target_verts.device, go.dtype,
        )
        _vertex_loss(out.vertices, target_verts, vert_mask, vert_weights, use_weighted).backward()
        opt.step()

    return betas.detach()


def _allocate_output(n: int, timestamps: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "body_pose": np.zeros((n, 63), dtype=np.float32),
        "global_orient": np.zeros((n, 3), dtype=np.float32),
        "transl": np.zeros((n, 3), dtype=np.float32),
        "left_hand_pose": np.zeros((n, 45), dtype=np.float32),
        "right_hand_pose": np.zeros((n, 45), dtype=np.float32),
        "jaw_pose": np.zeros((n, 3), dtype=np.float32),
        "leye_pose": np.zeros((n, 3), dtype=np.float32),
        "reye_pose": np.zeros((n, 3), dtype=np.float32),
        "expression": np.zeros((n, 10), dtype=np.float32),
        "timestamps": timestamps,
    }


def _fit_shared_betas(
    motion: dict[str, np.ndarray],
    smpl_model,
    smplx_model,
    corr: dict[str, np.ndarray],
    device: torch.device,
    shape_iters: int,
    shape_frames: int = 1,
    use_weighted: bool = False,
) -> np.ndarray:
    face_idx = corr["face_idx"]
    bary = corr["bary"]
    vert_mask = torch.as_tensor(corr["mask"], device=device)
    vert_weights = torch.as_tensor(corr.get("weights", build_vertex_weights(corr["dist"])), device=device)

    n = motion["num_frames"]
    if shape_frames <= 1:
        frame_ids = [0]
    else:
        frame_ids = sorted(set(int(i) for i in np.linspace(0, n - 1, shape_frames)))

    targets = []
    bp_list, go_list, tr_list = [], [], []
    for fi in frame_ids:
        body_pose_x = smpl_body_pose_to_smplx(motion["body_pose"][fi:fi + 1])
        with torch.no_grad():
            smpl_v = smpl_model(
                betas=torch.from_numpy(motion["betas"][fi:fi + 1]).to(device),
                global_orient=torch.from_numpy(motion["global_orient"][fi:fi + 1]).to(device),
                body_pose=torch.from_numpy(motion["body_pose"][fi:fi + 1]).to(device),
                transl=torch.from_numpy(motion["transl"][fi:fi + 1]).to(device),
            ).vertices
        targets.append(transfer_smpl_to_smplx_vertices(smpl_v, smpl_model.faces, face_idx, bary))
        bp_list.append(torch.from_numpy(body_pose_x).to(device))
        go_list.append(torch.from_numpy(motion["global_orient"][fi:fi + 1]).to(device))
        tr_list.append(torch.from_numpy(motion["transl"][fi:fi + 1]).to(device))

    target_stack = torch.cat(targets, dim=0)
    print(
        f"[shape] fitting shared betas ({shape_iters} iters, "
        f"{len(frame_ids)} ref frames: {frame_ids}) ...",
        flush=True,
    )
    betas = _fit_shape_betas(
        smplx_model, target_stack, vert_mask, vert_weights,
        torch.cat(bp_list, dim=0), torch.cat(go_list, dim=0), torch.cat(tr_list, dim=0),
        iters=shape_iters, use_weighted=use_weighted,
    )
    return betas[0].cpu().numpy().astype(np.float32)


def convert_sequence(
    motion: dict[str, np.ndarray],
    smpl_model,
    smplx_model,
    corr: dict[str, np.ndarray],
    *,
    batch_size: int,
    fit_iters: int,
    shape_iters: int,
    device: torch.device,
    start_frame: int = 0,
    max_frames: int | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 20000,
    betas_shared: np.ndarray | None = None,
    log_prefix: str = "",
    use_weighted: bool = False,
    two_stage: bool = False,
    transl_iters: int = 5,
    shape_frames: int = 1,
) -> dict[str, np.ndarray]:
    n_total = motion["num_frames"]
    end = n_total if max_frames is None else min(n_total, start_frame + max_frames)
    n = end - start_frame
    if n <= 0:
        raise ValueError(f"No frames to convert (start={start_frame}, total={n_total})")

    face_idx = corr["face_idx"]
    bary = corr["bary"]
    vert_mask = torch.as_tensor(corr["mask"], device=device)
    vert_weights = torch.as_tensor(
        corr.get("weights", build_vertex_weights(corr["dist"])), device=device,
    )

    out = _allocate_output(n, motion["timestamps"][start_frame:end].copy())
    smpl_model.eval()
    smplx_model.eval()

    sl = slice(start_frame, end)
    body_pose_x = smpl_body_pose_to_smplx(motion["body_pose"][sl])
    global_orient = motion["global_orient"][sl]
    transl = motion["transl"][sl]

    if betas_shared is not None:
        betas = torch.from_numpy(betas_shared).to(device).view(1, -1)
        out["betas"] = betas_shared.astype(np.float32)
    else:
        slice_motion = {
            "betas": motion["betas"][sl],
            "body_pose": motion["body_pose"][sl],
            "global_orient": motion["global_orient"][sl],
            "transl": motion["transl"][sl],
            "num_frames": n,
        }
        out["betas"] = _fit_shared_betas(
            slice_motion, smpl_model, smplx_model, corr, device,
            shape_iters, shape_frames=shape_frames, use_weighted=use_weighted,
        )
        betas = torch.from_numpy(out["betas"]).to(device).view(1, -1)

    if fit_iters <= 0:
        print(f"{log_prefix}[convert] fast path: vectorized joint remap", flush=True)
        out["body_pose"] = body_pose_x
        out["global_orient"] = global_orient
        out["transl"] = transl
        return out

    t0 = time.time()
    written = 0
    prev_go, prev_bp, prev_tr = None, None, None

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        bs = batch_end - batch_start

        go_in = torch.from_numpy(global_orient[batch_start:batch_end]).to(device)
        bp_in = torch.from_numpy(body_pose_x[batch_start:batch_end]).to(device)
        tr_in = torch.from_numpy(transl[batch_start:batch_end]).to(device)
        bp_smpl = torch.from_numpy(
            motion["body_pose"][start_frame + batch_start:start_frame + batch_end]
        ).to(device)

        if prev_go is not None and batch_start > 0:
            go_in[0] = prev_go
            bp_in[0] = prev_bp
            tr_in[0] = prev_tr

        with torch.no_grad():
            smpl_v = smpl_model(
                betas=torch.from_numpy(
                    motion["betas"][start_frame + batch_start:start_frame + batch_end]
                ).to(device),
                global_orient=go_in,
                body_pose=bp_smpl,
                transl=tr_in,
            ).vertices
        target = transfer_smpl_to_smplx_vertices(
            smpl_v, smpl_model.faces, face_idx, bary
        )

        go, bp, tr = _fit_smplx_batch(
            smplx_model, target, vert_mask, vert_weights,
            go_in, bp_in, tr_in, betas.expand(bs, -1),
            fit_iters=fit_iters,
            use_weighted=use_weighted, two_stage=two_stage, transl_iters=transl_iters,
        )
        prev_go, prev_bp, prev_tr = go[-1:], bp[-1:], tr[-1:]

        out["global_orient"][batch_start:batch_end] = go.cpu().numpy()
        out["body_pose"][batch_start:batch_end] = bp.cpu().numpy()
        out["transl"][batch_start:batch_end] = tr.cpu().numpy()

        written += bs
        elapsed = time.time() - t0
        print(
            f"{log_prefix}[convert] {written}/{n} ({100.0 * written / n:.1f}%) | "
            f"{written / max(elapsed, 1e-6):.1f} fps",
            flush=True,
        )

        if checkpoint_path and written % checkpoint_every < batch_size:
            _save_checkpoint(checkpoint_path, out, written)

    return out


def _frame_ranges(n: int, num_workers: int) -> list[tuple[int, int]]:
    chunk = (n + num_workers - 1) // num_workers
    return [(i * chunk, min((i + 1) * chunk, n)) for i in range(num_workers)]


def _mp_worker(
    gpu_id: int,
    worker_idx: int,
    start: int,
    end: int,
    motion_path: str,
    corr_path: str,
    betas_path: str,
    out_path: str,
    gender: str,
    batch_size: int,
    fit_iters: int,
    use_weighted: bool,
    two_stage: bool,
    transl_iters: int,
) -> None:
    """Run on one GPU; writes a partial npz for frames [start, end)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import smplx

    prefix = f"[gpu{gpu_id}] "
    device = torch.device("cuda:0")
    corr_np = dict(np.load(corr_path))
    betas_shared = np.load(betas_path)

    motion_full = load_smpl_motion(Path(motion_path))
    motion = {
        "betas": motion_full["betas"][start:end],
        "body_pose": motion_full["body_pose"][start:end],
        "global_orient": motion_full["global_orient"][start:end],
        "transl": motion_full["transl"][start:end],
        "timestamps": motion_full["timestamps"][start:end],
        "num_frames": end - start,
    }

    clean_dir = ensure_clean_smpl_model(gender=gender)
    smpl_model = smplx.create(
        str(clean_dir), model_type="smpl", gender=gender, num_betas=10, batch_size=1,
    ).to(device)
    smplx_model = smplx.create(
        str(smplx_model_file(gender)),
        model_type="smplx", gender=gender, num_betas=10,
        num_expression_coeffs=10, use_pca=False, flat_hand_mean=True, batch_size=1,
    ).to(device)

    print(f"{prefix}frames {start}:{end} ({motion['num_frames']})", flush=True)
    result = convert_sequence(
        motion, smpl_model, smplx_model, corr_np,
        batch_size=batch_size, fit_iters=fit_iters, shape_iters=0,
        device=device, betas_shared=betas_shared, log_prefix=prefix,
        use_weighted=use_weighted, two_stage=two_stage, transl_iters=transl_iters,
    )
    np.savez(out_path, **result)
    print(f"{prefix}done -> {out_path}", flush=True)


def convert_sequence_multi_gpu(
    motion: dict[str, np.ndarray],
    smpl_model,
    smplx_model,
    corr: dict[str, np.ndarray],
    gpus: list[int],
    *,
    gender: str,
    batch_size: int,
    fit_iters: int,
    shape_iters: int,
    motion_path: Path,
    use_weighted: bool = False,
    two_stage: bool = False,
    transl_iters: int = 5,
    shape_frames: int = 1,
) -> dict[str, np.ndarray]:
    n = motion["num_frames"]
    ranges = _frame_ranges(n, len(gpus))
    betas_shared = _fit_shared_betas(
        motion, smpl_model, smplx_model, corr,
        torch.device(f"cuda:{gpus[0]}" if gpus[0] < torch.cuda.device_count() else "cuda:0"),
        shape_iters, shape_frames=shape_frames, use_weighted=use_weighted,
    )

    with tempfile.TemporaryDirectory(prefix="smplx_convert_") as tmp:
        tmp_path = Path(tmp)
        corr_path = tmp_path / "corr.npz"
        betas_path = tmp_path / "betas.npy"
        np.savez(corr_path, **corr)
        np.save(betas_path, betas_shared)

        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []
        out_paths: list[Path] = []

        print(f"[multi-gpu] {len(gpus)} workers: {gpus} | chunks: {ranges}", flush=True)
        for worker_idx, (gpu_id, (start, end)) in enumerate(zip(gpus, ranges)):
            if start >= end:
                continue
            out_p = tmp_path / f"chunk_{worker_idx}.npz"
            out_paths.append(out_p)
            proc = ctx.Process(
                target=_mp_worker,
                args=(
                    gpu_id, worker_idx, start, end,
                    str(motion_path), str(corr_path), str(betas_path), str(out_p),
                    gender, batch_size, fit_iters,
                    use_weighted, two_stage, transl_iters,
                ),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"Worker failed with exit code {proc.exitcode}")

        merged = _allocate_output(n, motion["timestamps"].copy())
        merged["betas"] = betas_shared
        offset = 0
        for out_p in sorted(out_paths, key=lambda p: int(p.stem.split("_")[1])):
            chunk = np.load(out_p)
            m = chunk["body_pose"].shape[0]
            for key in ("body_pose", "global_orient", "transl",
                        "left_hand_pose", "right_hand_pose",
                        "jaw_pose", "leye_pose", "reye_pose", "expression"):
                merged[key][offset:offset + m] = chunk[key]
            merged["timestamps"][offset:offset + m] = chunk["timestamps"]
            offset += m

    return merged


def _save_checkpoint(path: Path, out: dict[str, np.ndarray], n_written: int) -> None:
    partial = {
        k: v[:n_written] if v.ndim > 0 and len(v) == out["body_pose"].shape[0] else v
        for k, v in out.items()
    }
    np.savez(path, **partial)
    print(f"[checkpoint] {path} ({n_written} frames)", flush=True)


def _vertex_error_mm(
    smpl_model, smplx_model, corr, motion, result, device, frame_idx: int = 0,
) -> tuple[float, float]:
    """Return (hard-mask error mm, weighted-all-verts error mm)."""
    mask = torch.as_tensor(corr["mask"], device=device)
    weights = torch.as_tensor(corr.get("weights", build_vertex_weights(corr["dist"])), device=device)
    with torch.no_grad():
        smpl_v = smpl_model(
            betas=torch.from_numpy(motion["betas"][frame_idx:frame_idx + 1]).to(device),
            global_orient=torch.from_numpy(motion["global_orient"][frame_idx:frame_idx + 1]).to(device),
            body_pose=torch.from_numpy(motion["body_pose"][frame_idx:frame_idx + 1]).to(device),
            transl=torch.from_numpy(motion["transl"][frame_idx:frame_idx + 1]).to(device),
        ).vertices
        tgt = transfer_smpl_to_smplx_vertices(
            smpl_v, smpl_model.faces, corr["face_idx"], corr["bary"],
        )
        be = torch.from_numpy(result["betas"]).to(device)
        if be.ndim == 1:
            be = be.unsqueeze(0)
        sx = smplx_model(
            betas=be,
            global_orient=torch.from_numpy(result["global_orient"][:1]).to(device),
            body_pose=torch.from_numpy(result["body_pose"][:1]).to(device),
            transl=torch.from_numpy(result["transl"][:1]).to(device),
        ).vertices
        err_mask = (sx[:, mask] - tgt[:, mask]).pow(2).sum(-1).sqrt().mean().item() * 1000
        err_w = weighted_vertex_loss(sx, tgt, weights).sqrt().item() * 1000
        return err_mask, err_w


def _load_models(device: torch.device, gender: str):
    import smplx

    clean_dir = ensure_clean_smpl_model(gender=gender)
    smpl_model = smplx.create(
        str(clean_dir), model_type="smpl", gender=gender, num_betas=10, batch_size=1,
    ).to(device)
    smplx_model = smplx.create(
        str(smplx_model_file(gender)),
        model_type="smplx", gender=gender, num_betas=10,
        num_expression_coeffs=10, use_pca=False, flat_hand_mean=True, batch_size=1,
    ).to(device)
    return smpl_model, smplx_model


def _parse_gpus(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def run_benchmark(motion, smpl_model, smplx_model, corr, device) -> None:
    n = min(512, motion["num_frames"])
    print(f"[benchmark] {n} frames on {device}", flush=True)
    for mode, preset in MODE_PRESETS.items():
        if not mode.endswith("-h100"):
            continue
        t0 = time.time()
        result = convert_sequence(
            motion, smpl_model, smplx_model, corr,
            batch_size=min(preset["batch_size"], n),
            fit_iters=preset["fit_iters"],
            shape_iters=preset["shape_iters"],
            device=device, start_frame=0, max_frames=n,
            use_weighted=preset.get("weighted", False),
            two_stage=preset.get("two_stage", False),
            transl_iters=preset.get("transl_iters", 5),
            shape_frames=preset.get("shape_frames", 1),
        )
        elapsed = time.time() - t0
        err_mask, err_w = _vertex_error_mm(smpl_model, smplx_model, corr, motion, result, device)
        if preset["fit_iters"] <= 0:
            est_note = f"est_full~{elapsed:.0f}s"
        else:
            est_note = f"est_full~{elapsed / n * motion['num_frames'] / 60:.1f} min"
        print(
            f"  {mode:12s}  {elapsed:6.2f}s  ({n / elapsed:5.0f} fps)  "
            f"mask={err_mask:5.1f}mm  weighted={err_w:5.1f}mm  {est_note}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="<episode>/body/xdata_smpl_neutral.npz")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mode", default="quality-h100",
                        choices=list(MODE_PRESETS),
                        help="Preset (default: quality-h100 for best quality on large GPUs).")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids for parallel refine, e.g. 0,1,2,3.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fit-iters", type=int, default=None)
    parser.add_argument("--shape-iters", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rebuild-corr", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    preset = MODE_PRESETS[args.mode]
    batch_size = args.batch_size if args.batch_size is not None else preset["batch_size"]
    fit_iters = args.fit_iters if args.fit_iters is not None else preset["fit_iters"]
    shape_iters = args.shape_iters if args.shape_iters is not None else preset["shape_iters"]
    use_weighted = preset.get("weighted", False)
    two_stage = preset.get("two_stage", False)
    transl_iters = preset.get("transl_iters", 5)
    shape_frames = preset.get("shape_frames", 1)
    gpus = _parse_gpus(args.gpus)

    if gpus and fit_iters <= 0:
        print("[warn] --gpus ignored in fast mode (no per-frame refine)", flush=True)
        gpus = None

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if gpus and device.type == "cpu":
        raise RuntimeError("--gpus requires CUDA")

    inp = args.input
    if not inp.is_file():
        raise FileNotFoundError(f"SMPL motion not found: {inp}")

    if args.selftest:
        args.max_frames = 8

    if args.output is not None:
        out_path = args.output
    elif args.selftest:
        out_path = Path(tempfile.mkdtemp()) / "selftest_smplx.npz"
    else:
        out_path = default_smplx_npz(inp.parent)

    print(f"[load] {inp}")
    motion = load_smpl_motion(inp, stride=args.stride)
    n_frames = motion["num_frames"]
    gpu_note = f"gpus={gpus}" if gpus else f"device={device}"
    print(
        f"[load] {n_frames} frames | mode={args.mode} "
        f"(fit={fit_iters}, shape={shape_iters}, batch={batch_size}, "
        f"weighted={use_weighted}, two_stage={two_stage}, {gpu_note})",
        flush=True,
    )

    primary_gpu = gpus[0] if gpus else 0
    load_device = (
        torch.device(f"cuda:{primary_gpu}")
        if gpus or device.type == "cuda"
        else device
    )
    smpl_model, smplx_model = _load_models(load_device, args.gender)
    corr = load_or_build_correspondence(
        smpl_model, smplx_model, gender=args.gender,
        force_rebuild=args.rebuild_corr,
    )

    if args.benchmark:
        run_benchmark(motion, smpl_model, smplx_model, corr, load_device)
        return

    t_all = time.time()
    if gpus and len(gpus) > 1 and fit_iters > 0:
        if args.start_frame != 0 or args.max_frames is not None:
            raise NotImplementedError("Multi-GPU mode does not support --start-frame/--max-frames yet")
        result = convert_sequence_multi_gpu(
            motion, smpl_model, smplx_model, corr, gpus,
            gender=args.gender,
            batch_size=batch_size, fit_iters=fit_iters, shape_iters=shape_iters,
            motion_path=inp,
            use_weighted=use_weighted, two_stage=two_stage,
            transl_iters=transl_iters, shape_frames=shape_frames,
        )
    else:
        if gpus and len(gpus) == 1:
            load_device = torch.device(f"cuda:{gpus[0]}")
            smpl_model, smplx_model = _load_models(load_device, args.gender)
        result = convert_sequence(
            motion, smpl_model, smplx_model, corr,
            batch_size=batch_size, fit_iters=fit_iters, shape_iters=shape_iters,
            device=load_device, start_frame=args.start_frame, max_frames=args.max_frames,
            checkpoint_path=args.checkpoint,
            use_weighted=use_weighted, two_stage=two_stage,
            transl_iters=transl_iters, shape_frames=shape_frames,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **result)
    elapsed = time.time() - t_all
    n_out = result["body_pose"].shape[0]
    print(f"[save] {out_path} ({n_out} frames in {elapsed:.1f}s, {n_out / elapsed:.0f} fps)",
          flush=True)

    if args.selftest:
        err_mask, err_w = _vertex_error_mm(
            smpl_model, smplx_model, corr, motion, result, load_device,
        )
        print(f"[selftest] mask={err_mask:.2f}mm weighted={err_w:.2f}mm - OK", flush=True)


if __name__ == "__main__":
    main()
