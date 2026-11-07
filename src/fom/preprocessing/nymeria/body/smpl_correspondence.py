"""SMPL -> SMPL-X surface correspondence (T-pose, cached)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh

from fom.preprocessing.nymeria.io.paths import correspondence_cache_dir


def correspondence_cache_path(gender: str = "neutral") -> Path:
    cache = correspondence_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"smpl2smplx_corr_{gender}.npz"


def build_vertex_weights(dist: np.ndarray, sigma_m: float = 0.025) -> np.ndarray:
    """Soft weights from T-pose correspondence distance (1.0 on body, lower on face/hands)."""
    w = np.exp(-np.square(dist / sigma_m, dtype=np.float64)).astype(np.float32)
    return np.clip(w, 0.05, 1.0)


def build_smpl_to_smplx_correspondence(
    smpl_model,
    smplx_model,
    max_dist_m: float = 0.12,
    weight_sigma_m: float = 0.025,
) -> dict[str, np.ndarray]:
    """For each SMPL-X vertex, store SMPL triangle + barycentrics + soft weights."""
    with torch.no_grad():
        smpl_out = smpl_model()
        smplx_out = smplx_model()

    smpl_verts = smpl_out.vertices[0].cpu().numpy()
    smpl_faces = smpl_model.faces
    smplx_verts = smplx_out.vertices[0].cpu().numpy()

    mesh = trimesh.Trimesh(smpl_verts, smpl_faces, process=False)
    closest, dist, face_idx = trimesh.proximity.closest_point(mesh, smplx_verts)
    tri_verts = smpl_verts[smpl_faces[face_idx]]
    bary = trimesh.triangles.points_to_barycentric(tri_verts, closest)

    mask = dist < max_dist_m
    weights = build_vertex_weights(dist, sigma_m=weight_sigma_m)
    return {
        "face_idx": face_idx.astype(np.int32),
        "bary": bary.astype(np.float32),
        "mask": mask.astype(np.bool_),
        "dist": dist.astype(np.float32),
        "weights": weights,
    }


def load_or_build_correspondence(
    smpl_model,
    smplx_model,
    gender: str = "neutral",
    max_dist_m: float = 0.12,
    weight_sigma_m: float = 0.025,
    force_rebuild: bool = False,
) -> dict[str, np.ndarray]:
    cache = correspondence_cache_path(gender)
    if cache.is_file() and not force_rebuild:
        data = np.load(cache)
        corr = {k: data[k] for k in data.files}
        if "weights" not in corr:
            corr["weights"] = build_vertex_weights(corr["dist"], sigma_m=weight_sigma_m)
        return corr

    corr = build_smpl_to_smplx_correspondence(
        smpl_model, smplx_model,
        max_dist_m=max_dist_m, weight_sigma_m=weight_sigma_m,
    )
    np.savez(cache, **corr)
    valid = int(corr["mask"].sum())
    wmean = float(corr["weights"].mean())
    print(
        f"[corr] cached {cache.name}: "
        f"{valid}/{len(corr['mask'])} hard-mask verts | mean weight={wmean:.2f}",
        flush=True,
    )
    return corr


def transfer_smpl_to_smplx_vertices(
    smpl_vertices: torch.Tensor,
    smpl_faces: np.ndarray,
    face_idx: np.ndarray,
    bary: np.ndarray,
) -> torch.Tensor:
    """Map posed SMPL vertices to SMPL-X topology. smpl_vertices: (B, V, 3)."""
    faces = torch.as_tensor(smpl_faces, device=smpl_vertices.device, dtype=torch.long)
    idx = torch.as_tensor(face_idx, device=smpl_vertices.device, dtype=torch.long)
    bc = torch.as_tensor(bary, device=smpl_vertices.device, dtype=smpl_vertices.dtype)

    tri = smpl_vertices[:, faces[idx]]  # (B, N, 3, 3)
    return (tri * bc.unsqueeze(0).unsqueeze(-1)).sum(dim=2)


def weighted_vertex_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted mean squared error over all SMPL-X vertices. weights shape (N,)."""
    diff = (pred - target).pow(2).sum(dim=-1)  # (B, N)
    w = weights.unsqueeze(0)
    return (diff * w).sum() / (w.sum() * pred.shape[0] + 1e-8)
