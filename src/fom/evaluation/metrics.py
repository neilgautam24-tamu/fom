"""Per-sequence 3D metrics (positions in metres).

=====================  ==============================================================
Key                    Definition
=====================  ==============================================================
head_rot_error         mean ||R_pred^-1 R_gt - I||_F of the global head joint (15)
head_trans_error       mean L2 error of the left-eye joint (23)
mpjpe                  all SMPL-X joints, no alignment
mpjpe_pa               all joints, per-frame Procrustes (similarity) alignment
mpjpe_body             joints 0:22, no alignment
mpjpe_hand             joints 25:55, no alignment
traj_*                 camera trajectory errors (raw and rigidly aligned)
sliding_*, pene_*,     foot sliding, floor penetration, air time and contact validity
air_*, contact_*       for ground truth and prediction
=====================  ==============================================================
"""

import numpy as np
import torch

from fom.body.smplx_model import HAND_JOINTS, JOINT_HEAD, JOINT_LEFT_EYE, NUM_BODY_JOINTS
from fom.utils.kinematics import local_to_global_pose


def compute_metrics(gt: dict, pred: dict, parents: list[int]) -> dict[str, float]:
    """Compare two decoded sequences.

    Both dicts hold ``camera_T`` (T, 4, 4), ``joints`` (T, J, 3), ``vertices`` (T, V, 3) and
    ``full_pose`` (T, 55, 3, 3) on the CPU; ``parents`` is the SMPL-X kinematic tree.
    """
    mdata, pred_mdata = gt, pred

    leye_idx = JOINT_LEFT_EYE
    head_idx = JOINT_HEAD

    # head rotation
    head_rot = local_to_global_pose(mdata["full_pose"], parents)[:, head_idx]  # T x 3 x 3
    pred_head_rot = local_to_global_pose(pred_mdata["full_pose"], parents)[:, head_idx]  # T x 3 x 3
    diff = pred_head_rot.inverse() @ head_rot - torch.eye(3, device=head_rot.device)[None]  # T x 3 x 3
    head_rot_error = torch.norm(diff, p="fro", dim=(1, 2)).mean().item()

    # head translation
    head_trans = mdata["joints"][:, leye_idx]  # T x 3
    pred_head_trans = pred_mdata["joints"][:, leye_idx]  # T x 3
    head_trans_error = torch.norm(pred_head_trans - head_trans, dim=1).mean().item()

    # Trajectory error
    traj_T = mdata["camera_T"]  # T x 4 x 4
    pred_traj_T = pred_mdata["camera_T"]  # T x 4 x 4
    diff = pred_traj_T[:, :3, :3].inverse() @ traj_T[:, :3, :3] - torch.eye(3, device=traj_T.device)[None]  # T x 3 x 3
    traj_rot_error = torch.norm(diff, p="fro", dim=(1, 2)).mean().item()
    traj_trans_error = torch.norm(pred_traj_T[:, :3, 3] - traj_T[:, :3, 3], dim=1).mean().item()

    # Trajectory error procrustes aligned
    traj_T = torch.from_numpy(align_trajectories(pred_traj_T.numpy(), traj_T.numpy())).float()
    diff = pred_traj_T[:, :3, :3].inverse() @ traj_T[:, :3, :3] - torch.eye(3, device=traj_T.device)[None]  # T x 3 x 3
    traj_rot_error_pa = torch.norm(diff, p="fro", dim=(1, 2)).mean().item()
    traj_trans_error_pa = torch.norm(pred_traj_T[:, :3, 3] - traj_T[:, :3, 3], dim=1).mean().item()

    # joint positions
    mpjpe = (pred_mdata["joints"] - mdata["joints"]).square().sum(dim=-1).sqrt().mean().item()
    mpjpe_pa = reconstruction_error(pred_mdata["joints"].numpy(), mdata["joints"].numpy()).item()
    mpjpe_body = (pred_mdata["joints"][:, :NUM_BODY_JOINTS] - mdata["joints"][:, :NUM_BODY_JOINTS]).square().sum(dim=-1).sqrt().mean().item()
    mpjpe_body_pa = reconstruction_error(pred_mdata["joints"][:, :NUM_BODY_JOINTS].numpy(), mdata["joints"][:, :NUM_BODY_JOINTS].numpy()).item()
    mpjpe_hand = (pred_mdata["joints"][:, HAND_JOINTS] - mdata["joints"][:, HAND_JOINTS]).square().sum(dim=-1).sqrt().mean().item()
    mpjpe_hand_pa = reconstruction_error(pred_mdata["joints"][:, HAND_JOINTS].numpy(), mdata["joints"][:, HAND_JOINTS].numpy()).item()

    # foot sliding
    sliding_gt = compute_foot_sliding_for_smpl(mdata["joints"].numpy(), 0).item()
    sliding_pred = compute_foot_sliding_for_smpl(pred_mdata["joints"].numpy(), 0).item()

    # floor penetration
    pene_gt = get_foot_penetration(mdata["vertices"], 0).item()
    pene_pred = get_foot_penetration(pred_mdata["vertices"], 0).item()

    # air time
    air_gt = get_air_time(mdata["vertices"], 0).item()
    air_pred = get_air_time(pred_mdata["vertices"], 0).item()

    # contact validity
    contact_validity_gt = get_contact_validity(mdata["vertices"], 0).item()
    contact_validity_pred = get_contact_validity(pred_mdata["vertices"], 0).item()

    return {
        "head_rot_error": head_rot_error,
        "head_trans_error": head_trans_error,
        "traj_rot_error": traj_rot_error,
        "traj_trans_error": traj_trans_error,
        "traj_rot_error_pa": traj_rot_error_pa,
        "traj_trans_error_pa": traj_trans_error_pa,
        "mpjpe": mpjpe,
        "mpjpe_pa": mpjpe_pa,
        "mpjpe_body": mpjpe_body,
        "mpjpe_body_pa": mpjpe_body_pa,
        "mpjpe_hand": mpjpe_hand,
        "mpjpe_hand_pa": mpjpe_hand_pa,
        "sliding_gt": sliding_gt,
        "sliding_pred": sliding_pred,
        "pene_gt": pene_gt,
        "pene_pred": pene_pred,
        "air_gt": air_gt,
        "air_pred": air_pred,
        "contact_validity_gt": contact_validity_gt,
        "contact_validity_pred": contact_validity_pred,
    }


def align_trajectories(A, B):
    # A, B are of shape (T, 4, 4)
    T = A.shape[0]

    # Extract rotations and translations
    R_A = A[:, :3, :3]
    t_A = A[:, :3, 3]
    R_B = B[:, :3, :3]
    t_B = B[:, :3, 3]

    # Stack rotations into (3, T*3) matrices
    M_A = np.hstack([R_A[i] for i in range(T)])
    M_B = np.hstack([R_B[i] for i in range(T)])

    # Compute optimal rotation using Kabsch algorithm
    H = M_B @ M_A.T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute optimal translation
    t_A_mean = np.mean(t_A, axis=0)
    t_B_mean = np.mean(t_B, axis=0)
    t = t_A_mean - R @ t_B_mean

    # Apply the rigid transformation to trajectory B
    B_aligned = np.zeros_like(B)
    for i in range(T):
        # Apply R and t to each transformation in B
        R_new = R @ R_B[i]
        t_new = R @ t_B[i] + t

        # Reconstruct the 4x4 transformation matrix
        B_aligned[i, :3, :3] = R_new
        B_aligned[i, :3, 3] = t_new
        B_aligned[i, 3, :] = [0, 0, 0, 1]

    return B_aligned


def compute_similarity_transform(S1, S2, ret_tsfm=False):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    if ret_tsfm:
        return R, t, scale

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2, reduction="mean"):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt(((S1_hat - S2) ** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == "mean":
        re = re.mean()
    elif reduction == "sum":
        re = re.sum()
    return re


def compute_foot_sliding_for_smpl(pred_global_jpos, floor_height):
    # pred_global_jpos: T X J X 3
    seq_len = pred_global_jpos.shape[0]
    pred_global_jpos = pred_global_jpos.copy()

    # Put human mesh to floor z = 0 and compute.
    pred_global_jpos[:, :, 2] -= floor_height

    lankle_pos = pred_global_jpos[:, 7, :]  # T X 3
    ltoe_pos = pred_global_jpos[:, 10, :]  # T X 3

    rankle_pos = pred_global_jpos[:, 8, :]  # T X 3
    rtoe_pos = pred_global_jpos[:, 11, :]  # T X 3

    H_ankle = 0.08  # meter
    H_toe = 0.04  # meter

    lankle_disp = np.linalg.norm(lankle_pos[1:, :2] - lankle_pos[:-1, :2], axis=1)  # T
    ltoe_disp = np.linalg.norm(ltoe_pos[1:, :2] - ltoe_pos[:-1, :2], axis=1)  # T
    rankle_disp = np.linalg.norm(rankle_pos[1:, :2] - rankle_pos[:-1, :2], axis=1)  # T
    rtoe_disp = np.linalg.norm(rtoe_pos[1:, :2] - rtoe_pos[:-1, :2], axis=1)  # T

    lankle_subset = lankle_pos[:-1, -1] < H_ankle
    ltoe_subset = ltoe_pos[:-1, -1] < H_toe
    rankle_subset = rankle_pos[:-1, -1] < H_ankle
    rtoe_subset = rtoe_pos[:-1, -1] < H_toe

    lankle_sliding_stats = np.abs(lankle_disp * (2 - 2 ** (lankle_pos[:-1, -1] / H_ankle)))[lankle_subset]
    lankle_sliding = np.sum(lankle_sliding_stats) / seq_len * 1000

    ltoe_sliding_stats = np.abs(ltoe_disp * (2 - 2 ** (ltoe_pos[:-1, -1] / H_toe)))[ltoe_subset]
    ltoe_sliding = np.sum(ltoe_sliding_stats) / seq_len * 1000

    rankle_sliding_stats = np.abs(rankle_disp * (2 - 2 ** (rankle_pos[:-1, -1] / H_ankle)))[rankle_subset]
    rankle_sliding = np.sum(rankle_sliding_stats) / seq_len * 1000

    rtoe_sliding_stats = np.abs(rtoe_disp * (2 - 2 ** (rtoe_pos[:-1, -1] / H_toe)))[rtoe_subset]
    rtoe_sliding = np.sum(rtoe_sliding_stats) / seq_len * 1000

    sliding = (lankle_sliding + ltoe_sliding + rankle_sliding + rtoe_sliding) / 4.0

    return sliding


def get_foot_penetration(jpos, floor_height):
    # jpos: ... x J x 3
    # This does not penalize if the motion is unrealistically floating above the floor.
    jpos_z = jpos[..., 2]
    jpos_z = jpos_z - floor_height + 0.04  # ... x J

    pene = torch.where(jpos_z < 0, -jpos_z, torch.zeros_like(jpos_z))  # ... x J
    pene = pene.sum(-1) / ((pene > 0).float().sum(-1) + 1)
    return pene.mean()


def get_air_time(jpos, floor_height):
    # jpos: ... x J x 3
    # This does not penalize if motion is unrealistically penetrating the floor.
    jpos_z = jpos[..., 2]
    jpos_z = jpos_z - floor_height - 0.04  # ... x J

    air = (jpos_z > 0).all(dim=-1)  # ...
    return air.float().mean()


def get_contact_validity(jpos, floor_height):
    # jpos: ... x J x 3
    # This penalizes if the lowest z point is away from the floor (both air and penetration).
    jpos_z = jpos[..., 2]
    jpos_z = jpos_z - floor_height  # ... x J

    jpos_z_min = jpos_z.min(dim=-1)[0].abs()  # ... x 1
    return jpos_z_min.mean()
