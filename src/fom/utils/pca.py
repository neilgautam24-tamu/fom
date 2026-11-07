"""Conversion between MANO hand rotations and SMPL-X hand PCA coefficients."""

from __future__ import annotations

import torch

from fom.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle


def matrix_to_pca(hand_pose: torch.Tensor, hand_components: torch.Tensor) -> torch.Tensor:
    """``(B, 15, 3, 3)`` rotations -> ``(B, C)`` PCA coefficients (``hand_components``: ``(C, 45)``)."""
    hand_pose = matrix_to_axis_angle(hand_pose).flatten(-2, -1)
    hand_pose = torch.einsum("...i,ij->...j", [hand_pose, hand_components.T])
    return hand_pose / hand_components.square().sum(-1)


def pca_to_matrix(hand_pose: torch.Tensor, hand_components: torch.Tensor) -> torch.Tensor:
    """``(B, C)`` PCA coefficients -> ``(B, 15, 3, 3)`` rotations."""
    hand_pose = torch.einsum("...i,ij->...j", [hand_pose, hand_components])
    return axis_angle_to_matrix(hand_pose.unflatten(-1, (15, 3)))
