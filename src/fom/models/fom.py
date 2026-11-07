"""Foresight to Motion: causal autoregressive decoder from egocentric observations to SMPL-X motion.

Conditioning dictionary ``y`` (all tensors batch-first, ``T`` = motion frames)::

    valid_frames  (B, T)          1 for real frames, 0 for padding           [required]
    traj          (B, T, 18)      normalized camera trajectory features      [optional]
    traj_mask     (B, T)          1 = replace trajectory by the mask token   [optional]
    rgb           (B, T, 1024)    DINOv2 CLS features                        [optional]
    rgb_valid     (B, T)                                                     [with rgb]
    rgb_mask      (B, T)          1 = replace RGB by the mask token          [optional]
    depth         (B, T, 5, 1024) depth patch tokens                         [optional]
    depth_valid   (B, T)                                                     [with depth]
    goal          (B, T, 256)     flattened 16x16 goal heatmap               [optional]

The trajectory is added to the motion query stream; RGB, depth and goal tokens form the
cross-attention context, causally masked per frame (query ``t`` sees context frames ``<= t``).
A missing key means "modality unavailable" and the learned mask token / nothing is used instead.

Parameter names are part of the checkpoint format; do not rename module attributes.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from yacs.config import CfgNode

from fom.body.representation import MOTION_DIM, TRAJ_DIM
from fom.models.transformer import DecoderBlock, KVPair

RGB_FEAT_DIM = 1024  # DINOv2 ViT-L/14 CLS token
DEPTH_PATCHES_PER_FRAME = 5
DEPTH_PATCH_DIM = 1024  # 32 x 32 log-depth values
GOAL_HEATMAP_DIM = 256  # 16 x 16 Gaussian heatmap

# (tokens_per_frame, num_tokens) for each block of the cross-attention context.
ContextLayout = list[tuple[int, int]]

_PER_FRAME_KEYS = ("traj", "rgb", "valid_frames", "rgb_valid", "traj_mask", "rgb_mask", "goal", "depth", "depth_valid")


def _apply_mask(mask: torch.Tensor, cond: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    """Blend ``cond`` towards ``token`` where ``mask`` is 1. ``mask``: ``(B,)`` or ``(B, T)``."""
    mask = mask.view(*mask.shape, *([1] * (cond.ndim - mask.ndim))).float()
    token = token.view(*([1] * (cond.ndim - 1)), -1)
    return cond * (1.0 - mask) + mask * token


def _sequence_dropout(x: torch.Tensor, prob: float, token: torch.Tensor) -> torch.Tensor:
    """Replace the whole sequence of each batch element by ``token`` with probability ``prob``."""
    batch = x.shape[0]
    drop = (torch.rand(batch, device=x.device) < prob).float().view(batch, *([1] * (x.ndim - 1)))
    return x * (1.0 - drop) + drop * token


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe[None])  # 1 x max_len x D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.shape[1], :])


class ForesightToMotion(nn.Module):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        model_cfg = cfg.MODEL
        dim = model_cfg.LATENT_DIM
        self.latent_dim = dim
        self.input_feats = MOTION_DIM
        self.cond_traj = model_cfg.COND_TRAJ
        self.cond_depth = model_cfg.COND_DEPTH
        self.cond_goal = model_cfg.COND_GOAL
        self.use_kv_cache = model_cfg.USE_KV_CACHE
        masking = model_cfg.MASKING
        self.mask_prob = {
            "traj": masking.TRAJ,
            "rgb": masking.RGB,
            "subsequent": masking.SUBSEQUENT,
            "depth": masking.DEPTH,
            "goal": masking.GOAL,
        }

        self.start_motion_token = nn.Parameter(torch.zeros(MOTION_DIM))
        self.input_process = nn.Linear(MOTION_DIM, dim)
        self.pos_enc = PositionalEncoding(dim, model_cfg.DROPOUT)
        self.embed_traj_cond = nn.Linear(TRAJ_DIM, dim)
        self.embed_clip_cond = nn.Linear(RGB_FEAT_DIM, dim)  # RGB (DINOv2 CLS)
        if self.cond_depth:
            self.embed_depth = nn.Linear(DEPTH_PATCH_DIM, dim)
            self.depth_mask_token = nn.Parameter(torch.randn(dim) * 0.05)
        self.mask_tokens = nn.ParameterDict(
            {
                "traj": nn.Parameter(torch.randn(dim) * 0.05),
                "clip": nn.Parameter(torch.randn(dim) * 0.05),  # RGB mask token
            }
        )
        if self.cond_goal:
            self.embed_goal_heatmap = nn.Linear(GOAL_HEATMAP_DIM, dim)
            self.goal_mask_token = nn.Parameter(torch.randn(dim) * 0.05)
        self.tsfm = nn.ModuleList(
            [
                DecoderBlock(dim, model_cfg.NUM_HEADS, model_cfg.DROPOUT, model_cfg.FF_MULT)
                for _ in range(model_cfg.NUM_LAYERS)
            ]
        )
        self.output_process = nn.Linear(dim, MOTION_DIM)

    def _mask_modality(self, cond: torch.Tensor, kind: str, explicit_mask: torch.Tensor | None) -> torch.Tensor:
        """Apply an explicit mask, or (training only) random modality dropout.

        ``kind`` is ``"traj"`` or ``"rgb"``. Random dropout first masks the whole sequence with
        ``mask_prob[kind]``; a second draw with ``mask_prob["subsequent"]`` masks the whole
        trajectory again, or every RGB frame except the first.
        """
        token = self.mask_tokens["traj" if kind == "traj" else "clip"]
        if explicit_mask is not None:
            return _apply_mask(explicit_mask, cond, token)
        if not self.training:
            return cond

        batch = cond.shape[0]
        cond = _apply_mask(torch.rand(batch, device=cond.device) < self.mask_prob[kind], cond, token)
        second = torch.rand(batch, device=cond.device) < self.mask_prob["subsequent"]
        if kind == "traj":
            return _apply_mask(second, cond, token)
        return torch.cat((cond[:, :1], _apply_mask(second, cond[:, 1:], token)), dim=1)

    def _embed_query_conditions(self, x: torch.Tensor, y: dict, t: int | None = None) -> torch.Tensor:
        """Add trajectory conditioning to motion query tokens (all frames, or frame ``t`` only)."""
        if not ("traj" in y and self.cond_traj):
            return x + self.mask_tokens["traj"]

        if t is None:
            enc = self.embed_traj_cond(y["traj"])
            return x + self._mask_modality(enc, "traj", y.get("traj_mask"))

        # Single step (KV-cached rollout): only explicit masks are applied.
        enc = self.embed_traj_cond(y["traj"][:, t : t + 1])
        if y.get("traj_mask") is not None:
            mask = y["traj_mask"][:, t : t + 1].float().view(-1, 1, 1)
            enc = enc * (1.0 - mask) + mask * self.mask_tokens["traj"].view(1, 1, -1)
        return x + enc

    def _build_context(
        self, y: dict, batch: int, num_frames: int, with_valid: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None, ContextLayout]:
        """Concatenate RGB, depth and goal tokens into the cross-attention context.

        Returns the context ``(B, C, D)``, its validity ``(B, C)`` (if ``with_valid``) and the
        per-block layout used for causal masking.
        """
        # Context tensors may be longer than the motion being decoded (padded window during
        # rollout), so every block's length is taken from its own tensor.
        if "rgb" in y:
            rgb = self._mask_modality(self.embed_clip_cond(y["rgb"]), "rgb", y.get("rgb_mask"))
            valid = y["rgb_valid"] if with_valid else None
        else:
            rgb = self.mask_tokens["clip"].view(1, 1, -1).repeat(batch, num_frames, 1)
            valid = y["valid_frames"] if with_valid else None
        blocks = [self.pos_enc(rgb)]
        valids = [valid]
        layout: ContextLayout = [(1, rgb.shape[1])]

        if "depth" in y and self.cond_depth:
            depth = self.embed_depth(y["depth"])  # B x T x 5 x D
            if self.training:
                depth = _sequence_dropout(depth, self.mask_prob["depth"], self.depth_mask_token)
            blocks.append(depth.flatten(1, 2))
            layout.append((DEPTH_PATCHES_PER_FRAME, DEPTH_PATCHES_PER_FRAME * depth.shape[1]))
            if with_valid:
                depth_valid = y.get("depth_valid", torch.ones(batch, depth.shape[1], device=depth.device))
                valids.append(depth_valid[:, :, None].repeat(1, 1, DEPTH_PATCHES_PER_FRAME).flatten(1, 2))

        if "goal" in y and self.cond_goal:
            goal = self.embed_goal_heatmap(y["goal"])  # B x T x D
            if self.training:
                goal = _sequence_dropout(goal, self.mask_prob["goal"], self.goal_mask_token)
            blocks.append(goal)
            layout.append((1, goal.shape[1]))
            if with_valid:
                valids.append(torch.ones(batch, goal.shape[1], device=goal.device))

        context = torch.cat(blocks, dim=1)
        context_valid = torch.cat(valids, dim=1) if with_valid else None
        return context, context_valid, layout

    @staticmethod
    def _causal_cross_mask(num_frames: int, layout: ContextLayout, device: torch.device) -> torch.Tensor:
        """``(T, C)`` bool mask, True where query frame ``t`` would see a context frame ``> t``."""
        query_idx = torch.arange(num_frames, device=device)
        masks = []
        for tokens_per_frame, num_tokens in layout:
            frame_idx = torch.arange(num_tokens, device=device) // tokens_per_frame
            masks.append(frame_idx.unsqueeze(0) > query_idx.unsqueeze(1))
        return torch.cat(masks, dim=1)

    def _forward_sequence(self, x: torch.Tensor, y: dict) -> torch.Tensor:
        """``x[:, t]`` is the AR input used to predict frame ``t``. Returns ``(B, T, MOTION_DIM)``."""
        batch, num_frames, num_feats = x.shape
        x = self.pos_enc(self.input_process(x))
        x = self._embed_query_conditions(x, y)
        context, context_valid, layout = self._build_context(y, batch, num_frames, with_valid=True)

        self_mask = torch.triu(torch.ones(num_frames, num_frames, device=x.device, dtype=torch.bool), diagonal=1)
        cross_mask = self._causal_cross_mask(num_frames, layout, x.device)
        for block in self.tsfm:
            x = block(
                x,
                context=context,
                query_valid=y["valid_frames"],
                context_valid=context_valid,
                self_attn_mask=self_mask,
                cross_attn_mask=cross_mask,
            )
        return self.output_process(x).view(batch, num_frames, num_feats)

    def _shift_right(self, x: torch.Tensor) -> torch.Tensor:
        x_inp = torch.zeros_like(x)
        x_inp[:, 0] = self.start_motion_token.view(1, -1).expand(x.shape[0], -1)
        x_inp[:, 1:] = x[:, :-1]
        return x_inp

    @staticmethod
    def _slice_conditions(y: dict, num_frames: int) -> dict:
        y = copy.copy(y)
        for key in _PER_FRAME_KEYS:
            if key in y and isinstance(y[key], torch.Tensor) and y[key].ndim >= 2:
                y[key] = y[key][:, :num_frames]
        return y

    def _cached_step(
        self, x_t: torch.Tensor, t: int, y: dict, self_kv: list[KVPair | None], cross_kv: list[KVPair]
    ) -> tuple[torch.Tensor, list[KVPair]]:
        x = self.input_process(x_t.unsqueeze(1)) + self.pos_enc.pe[:, t : t + 1, :]
        x = self._embed_query_conditions(x, y, t=t)
        new_kv = []
        for block, past, ctx in zip(self.tsfm, self_kv, cross_kv):
            x, kv = block.forward_step(x, past, ctx)
            new_kv.append(kv)
        return self.output_process(x).squeeze(1), new_kv

    @staticmethod
    def _visible_cross_kv(cross_kv: list[KVPair], layout: ContextLayout, t: int) -> list[KVPair]:
        """Restrict precomputed context K/V to the frames visible at step ``t``."""
        spans, start = [], 0
        for tokens_per_frame, num_tokens in layout:
            spans.append((start, start + (t + 1) * tokens_per_frame))
            start += num_tokens
        return [
            (torch.cat([k[:, s:e] for s, e in spans], dim=1), torch.cat([v[:, s:e] for s, e in spans], dim=1))
            for k, v in cross_kv
        ]

    def _rollout(self, y: dict, batch: int, num_frames: int, detach_inputs: bool = False) -> torch.Tensor:
        """Closed-loop autoregressive rollout.

        With ``detach_inputs`` (self-forcing training) each step's prediction is fed back with a
        stop-gradient, so gradients flow through every step's output but not across steps.
        """
        outputs = []
        prev = self.start_motion_token.view(1, -1).expand(batch, -1)

        if self.use_kv_cache:
            context, _, layout = self._build_context(y, batch, num_frames, with_valid=False)
            cross_kv = [block.attn2.project_kv(context) for block in self.tsfm]
            self_kv: list[KVPair | None] = [None] * len(self.tsfm)
            for t in range(num_frames):
                pred, self_kv = self._cached_step(prev, t, y, self_kv, self._visible_cross_kv(cross_kv, layout, t))
                outputs.append(pred)
                prev = pred.detach() if detach_inputs else pred
            return torch.stack(outputs, dim=1)

        inputs = []
        for t in range(num_frames):
            inputs.append(prev)
            pred = self._forward_sequence(torch.stack(inputs, dim=1), self._slice_conditions(y, t + 1))[:, -1]
            outputs.append(pred)
            prev = pred.detach() if detach_inputs else pred
        return torch.stack(outputs, dim=1)

    def forward_train(self, x: torch.Tensor, y: dict, forcing: str = "teacher") -> torch.Tensor:
        """Predict all frames of the normalized target motion ``x`` ``(B, T, MOTION_DIM)``.

        ``forcing="teacher"`` feeds ground-truth previous frames (parallel);
        ``forcing="self"`` feeds the model's own stop-gradient predictions.
        """
        if forcing == "teacher":
            return self._forward_sequence(self._shift_right(x), y)
        if forcing == "self":
            return self._rollout(y, x.shape[0], x.shape[1], detach_inputs=True)
        raise ValueError(f"Unknown forcing mode {forcing!r}")

    @torch.no_grad()
    def sample(self, y: dict, batch: int, num_frames: int) -> torch.Tensor:
        """Closed-loop generation of ``num_frames`` normalized motion frames."""
        return self._rollout(y, batch, num_frames)
