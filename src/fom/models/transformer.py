"""Pre-norm Transformer decoder blocks with optional single-step KV-cached execution.

Attribute names (``attn1``, ``attn2``, ``ff.net``, ``attn.attn`` ...) are part of the checkpoint
format; do not rename them.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

KVPair = tuple[torch.Tensor, torch.Tensor]


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Pre-norm multi-head attention (queries are normalized, keys/values are not)."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_x: torch.Tensor,
        key_valid: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, Tq, D)`` queries.
            kv_x: ``(B, Tk, D)`` keys/values.
            key_valid: ``(B, Tk)`` 1 for valid keys, 0 for padding.
            attn_mask: ``(Tq, Tk)`` bool, True = blocked.
        """
        x = self.norm(x)
        padding = key_valid.bool().logical_not() if key_valid is not None else None
        out = self.attn(x, kv_x, kv_x, key_padding_mask=padding, attn_mask=attn_mask)[0]
        return self.out_dropout(out)

    def forward_step(
        self, x_new: torch.Tensor, self_kv: KVPair | None = None, cross_kv: KVPair | None = None
    ) -> tuple[torch.Tensor, KVPair]:
        """Attend from a single new token ``(B, 1, D)``.

        Self-attention (``cross_kv is None``): K/V of the new token are appended to ``self_kv``.
        Cross-attention: K/V are the precomputed ``cross_kv`` (returned unchanged).
        """
        batch, _, dim = x_new.shape
        num_heads = self.attn.num_heads
        head_dim = dim // num_heads
        x_norm = self.norm(x_new)

        if cross_kv is None:
            q, k_new, v_new = F.linear(x_norm, self.attn.in_proj_weight, self.attn.in_proj_bias).split(dim, dim=-1)
            k = torch.cat([self_kv[0], k_new], dim=1) if self_kv is not None else k_new
            v = torch.cat([self_kv[1], v_new], dim=1) if self_kv is not None else v_new
            new_kv = (k, v)
        else:
            q = F.linear(x_norm, self.attn.in_proj_weight[:dim], self.attn.in_proj_bias[:dim])
            k, v = cross_kv
            new_kv = cross_kv

        q = q.view(batch, -1, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, -1, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch, -1, num_heads, head_dim).transpose(1, 2)

        weights = torch.matmul(q, k.transpose(-2, -1)) * (math.sqrt(head_dim) ** -1)
        weights = F.dropout(F.softmax(weights, dim=-1), p=self.attn.dropout, training=self.training)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, 1, dim)
        return self.out_dropout(self.attn.out_proj(out)), new_kv

    def project_kv(self, context: torch.Tensor) -> KVPair:
        """Key/value projections of ``context`` for :meth:`forward_step` cross-attention."""
        dim = self.attn.embed_dim
        kv = F.linear(context, self.attn.in_proj_weight[dim:], self.attn.in_proj_bias[dim:])
        k, v = kv.split(dim, dim=-1)
        return k, v


class DecoderBlock(nn.Module):
    """Causal self-attention -> cross-attention to the conditioning context -> feed-forward."""

    def __init__(self, dim: int, heads: int, dropout: float, ff_mult: int):
        super().__init__()
        self.attn1 = Attention(dim, heads, dropout)
        self.attn2 = Attention(dim, heads, dropout)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        query_valid: torch.Tensor | None = None,
        context_valid: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.attn1(x, kv_x=x, key_valid=query_valid, attn_mask=self_attn_mask) + x
        x = self.attn2(x, kv_x=context, key_valid=context_valid, attn_mask=cross_attn_mask) + x
        return self.ff(x) + x

    def forward_step(self, x: torch.Tensor, self_kv: KVPair | None, cross_kv: KVPair) -> tuple[torch.Tensor, KVPair]:
        out, new_self_kv = self.attn1.forward_step(x, self_kv=self_kv)
        x = out + x
        out, _ = self.attn2.forward_step(x, cross_kv=cross_kv)
        x = out + x
        return self.ff(x) + x, new_self_kv
