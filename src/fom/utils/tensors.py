"""Small helpers for moving nested containers of tensors around."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import default_collate


def recursive_apply(x: Any, func: Callable[[Any], Any]) -> Any:
    """Apply ``func`` to every leaf of nested dicts / lists / tuples."""
    if isinstance(x, dict):
        return {k: recursive_apply(v, func) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_apply(i, func) for i in x]
    if isinstance(x, tuple):
        return tuple(recursive_apply(i, func) for i in x)
    if callable(x):
        return x
    return func(x)


def _to_tensor(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray) and x.dtype.type != np.str_:
        x = torch.from_numpy(x)
        return x.float() if x.dtype == torch.double else x
    return x


def _to_numpy(x: Any) -> Any:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


def _to_device(x: Any, device: torch.device | str) -> Any:
    return x.to(device) if isinstance(x, torch.Tensor) else x


def to_tensor(x: Any) -> Any:
    return recursive_apply(x, _to_tensor)


def to_numpy(x: Any) -> Any:
    return recursive_apply(x, _to_numpy)


def to_device(x: Any, device: torch.device | str) -> Any:
    return recursive_apply(x, partial(_to_device, device=device))


def collate(batch: list[dict]) -> dict:
    """Default collation; kept as a named function so DataLoaders stay picklable."""
    return default_collate(batch)
