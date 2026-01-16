"""Shared utilities for search modules."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


def set_seed(seed: Optional[int]) -> None:
    """Set random seeds across Python, NumPy, and Torch (if available)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
