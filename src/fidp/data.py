"""Shared data objects for evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np
import scipy.sparse as sp


@dataclass
class ImpedanceSweep:
    """Frequency-domain impedance data."""

    freqs_hz: np.ndarray
    Z: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        freqs = np.asarray(self.freqs_hz)
        Z = np.asarray(self.Z)
        if freqs.ndim != 1:
            raise ValueError("freqs_hz must be a 1D array.")
        if Z.ndim == 1:
            if freqs.shape != Z.shape:
                raise ValueError("freqs_hz and Z must have the same shape.")
        else:
            if Z.shape[0] != freqs.shape[0]:
                raise ValueError("Z must have the same leading dimension as freqs_hz.")


@dataclass
class DescriptorSystem:
    """Descriptor-form system (G + sC) x = B with output Z(s) = L^T x."""

    G: sp.spmatrix
    C: sp.spmatrix
    B: np.ndarray
    L: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.G.shape != self.C.shape:
            raise ValueError("G and C must have the same shape.")
        if self.G.shape[0] != self.B.shape[0] or self.G.shape[0] != self.L.shape[0]:
            raise ValueError("G/C dimension must match B and L rows.")


@dataclass
class ReducedDescriptorSystem:
    """Reduced descriptor system after MOR."""

    G_r: np.ndarray
    C_r: np.ndarray
    B_r: np.ndarray
    L_r: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.G_r.shape != self.C_r.shape:
            raise ValueError("G_r and C_r must have the same shape.")
        if self.G_r.shape[0] != self.B_r.shape[0] or self.G_r.shape[0] != self.L_r.shape[0]:
            raise ValueError("Reduced dimensions must match B_r and L_r rows.")
