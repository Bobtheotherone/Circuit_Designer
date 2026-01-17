"""Passivity tests for impedance/admittance data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from fidp.analysis.fitting.vector_fitting import RationalModel
from fidp.errors import InvalidFrequencyGridError


@dataclass(frozen=True)
class PassivityReport:
    """Passivity report over a frequency grid."""

    margin: float
    worst_freq_hz: float
    violation_bands: list[tuple[float, float]]
    grid_used: list[float]
    method: str
    tol: float
    n_violations: int
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_passive(self) -> bool:
        return self.n_violations == 0 and self.margin >= -self.tol

    def to_dict(self) -> dict[str, Any]:
        return {
            "margin": float(self.margin),
            "worst_freq_hz": float(self.worst_freq_hz),
            "violation_bands": [list(band) for band in self.violation_bands],
            "grid_used": list(self.grid_used),
            "method": self.method,
            "tol": float(self.tol),
            "n_violations": int(self.n_violations),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PoleGroup:
    """Grouped poles for conjugate symmetry enforcement."""

    indices: tuple[int, ...]
    pole: complex
    is_complex_pair: bool


def _validate_frequency_grid(freq_hz: np.ndarray) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=float)
    if freq_hz.ndim != 1:
        raise InvalidFrequencyGridError("freq_hz must be a 1D array.")
    if not np.isfinite(freq_hz).all():
        raise InvalidFrequencyGridError("freq_hz must be finite.")
    if np.any(freq_hz <= 0.0):
        raise InvalidFrequencyGridError("freq_hz must be strictly positive.")
    if np.any(np.diff(freq_hz) <= 0.0):
        raise InvalidFrequencyGridError("freq_hz must be strictly increasing.")
    return freq_hz


def _margin_curve(freq_hz: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=complex)
    if Z.ndim == 1:
        if Z.shape != freq_hz.shape:
            raise ValueError("Z must match freq_hz shape for one-port.")
        return Z.real
    if Z.ndim != 3 or Z.shape[0] != freq_hz.shape[0] or Z.shape[1] != Z.shape[2]:
        raise ValueError("Z must have shape (n_freq, n_ports, n_ports) for multiport.")
    min_eigs = np.empty(freq_hz.size, dtype=float)
    for idx in range(freq_hz.size):
        hermitian = 0.5 * (Z[idx] + Z[idx].conj().T)
        eigs = np.linalg.eigvalsh(hermitian)
        min_eigs[idx] = float(np.min(eigs.real))
    return min_eigs


def _violation_bands(freq_hz: np.ndarray, margin_curve: np.ndarray, tol: float) -> list[tuple[float, float]]:
    mask = margin_curve < -tol
    if not np.any(mask):
        return []
    idx = np.where(mask)[0]
    bands: list[tuple[float, float]] = []
    start = idx[0]
    prev = idx[0]
    for current in idx[1:]:
        if current != prev + 1:
            bands.append((float(freq_hz[start]), float(freq_hz[prev])))
            start = current
        prev = current
    bands.append((float(freq_hz[start]), float(freq_hz[prev])))
    return bands


def _refine_grid(freq_hz: np.ndarray, margin_curve: np.ndarray, refine_tol: float, max_points: int) -> np.ndarray:
    candidates: list[float] = []
    for idx in range(freq_hz.size - 1):
        m0 = margin_curve[idx]
        m1 = margin_curve[idx + 1]
        if np.sign(m0) != np.sign(m1) or abs(m0) <= refine_tol or abs(m1) <= refine_tol:
            candidates.append(0.5 * (freq_hz[idx] + freq_hz[idx + 1]))
    if not candidates:
        return np.array([], dtype=float)
    candidates = sorted(set(candidates))
    if len(candidates) > max_points:
        candidates = candidates[:max_points]
    return np.array(candidates, dtype=float)


def check_passivity(
    freq_hz: np.ndarray,
    Z: np.ndarray | None = None,
    *,
    model: RationalModel | None = None,
    tol: float = 1e-9,
    refine: bool = True,
    refine_tol: float = 1e-4,
    max_refinements: int = 2,
    max_refine_points: int = 32,
    method: str = "grid",
) -> PassivityReport:
    """Check passivity over a frequency grid with optional refinement."""
    freq_hz = _validate_frequency_grid(freq_hz)
    if model is None and Z is None:
        raise ValueError("Either model or Z must be provided.")
    if model is None:
        refine = False

    def evaluate(target_freq: np.ndarray) -> np.ndarray:
        if model is not None:
            return model.eval_freq(target_freq)
        if Z is None:
            raise ValueError("Z data missing.")
        if target_freq.shape != freq_hz.shape:
            raise ValueError("Z data cannot be interpolated without a model.")
        return np.asarray(Z, dtype=complex)

    current_freq = freq_hz
    current_Z = evaluate(current_freq)
    margin_curve = _margin_curve(current_freq, current_Z)

    for _ in range(max_refinements if refine else 0):
        new_points = _refine_grid(current_freq, margin_curve, refine_tol, max_refine_points)
        if new_points.size == 0:
            break
        current_freq = np.unique(np.concatenate([current_freq, new_points]))
        current_Z = evaluate(current_freq)
        margin_curve = _margin_curve(current_freq, current_Z)

    margin = float(np.min(margin_curve))
    idx = int(np.argmin(margin_curve))
    worst_freq = float(current_freq[idx])
    n_violations = int(np.sum(margin_curve < -tol))
    bands = _violation_bands(current_freq, margin_curve, tol)

    details = {
        "mode": "oneport" if current_Z.ndim == 1 else "multiport",
        "n_samples": int(current_freq.size),
    }
    if current_Z.ndim == 3:
        details["n_ports"] = int(current_Z.shape[1])

    return PassivityReport(
        margin=margin,
        worst_freq_hz=worst_freq,
        violation_bands=bands,
        grid_used=[float(x) for x in current_freq],
        method=method,
        tol=float(tol),
        n_violations=n_violations,
        details=details,
    )


def group_conjugate_poles(poles: Iterable[complex], tol: float = 1e-6) -> list[PoleGroup]:
    poles_arr = np.asarray(list(poles), dtype=complex)
    order = np.lexsort((poles_arr.real, poles_arr.imag))
    used = np.zeros(poles_arr.size, dtype=bool)
    groups: list[PoleGroup] = []

    for idx in order:
        if used[idx]:
            continue
        pole = poles_arr[idx]
        if abs(pole.imag) <= tol:
            groups.append(PoleGroup(indices=(int(idx),), pole=complex(pole.real, 0.0), is_complex_pair=False))
            used[idx] = True
            continue
        if pole.imag < -tol:
            continue
        target = np.conj(pole)
        candidates = [j for j in order if (not used[j]) and poles_arr[j].imag < -tol]
        if not candidates:
            groups.append(PoleGroup(indices=(int(idx),), pole=pole, is_complex_pair=False))
            used[idx] = True
            continue
        diffs = np.abs(poles_arr[candidates] - target)
        partner = candidates[int(np.argmin(diffs))]
        groups.append(PoleGroup(indices=(int(idx), int(partner)), pole=pole, is_complex_pair=True))
        used[idx] = True
        used[partner] = True

    return groups
