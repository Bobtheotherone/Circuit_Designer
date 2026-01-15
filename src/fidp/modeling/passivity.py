"""Passivity checks and conservative one-port passivation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from fidp.modeling.vector_fit import RationalModel


@dataclass
class PassivityReport:
    """Passivity report for one-port data."""

    is_passive: bool
    min_real: float
    worst_freq_hz: float
    n_violations: int
    tol: float
    details: dict[str, Any] = field(default_factory=dict)


def _validate_oneport_inputs(freq_hz: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freq_hz = np.asarray(freq_hz, dtype=float)
    H = np.asarray(H, dtype=complex)
    if freq_hz.ndim != 1:
        raise ValueError("freq_hz must be a 1D array.")
    if freq_hz.shape != H.shape:
        raise ValueError("freq_hz and H must have the same shape.")
    if np.any(freq_hz < 0.0):
        raise ValueError("freq_hz must be non-negative.")
    if not np.isfinite(freq_hz).all():
        raise ValueError("freq_hz must be finite.")
    if not np.isfinite(H.real).all() or not np.isfinite(H.imag).all():
        raise ValueError("H must be finite.")
    return freq_hz, H


def check_oneport_passivity(
    freq_hz: np.ndarray,
    H: np.ndarray,
    kind: Literal["impedance", "admittance"],
    tol: float = 1e-9,
) -> PassivityReport:
    """
    Check one-port passivity by verifying Re{H(jw)} >= -tol on the grid.
    """
    if kind not in ("impedance", "admittance"):
        raise ValueError("kind must be 'impedance' or 'admittance'.")
    freq_hz, H = _validate_oneport_inputs(freq_hz, H)
    real_part = H.real
    min_real = float(np.min(real_part))
    idx = int(np.argmin(real_part))
    worst_freq = float(freq_hz[idx])
    n_violations = int(np.sum(real_part < -tol))
    is_passive = n_violations == 0

    details = {
        "kind": kind,
        "n_samples": int(freq_hz.size),
    }
    return PassivityReport(
        is_passive=is_passive,
        min_real=min_real,
        worst_freq_hz=worst_freq,
        n_violations=n_violations,
        tol=float(tol),
        details=details,
    )


def passivate_oneport_min_offset(
    model: RationalModel,
    freq_hz: np.ndarray,
    tol: float = 1e-9,
) -> tuple[RationalModel, PassivityReport]:
    """
    Enforce passivity by adding the minimum series/shunt offset on the grid.
    """
    H = model.eval_freq(freq_hz)
    report = check_oneport_passivity(freq_hz, H, model.kind, tol=tol)
    delta = 0.0
    if report.min_real < -tol:
        delta = -(report.min_real) + tol

    if delta > 0.0:
        updated = RationalModel(
            poles=model.poles.copy(),
            residues=model.residues.copy(),
            d=model.d + delta,
            h=model.h,
            kind=model.kind,
        )
    else:
        updated = RationalModel(
            poles=model.poles.copy(),
            residues=model.residues.copy(),
            d=model.d,
            h=model.h,
            kind=model.kind,
        )

    updated_report = check_oneport_passivity(freq_hz, updated.eval_freq(freq_hz), updated.kind, tol=tol)
    updated_report.details["delta_offset"] = float(delta)
    return updated, updated_report
