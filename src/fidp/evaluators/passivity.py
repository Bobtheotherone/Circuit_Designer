"""Passivity and stability checks for evaluator outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as la

from fidp.evaluators.types import StateSpaceModel, StateSpacePassivityReport


@dataclass(frozen=True)
class MultiportPassivityReport:
    """Passivity report for impedance matrices over a grid."""

    is_passive: bool
    min_eig: float
    worst_freq_hz: float
    n_violations: int
    tol: float
    details: dict[str, Any] = field(default_factory=dict)


def check_impedance_passivity(freq_hz: np.ndarray, Z: np.ndarray, tol: float = 1e-9) -> MultiportPassivityReport:
    freq_hz = np.asarray(freq_hz, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    if freq_hz.ndim != 1:
        raise ValueError("freq_hz must be 1D.")
    if Z.ndim == 1:
        if Z.shape != freq_hz.shape:
            raise ValueError("Z must match freq_hz shape for one-port.")
        real_part = Z.real
        min_real = float(np.min(real_part))
        idx = int(np.argmin(real_part))
        n_violations = int(np.sum(real_part < -tol))
        return MultiportPassivityReport(
            is_passive=n_violations == 0,
            min_eig=min_real,
            worst_freq_hz=float(freq_hz[idx]),
            n_violations=n_violations,
            tol=float(tol),
            details={"mode": "oneport"},
        )

    if Z.shape[0] != freq_hz.shape[0]:
        raise ValueError("Z leading dimension must match freq_hz.")
    if Z.ndim != 3 or Z.shape[1] != Z.shape[2]:
        raise ValueError("Z must have shape (n_freq, n_ports, n_ports) for multi-port.")

    min_eig = np.inf
    worst_freq = float(freq_hz[0])
    n_violations = 0
    for idx, freq in enumerate(freq_hz):
        hermitian = 0.5 * (Z[idx] + Z[idx].conj().T)
        eigs = np.linalg.eigvalsh(hermitian)
        local_min = float(np.min(eigs.real))
        if local_min < min_eig:
            min_eig = local_min
            worst_freq = float(freq)
        if local_min < -tol:
            n_violations += 1

    return MultiportPassivityReport(
        is_passive=n_violations == 0,
        min_eig=float(min_eig),
        worst_freq_hz=worst_freq,
        n_violations=n_violations,
        tol=float(tol),
        details={"mode": "multiport", "n_ports": int(Z.shape[1])},
    )


def check_state_space_passivity(
    model: StateSpaceModel,
    tol: float = 1e-9,
) -> StateSpacePassivityReport:
    A = np.asarray(model.A, dtype=float)
    B = np.asarray(model.B, dtype=float)
    C = np.asarray(model.C, dtype=float)
    D = np.asarray(model.D, dtype=float)
    E = None if model.E is None else np.asarray(model.E, dtype=float)

    stability_ok = _check_state_space_stability(A, E, tol=tol)

    if E is not None and _is_symmetric(A, tol=tol) and _is_symmetric(D, tol=tol):
        sym_a = 0.5 * (A + A.T)
        max_a = _max_eig(sym_a)
        if max_a <= tol and np.allclose(B, C.T, rtol=0.0, atol=tol):
            return StateSpacePassivityReport(
                is_passive=stability_ok,
                method="structure",
                stability_ok=stability_ok,
                max_eig=max_a,
                details={"tol": float(tol)},
            )

    if E is not None and _is_symmetric(E, tol=tol) and _min_eig(E) >= -tol:
        kyp = _kyp_matrix(A, B, C, D, E)
        max_eig = _max_eig(kyp)
        return StateSpacePassivityReport(
            is_passive=max_eig <= tol and stability_ok,
            method="kyp_energy",
            stability_ok=stability_ok,
            max_eig=max_eig,
            details={"kyp_dim": int(kyp.shape[0]), "tol": float(tol)},
        )

    if E is None:
        kyp = _kyp_matrix(A, B, C, D, np.eye(A.shape[0]))
        max_eig = _max_eig(kyp)
        return StateSpacePassivityReport(
            is_passive=max_eig <= tol and stability_ok,
            method="kyp_identity",
            stability_ok=stability_ok,
            max_eig=max_eig,
            details={"kyp_dim": int(kyp.shape[0]), "tol": float(tol)},
        )

    return StateSpacePassivityReport(
        is_passive=False,
        method="kyp_not_applicable",
        stability_ok=stability_ok,
        max_eig=float("nan"),
        details={"reason": "E not symmetric positive semidefinite."},
    )


def _kyp_matrix(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, P: np.ndarray) -> np.ndarray:
    top_left = A.T @ P + P @ A
    top_right = P @ B - C.T
    bottom_left = top_right.T
    bottom_right = -(D + D.T)
    return np.block([[top_left, top_right], [bottom_left, bottom_right]])


def _check_state_space_stability(A: np.ndarray, E: np.ndarray | None, tol: float) -> bool:
    if E is None:
        eigs = la.eigvals(A)
    else:
        eigs = la.eigvals(A, E)
    return bool(np.all(np.real(eigs) <= tol))


def _is_symmetric(matrix: np.ndarray, tol: float) -> bool:
    return bool(np.allclose(matrix, matrix.T, rtol=0.0, atol=tol))


def _min_eig(matrix: np.ndarray) -> float:
    eigs = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    return float(np.min(eigs.real))


def _max_eig(matrix: np.ndarray) -> float:
    eigs = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    return float(np.max(eigs.real))
