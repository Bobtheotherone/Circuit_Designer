"""PRIMA-style model order reduction for descriptor systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fidp.data import DescriptorSystem, ReducedDescriptorSystem, ImpedanceSweep
from fidp.errors import ReductionError


@dataclass
class _ArnoldiState:
    basis: List[np.ndarray]


def _orthonormalize(vec: np.ndarray, basis: List[np.ndarray], tol: float) -> np.ndarray | None:
    v = vec.astype(float, copy=True)
    for q in basis:
        v -= np.dot(q, v) * q
    norm = np.linalg.norm(v)
    if norm < tol:
        return None
    return v / norm


def _add_real_imag(vec: np.ndarray, state: _ArnoldiState, tol: float) -> None:
    for part in (vec.real, vec.imag):
        if np.linalg.norm(part) < tol:
            continue
        v = _orthonormalize(part, state.basis, tol)
        if v is not None:
            state.basis.append(v)


def _build_real_krylov_basis(
    system: DescriptorSystem,
    order_r: int,
    s0: complex,
    tol: float = 1e-12,
) -> np.ndarray:
    if order_r <= 0:
        raise ReductionError("Reduction order must be positive.")

    G = system.G
    C = system.C
    B = system.B

    if B.ndim != 2:
        raise ReductionError("B must be a 2D matrix.")

    A = G + s0 * C
    try:
        solve = spla.factorized(A.tocsc())
    except Exception as exc:
        raise ReductionError("Failed to factorize expansion matrix.") from exc

    state = _ArnoldiState(basis=[])
    block = [solve(B[:, i]) for i in range(B.shape[1])]

    iter_limit = order_r * 3
    iteration = 0
    while len(state.basis) < order_r and iteration < iter_limit:
        for vec in block:
            _add_real_imag(vec, state, tol)
            if len(state.basis) >= order_r:
                break
        if len(state.basis) >= order_r:
            break
        block = [-solve(C @ vec) for vec in block]
        iteration += 1

    if len(state.basis) < order_r:
        raise ReductionError("Unable to construct a Krylov basis of requested size.")

    V = np.column_stack(state.basis[:order_r])
    return V


def prima_reduce(
    system: DescriptorSystem,
    order_r: int,
    expansion_point_s0: complex,
) -> ReducedDescriptorSystem:
    """
    Reduce descriptor system using PRIMA with a real Krylov basis around s0.

    Complex expansion points are handled by using real and imaginary components
    of the Krylov vectors and re-orthonormalizing them.
    """
    V = _build_real_krylov_basis(system, order_r, expansion_point_s0)

    G = system.G
    C = system.C
    B = system.B
    L = system.L

    G_r = V.T @ (G @ V)
    C_r = V.T @ (C @ V)
    B_r = V.T @ B
    L_r = V.T @ L

    meta = {
        "order_r": order_r,
        "expansion_point_s0": expansion_point_s0,
        "method": "PRIMA",
    }
    return ReducedDescriptorSystem(G_r=G_r, C_r=C_r, B_r=B_r, L_r=L_r, meta=meta)


def evaluate_impedance_reduced(
    reduced: ReducedDescriptorSystem,
    freqs_hz: np.ndarray,
) -> ImpedanceSweep:
    """Evaluate impedance from a reduced descriptor system."""
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    Z = np.zeros_like(freqs_hz, dtype=complex)

    for idx, freq in enumerate(freqs_hz):
        s = 1j * 2.0 * np.pi * freq
        A = reduced.G_r + s * reduced.C_r
        try:
            x = np.linalg.solve(A, reduced.B_r[:, 0])
        except np.linalg.LinAlgError as exc:
            raise ReductionError("Reduced system solve failed.") from exc
        Z[idx] = reduced.L_r[:, 0].T @ x

    meta = {"reduced_meta": reduced.meta}
    return ImpedanceSweep(freqs_hz=freqs_hz, Z=Z, meta=meta)
