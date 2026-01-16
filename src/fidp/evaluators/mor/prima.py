"""PRIMA-style model order reduction for descriptor systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import scipy.sparse.linalg as spla

from fidp.data import DescriptorSystem, ReducedDescriptorSystem, ImpedanceSweep
from fidp.errors import ReductionError
from fidp.evaluators.mna.descriptor import evaluate_impedance_descriptor
from fidp.evaluators.types import StateSpaceModel


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
    expansion_point_s0: complex | Sequence[complex],
) -> ReducedDescriptorSystem:
    """
    Reduce descriptor system using PRIMA with a real Krylov basis around s0.

    Complex expansion points are handled by using real and imaginary components
    of the Krylov vectors and re-orthonormalizing them.
    """
    expansion_points = _normalize_expansion_points(expansion_point_s0)
    V = _build_multi_point_basis(system, order_r, expansion_points)

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
        "expansion_points_s0": expansion_points,
        "method": "PRIMA",
    }
    return ReducedDescriptorSystem(G_r=G_r, C_r=C_r, B_r=B_r, L_r=L_r, meta=meta)


def evaluate_impedance_reduced(
    reduced: ReducedDescriptorSystem,
    freqs_hz: np.ndarray,
) -> ImpedanceSweep:
    """Evaluate impedance from a reduced descriptor system."""
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    n_ports = reduced.B_r.shape[1] if reduced.B_r.ndim == 2 else 1
    if n_ports == 1:
        Z = np.zeros_like(freqs_hz, dtype=complex)
    else:
        Z = np.zeros((freqs_hz.size, n_ports, n_ports), dtype=complex)

    for idx, freq in enumerate(freqs_hz):
        s = 1j * 2.0 * np.pi * freq
        A = reduced.G_r + s * reduced.C_r
        try:
            x = np.linalg.solve(A, reduced.B_r)
        except np.linalg.LinAlgError as exc:
            raise ReductionError("Reduced system solve failed.") from exc
        Z_slice = reduced.L_r.T @ x
        if n_ports == 1:
            Z[idx] = Z_slice[0, 0]
        else:
            Z[idx] = Z_slice

    meta = {"reduced_meta": reduced.meta}
    return ImpedanceSweep(freqs_hz=freqs_hz, Z=Z, meta=meta)


@dataclass(frozen=True)
class PrimaConfig:
    """Configuration for adaptive PRIMA reduction."""

    min_order: int = 4
    max_order: int = 20
    order_step: int = 2
    expansion_points_s0: Sequence[complex] = field(default_factory=lambda: (1j * 2.0 * np.pi * 1e3,))
    target_rel_error: float = 0.05
    tol: float = 1e-12


@dataclass
class PrimaDiagnostics:
    """Diagnostics from adaptive PRIMA reduction."""

    orders: list[int]
    max_rel_errors: list[float]
    selected_order: int
    converged: bool


def prima_reduce_adaptive(
    system: DescriptorSystem,
    config: PrimaConfig,
    validation_freqs: np.ndarray,
) -> tuple[ReducedDescriptorSystem, PrimaDiagnostics]:
    if config.min_order <= 0:
        raise ReductionError("min_order must be positive.")
    if config.max_order < config.min_order:
        raise ReductionError("max_order must be >= min_order.")
    if config.order_step <= 0:
        raise ReductionError("order_step must be positive.")

    full = evaluate_impedance_descriptor(system, validation_freqs)
    orders: list[int] = []
    max_errors: list[float] = []
    selected = config.max_order
    converged = False

    for order in range(config.min_order, config.max_order + 1, config.order_step):
        reduced = prima_reduce(system, order, config.expansion_points_s0)
        reduced_sweep = evaluate_impedance_reduced(reduced, validation_freqs)
        rel_err = _relative_error(full.Z, reduced_sweep.Z)
        max_err = float(np.max(rel_err))
        orders.append(order)
        max_errors.append(max_err)
        if max_err <= config.target_rel_error:
            selected = order
            converged = True
            return reduced, PrimaDiagnostics(orders, max_errors, selected, converged)

    reduced = prima_reduce(system, config.max_order, config.expansion_points_s0)
    return reduced, PrimaDiagnostics(orders, max_errors, selected, converged)


def descriptor_to_state_space(system: DescriptorSystem) -> StateSpaceModel:
    """Convert descriptor system to state-space representation."""
    A = -system.G.toarray()
    E = system.C.toarray()
    B = system.B.astype(float)
    C = system.L.T.astype(float)
    D = np.zeros((C.shape[0], B.shape[1]), dtype=float)
    return StateSpaceModel(A=A, B=B, C=C, D=D, E=E, meta=dict(system.meta))


def reduced_to_state_space(reduced: ReducedDescriptorSystem) -> StateSpaceModel:
    """Convert reduced descriptor system to state-space representation."""
    A = -np.asarray(reduced.G_r, dtype=float)
    E = np.asarray(reduced.C_r, dtype=float)
    B = np.asarray(reduced.B_r, dtype=float)
    C = np.asarray(reduced.L_r, dtype=float).T
    D = np.zeros((C.shape[0], B.shape[1]), dtype=float)
    return StateSpaceModel(A=A, B=B, C=C, D=D, E=E, meta=dict(reduced.meta))


def _normalize_expansion_points(expansion_points: complex | Sequence[complex]) -> list[complex]:
    if isinstance(expansion_points, complex):
        return [expansion_points]
    return [complex(point) for point in expansion_points]


def _build_multi_point_basis(
    system: DescriptorSystem,
    order_r: int,
    expansion_points: Sequence[complex],
) -> np.ndarray:
    if not expansion_points:
        raise ReductionError("At least one expansion point is required.")
    base = max(1, order_r // len(expansion_points))
    remainder = order_r - base * len(expansion_points)
    basis_list: list[np.ndarray] = []
    for idx, point in enumerate(expansion_points):
        order = base + (1 if idx < remainder else 0)
        basis_list.append(_build_real_krylov_basis(system, order, point))
    V = np.column_stack(basis_list)
    Q, _ = np.linalg.qr(V)
    if Q.shape[1] < order_r:
        raise ReductionError("Unable to construct a basis of requested size.")
    return Q[:, :order_r]


def _relative_error(full: np.ndarray, reduced: np.ndarray) -> np.ndarray:
    full = np.asarray(full)
    reduced = np.asarray(reduced)
    denom = np.maximum(np.abs(full), 1e-12)
    return np.abs(full - reduced) / denom
