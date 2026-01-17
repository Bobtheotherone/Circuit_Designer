"""Passivity enforcement via quadratic programming residue perturbation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.optimize

from fidp.analysis.fitting.vector_fitting import RationalModel
from fidp.analysis.fitting.passivity.tests import PassivityReport, check_passivity, group_conjugate_poles
from fidp.errors import PassivityViolationError, OptimizationFailureError


@dataclass(frozen=True)
class QPPassivityConfig:
    """Configuration for QP passivity enforcement."""

    tol: float = 1e-9
    max_rel_error_increase: float = 0.05
    include_d_offset: bool = True
    max_iter: int = 500
    max_adjustment: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tol < 0.0:
            raise ValueError("tol must be non-negative.")
        if self.max_rel_error_increase < 0.0:
            raise ValueError("max_rel_error_increase must be non-negative.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.max_adjustment is not None and self.max_adjustment <= 0.0:
            raise ValueError("max_adjustment must be positive if provided.")


def _ensure_stable(poles: np.ndarray) -> None:
    if np.any(np.real(poles) >= 0.0):
        raise PassivityViolationError("Cannot enforce passivity with unstable poles present.")


def enforce_passivity_qp(
    model: RationalModel,
    freq_hz: np.ndarray,
    config: QPPassivityConfig | None = None,
) -> tuple[RationalModel, PassivityReport]:
    """Enforce passivity by solving a quadratic program for residue perturbations."""
    config = config or QPPassivityConfig()
    _ensure_stable(model.poles)

    freq_hz = np.asarray(freq_hz, dtype=float)
    original_response = model.eval_freq(freq_hz)
    report = check_passivity(freq_hz, original_response, tol=config.tol, refine=False, method="qp")
    if report.is_passive:
        report.details["adjustment"] = "none"
        return model, report

    s = 1j * 2.0 * np.pi * freq_hz
    groups = group_conjugate_poles(model.poles)

    A_cols: list[np.ndarray] = []
    var_map: list[tuple[tuple[int, ...], bool, bool]] = []
    weights: list[float] = []

    for group in groups:
        if group.is_complex_pair:
            basis = 1.0 / (s - group.pole)
            A_cols.append(2.0 * basis.real)
            A_cols.append(-2.0 * basis.imag)
            var_map.append((group.indices, True, False))
            var_map.append((group.indices, True, True))
            residue = model.residues[group.indices[0]]
            scale = max(abs(residue), 1e-3)
            weights.extend([1.0 / scale, 1.0 / scale])
        else:
            pole = model.poles[group.indices[0]]
            basis = 1.0 / (s - pole)
            A_cols.append(basis.real)
            var_map.append((group.indices, False, False))
            residue = model.residues[group.indices[0]]
            scale = max(abs(residue), 1e-3)
            weights.append(1.0 / scale)

    if config.include_d_offset:
        A_cols.append(np.ones_like(freq_hz, dtype=float))
        var_map.append(((-1,), False, False))
        weights.append(1.0)

    A = np.column_stack(A_cols)
    deficit = np.maximum(0.0, -config.tol - original_response.real)

    if np.all(deficit == 0.0):
        report.details["adjustment"] = "none"
        return model, report

    weights_arr = np.asarray(weights, dtype=float)

    def objective(x: np.ndarray) -> float:
        scaled = weights_arr * x
        return 0.5 * float(np.dot(scaled, scaled))

    def grad(x: np.ndarray) -> np.ndarray:
        return weights_arr**2 * x

    if config.max_adjustment is not None:
        bounds = [(-config.max_adjustment, config.max_adjustment)] * A.shape[1]
    else:
        bounds = None

    constraint = scipy.optimize.LinearConstraint(A, deficit, np.full_like(deficit, np.inf))

    result = scipy.optimize.minimize(
        objective,
        np.zeros(A.shape[1], dtype=float),
        method="trust-constr",
        jac=grad,
        hess=lambda _x: np.diag(weights_arr**2),
        constraints=[constraint],
        bounds=bounds,
        options={"maxiter": config.max_iter, "gtol": 1e-8, "xtol": 1e-10},
    )

    if not result.success:
        raise OptimizationFailureError(f"QP enforcement failed: {result.message}")

    x = result.x

    residues = model.residues.copy()
    d = model.d

    for idx, (indices, is_pair, is_imag) in enumerate(var_map):
        delta = x[idx]
        if indices == (-1,):
            d = complex(d.real + delta, d.imag)
            continue
        if is_pair:
            r = residues[indices[0]]
            delta_complex = complex(delta, 0.0) if not is_imag else complex(0.0, delta)
            updated = r + delta_complex
            residues[indices[0]] = updated
            residues[indices[1]] = np.conj(updated)
        else:
            pole_idx = indices[0]
            r = residues[pole_idx]
            delta_complex = complex(delta, 0.0)
            residues[pole_idx] = r + delta_complex

    updated = RationalModel(
        poles=model.poles.copy(),
        residues=residues,
        d=d,
        h=model.h,
        kind=model.kind,
        metadata=dict(model.metadata),
    )

    updated_response = updated.eval_freq(freq_hz)
    updated_report = check_passivity(freq_hz, updated_response, tol=config.tol, refine=False, method="qp")

    diff = np.abs(updated_response - original_response)
    denom = np.sqrt(np.mean(np.abs(original_response) ** 2))
    rel_rms = float(np.sqrt(np.mean(diff**2)) / max(denom, 1e-12))
    updated_report.details["rel_error_increase"] = rel_rms
    updated_report.details["adjustment"] = x.tolist()

    if rel_rms > config.max_rel_error_increase:
        raise PassivityViolationError("QP enforcement exceeded accuracy degradation budget.")
    if not updated_report.is_passive:
        raise PassivityViolationError("QP enforcement failed to achieve passivity.")

    return updated, updated_report
