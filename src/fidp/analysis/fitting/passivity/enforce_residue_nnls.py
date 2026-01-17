"""Passivity enforcement via constrained NNLS residue perturbation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.optimize

from fidp.analysis.fitting.vector_fitting import RationalModel
from fidp.analysis.fitting.passivity.tests import PassivityReport, check_passivity, group_conjugate_poles
from fidp.errors import PassivityViolationError


@dataclass(frozen=True)
class NNLSPassivityConfig:
    """Configuration for NNLS passivity enforcement."""

    tol: float = 1e-9
    max_rel_error_increase: float = 0.05
    include_d_offset: bool = True
    max_adjustment: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tol < 0.0:
            raise ValueError("tol must be non-negative.")
        if self.max_rel_error_increase < 0.0:
            raise ValueError("max_rel_error_increase must be non-negative.")
        if self.max_adjustment is not None and self.max_adjustment <= 0.0:
            raise ValueError("max_adjustment must be positive if provided.")


def _ensure_stable(poles: np.ndarray) -> None:
    if np.any(np.real(poles) >= 0.0):
        raise PassivityViolationError("Cannot enforce passivity with unstable poles present.")


def enforce_passivity_nnls(
    model: RationalModel,
    freq_hz: np.ndarray,
    config: NNLSPassivityConfig | None = None,
) -> tuple[RationalModel, PassivityReport]:
    """Enforce passivity by adjusting residue real parts using NNLS."""
    config = config or NNLSPassivityConfig()
    _ensure_stable(model.poles)

    freq_hz = np.asarray(freq_hz, dtype=float)
    original_response = model.eval_freq(freq_hz)
    report = check_passivity(freq_hz, original_response, tol=config.tol, refine=False, method="nnls")
    if report.is_passive:
        report.details["adjustment"] = "none"
        return model, report

    s = 1j * 2.0 * np.pi * freq_hz
    groups = group_conjugate_poles(model.poles)

    basis_cols: list[np.ndarray] = []
    group_map: list[tuple[tuple[int, ...], bool]] = []
    for group in groups:
        if group.is_complex_pair:
            basis = 1.0 / (s - group.pole)
            basis_cols.append(2.0 * basis.real)
            group_map.append((group.indices, True))
        else:
            pole = model.poles[group.indices[0]]
            basis = 1.0 / (s - pole)
            basis_cols.append(basis.real)
            group_map.append((group.indices, False))

    if config.include_d_offset:
        basis_cols.append(np.ones_like(freq_hz, dtype=float))

    A = np.column_stack(basis_cols)
    deficit = np.maximum(0.0, -config.tol - original_response.real)

    if np.all(deficit == 0.0):
        report.details["adjustment"] = "none"
        return model, report

    solution, _ = scipy.optimize.nnls(A, deficit)
    if config.max_adjustment is not None:
        solution = np.minimum(solution, config.max_adjustment)

    residues = model.residues.copy()
    d = model.d
    for idx, (indices, is_pair) in enumerate(group_map):
        delta = solution[idx]
        if is_pair:
            for pole_idx in indices:
                residues[pole_idx] = complex(residues[pole_idx].real + delta, residues[pole_idx].imag)
        else:
            pole_idx = indices[0]
            residues[pole_idx] = complex(residues[pole_idx].real + delta, residues[pole_idx].imag)

    if config.include_d_offset:
        d = complex(d.real + solution[-1], d.imag)

    updated = RationalModel(
        poles=model.poles.copy(),
        residues=residues,
        d=d,
        h=model.h,
        kind=model.kind,
        metadata=dict(model.metadata),
    )

    updated_response = updated.eval_freq(freq_hz)
    updated_report = check_passivity(freq_hz, updated_response, tol=config.tol, refine=False, method="nnls")

    if not updated_report.is_passive and config.include_d_offset:
        delta_offset = -(updated_report.margin) + config.tol
        if delta_offset > 0.0:
            d = complex(updated.d.real + delta_offset, updated.d.imag)
            updated = RationalModel(
                poles=updated.poles.copy(),
                residues=updated.residues.copy(),
                d=d,
                h=updated.h,
                kind=updated.kind,
                metadata=dict(updated.metadata),
            )
            updated_response = updated.eval_freq(freq_hz)
            updated_report = check_passivity(
                freq_hz, updated_response, tol=config.tol, refine=False, method="nnls"
            )
            updated_report.details["delta_offset"] = float(delta_offset)

    diff = np.abs(updated_response - original_response)
    denom = np.sqrt(np.mean(np.abs(original_response) ** 2))
    rel_rms = float(np.sqrt(np.mean(diff**2)) / max(denom, 1e-12))
    updated_report.details["rel_error_increase"] = rel_rms
    updated_report.details["adjustment"] = solution.tolist()

    if rel_rms > config.max_rel_error_increase:
        raise PassivityViolationError("NNLS enforcement exceeded accuracy degradation budget.")
    if not updated_report.is_passive:
        raise PassivityViolationError("NNLS enforcement failed to achieve passivity.")

    return updated, updated_report
