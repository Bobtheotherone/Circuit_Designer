"""Passivity checks and one-port passivation (compatibility wrapper)."""

from __future__ import annotations

from fidp.analysis.fitting.passivity.tests import PassivityReport, check_passivity
from fidp.analysis.fitting.vector_fitting import RationalModel


def check_oneport_passivity(
    freq_hz,
    H,
    kind,
    tol: float = 1e-9,
) -> PassivityReport:
    """Check one-port passivity by verifying Re{H(jw)} >= -tol."""
    if kind not in ("impedance", "admittance"):
        raise ValueError("kind must be 'impedance' or 'admittance'.")
    return check_passivity(freq_hz, H, tol=tol, refine=False, method="oneport")


def passivate_oneport_min_offset(
    model: RationalModel,
    freq_hz,
    tol: float = 1e-9,
) -> tuple[RationalModel, PassivityReport]:
    """Enforce passivity by adding the minimum series/shunt offset on the grid."""
    report = check_passivity(freq_hz, model.eval_freq(freq_hz), tol=tol, refine=False, method="offset")
    delta = 0.0
    if report.margin < -tol:
        delta = -(report.margin) + tol

    updated = RationalModel(
        poles=model.poles.copy(),
        residues=model.residues.copy(),
        d=model.d + delta,
        h=model.h,
        kind=model.kind,
        metadata=dict(model.metadata),
    )

    updated_report = check_passivity(
        freq_hz,
        updated.eval_freq(freq_hz),
        tol=tol,
        refine=False,
        method="offset",
    )
    updated_report.details["delta_offset"] = float(delta)
    return updated, updated_report


__all__ = [
    "PassivityReport",
    "check_oneport_passivity",
    "passivate_oneport_min_offset",
]
