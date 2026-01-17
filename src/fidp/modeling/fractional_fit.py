"""Fractional-order impedance fitting utilities (compatibility wrapper)."""

from fidp.analysis.fitting.fractional_fit import (
    CPEFitResult,
    fit_cpe,
    estimate_alpha_profile,
    FractionalFitConfig,
    FractionalOrderReport,
    estimate_fractional_order,
)

__all__ = [
    "CPEFitResult",
    "fit_cpe",
    "estimate_alpha_profile",
    "FractionalFitConfig",
    "FractionalOrderReport",
    "estimate_fractional_order",
]
