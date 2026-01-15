"""Model extraction utilities."""

from fidp.modeling.fractional_fit import CPEFitResult, fit_cpe, estimate_alpha_profile
from fidp.modeling.vector_fit import RationalModel, VectorFitConfig, VectorFitResult, vector_fit
from fidp.modeling.passivity import (
    PassivityReport,
    check_oneport_passivity,
    passivate_oneport_min_offset,
)
from fidp.modeling.macromodel_io import (
    load_rational_model_json,
    save_rational_model_json,
    format_pole_residue_text,
)

__all__ = [
    "CPEFitResult",
    "fit_cpe",
    "estimate_alpha_profile",
    "RationalModel",
    "VectorFitConfig",
    "VectorFitResult",
    "vector_fit",
    "PassivityReport",
    "check_oneport_passivity",
    "passivate_oneport_min_offset",
    "load_rational_model_json",
    "save_rational_model_json",
    "format_pole_residue_text",
]
