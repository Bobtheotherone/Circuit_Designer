"""Model extraction and fitting utilities."""

from fidp.analysis.fitting.vector_fitting import (
    RationalModel,
    VectorFitConfig,
    VectorFitResult,
    vector_fit,
    log_frequency_grid,
    refine_frequency_grid,
)
from fidp.analysis.fitting.passivity.tests import PassivityReport, check_passivity
from fidp.analysis.fitting.passivity.enforce_residue_nnls import enforce_passivity_nnls
from fidp.analysis.fitting.passivity.enforce_qp import enforce_passivity_qp
from fidp.analysis.fitting.fractional_fit import (
    FractionalOrderReport,
    FractionalFitConfig,
    estimate_fractional_order,
    fit_cpe,
    estimate_alpha_profile,
)
from fidp.analysis.fitting.symbolic_regression import (
    SymbolicRegressionConfig,
    SymbolicRegressionResult,
    symbolic_regression,
)

__all__ = [
    "RationalModel",
    "VectorFitConfig",
    "VectorFitResult",
    "vector_fit",
    "log_frequency_grid",
    "refine_frequency_grid",
    "PassivityReport",
    "check_passivity",
    "enforce_passivity_nnls",
    "enforce_passivity_qp",
    "FractionalOrderReport",
    "FractionalFitConfig",
    "estimate_fractional_order",
    "fit_cpe",
    "estimate_alpha_profile",
    "SymbolicRegressionConfig",
    "SymbolicRegressionResult",
    "symbolic_regression",
]
