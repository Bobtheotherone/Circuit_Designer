"""Model order reduction utilities."""

from fidp.evaluators.mor.prima import (
    prima_reduce,
    prima_reduce_adaptive,
    evaluate_impedance_reduced,
    descriptor_to_state_space,
    reduced_to_state_space,
    PrimaConfig,
    PrimaDiagnostics,
)
from fidp.evaluators.mor.evaluator import PrimaEvaluator, MorOptions

__all__ = [
    "prima_reduce",
    "prima_reduce_adaptive",
    "evaluate_impedance_reduced",
    "descriptor_to_state_space",
    "reduced_to_state_space",
    "PrimaConfig",
    "PrimaDiagnostics",
    "PrimaEvaluator",
    "MorOptions",
]
