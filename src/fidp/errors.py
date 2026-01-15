"""Custom exceptions for FIDP evaluators and circuit handling."""


class CircuitValidationError(ValueError):
    """Raised when a circuit or component is invalid."""


class SingularCircuitError(RuntimeError):
    """Raised when the circuit matrix is singular or disconnected."""


class SpiceNotAvailableError(RuntimeError):
    """Raised when a requested SPICE binary is unavailable."""


class ReductionError(RuntimeError):
    """Raised when model order reduction fails."""
