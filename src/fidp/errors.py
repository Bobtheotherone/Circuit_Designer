"""Custom exceptions for FIDP evaluators and circuit handling."""


class CircuitValidationError(ValueError):
    """Raised when a circuit or component is invalid."""


class CircuitIRValidationError(ValueError):
    """Raised when a CircuitIR structure is invalid."""


class DSLParseError(ValueError):
    """Raised when DSL parsing fails with location context."""

    def __init__(self, message: str, line: int | None = None, column: int | None = None, context: str | None = None):
        super().__init__(message)
        self.line = line
        self.column = column
        self.context = context


class DSLValidationError(ValueError):
    """Raised when DSL semantics are invalid."""


class SingularCircuitError(RuntimeError):
    """Raised when the circuit matrix is singular or disconnected."""


class SpiceNotAvailableError(RuntimeError):
    """Raised when a requested SPICE binary is unavailable."""


class ReductionError(RuntimeError):
    """Raised when model order reduction fails."""


class SpiceNetlistError(RuntimeError):
    """Raised when a SPICE netlist fails lint validation."""


class EvaluatorError(RuntimeError):
    """Base class for evaluator failures."""


class EvaluatorNotApplicableError(EvaluatorError):
    """Raised when an evaluator cannot handle a circuit/request."""


class EvaluatorConvergenceError(EvaluatorError):
    """Raised when a solver fails to converge."""


class EvaluatorNumericalError(EvaluatorError):
    """Raised when numerical issues prevent evaluation."""


class PassivityViolationError(EvaluatorError):
    """Raised when passivity checks fail."""


class SpiceSimulationError(EvaluatorError):
    """Raised when a SPICE run fails or produces invalid output."""
