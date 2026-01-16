"""Recurrence evaluator for self-similar CircuitIR families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import time

import numpy as np

from fidp.circuits import CircuitIR
from fidp.errors import CircuitIRValidationError
from fidp.evaluators.recurrence.solver import FixedPointImpedanceSolver, RecurrenceFn
from fidp.evaluators.types import EvalError, EvalRequest, EvalResult


@dataclass(frozen=True)
class RecurrenceOptions:
    """Options for recurrence evaluation."""

    max_iter: int = 200
    tol: float = 1e-8
    residual_tol: float = 1e-8
    damping: float = 0.6
    anderson_m: int = 4
    anderson_start: int = 5
    stall_iter: int = 4
    truncation_tol: float = 1e-6


@dataclass(frozen=True)
class _RecurrenceSpec:
    kind: str
    r_value: float
    c_value: float
    stages: Optional[int] = None
    scale: Optional[float] = None
    depth: Optional[int] = None


class RecurrenceEvaluator:
    """Evaluator for self-similar recurrence families."""

    def __init__(self, options: Optional[RecurrenceOptions] = None) -> None:
        self.options = options or RecurrenceOptions()

    def evaluate(self, circuit: CircuitIR, request: EvalRequest) -> EvalResult:
        freqs = request.grid.make_grid()
        start = time.perf_counter()
        try:
            circuit.validate()
        except CircuitIRValidationError as exc:
            return _error_result(freqs, "circuit_invalid", str(exc), {})
        try:
            spec = _infer_recurrence_spec(circuit, request.ports, request.value_mode)
        except CircuitIRValidationError as exc:
            return _error_result(freqs, "recurrence_not_applicable", str(exc), {})

        if spec.kind == "fractal":
            result = _evaluate_fractal(freqs, spec)
        else:
            solver = FixedPointImpedanceSolver(
                max_iter=self.options.max_iter,
                tol=self.options.tol,
                residual_tol=self.options.residual_tol,
                damping=self.options.damping,
                anderson_m=self.options.anderson_m,
                anderson_start=self.options.anderson_start,
                stall_iter=self.options.stall_iter,
            )
            recurrence = _build_recurrence(spec)
            result = solver.solve(freqs, recurrence, initial_z=spec.r_value, warm_start=True)

        truncation = None
        if spec.stages is not None:
            truncation = _truncate_ladder(freqs, spec, spec.stages)
        elif spec.depth is not None:
            truncation = _truncate_fractal(freqs, spec)

        intervals = _contractive_intervals(freqs, result.contraction)
        meta = {
            "family": spec.kind,
            "r_value": spec.r_value,
            "c_value": spec.c_value,
            "stages": spec.stages,
            "converged_fraction": float(np.mean(result.converged)),
            "contraction": result.contraction.tolist(),
            "method": result.method.tolist(),
            "intervals": intervals,
        }
        if spec.kind == "cross":
            meta["approximation"] = "differential_mode_series_resistance=2R"
        if truncation is not None:
            meta["truncation_error"] = truncation["error"].tolist()
            meta["truncation_depth"] = truncation["depth"]

        if not result.converged.all():
            return EvalResult(
                freqs_hz=freqs,
                Z=result.Z,
                status="error",
                errors=[
                    EvalError(
                        code="recurrence_nonconverged",
                        message="Recurrence solver failed to converge for all frequencies.",
                        details={"converged_fraction": float(np.mean(result.converged))},
                    )
                ],
                meta=meta,
                timing_s={"total": time.perf_counter() - start},
            )

        return EvalResult(
            freqs_hz=freqs,
            Z=result.Z,
            status="ok",
            meta=meta,
            timing_s={"total": time.perf_counter() - start},
        )


def _infer_recurrence_spec(
    circuit: CircuitIR, ports: Optional[list[str]], value_mode: str
) -> _RecurrenceSpec:
    generator = circuit.metadata.get("generator")
    if ports is not None and len(ports) != 1:
        raise CircuitIRValidationError("Recurrence evaluator supports one-port requests only.")
    if ports is None and len(circuit.ports) != 1 and generator != "cross_ladder":
        raise CircuitIRValidationError("Recurrence evaluator requires a single-port circuit.")
    if generator == "domino_ladder":
        return _spec_from_components(circuit, "domino", r_scale=1.0, value_mode=value_mode)
    if generator == "cross_ladder":
        return _spec_from_components(circuit, "cross", r_scale=2.0, value_mode=value_mode)
    if generator == "fractal_ladder":
        scale = circuit.metadata.get("scale")
        if scale is None:
            raise CircuitIRValidationError("Fractal ladder metadata must include scale.")
        spec = _spec_from_components(circuit, "fractal", r_scale=1.0, value_mode=value_mode)
        return _RecurrenceSpec(
            kind=spec.kind,
            r_value=spec.r_value,
            c_value=spec.c_value,
            stages=None,
            scale=float(scale),
            depth=int(circuit.metadata.get("recursion_depth", 0)),
        )
    raise CircuitIRValidationError("Circuit is not tagged as a recurrence-capable family.")


def _spec_from_components(
    circuit: CircuitIR, kind: str, r_scale: float, value_mode: str
) -> _RecurrenceSpec:
    mode = "snapped" if value_mode == "snapped" else "continuous"
    r_values = [comp.value.resolved(mode) for comp in circuit.components if comp.kind == "R"]
    c_values = [comp.value.resolved(mode) for comp in circuit.components if comp.kind == "C"]
    if not r_values or not c_values:
        raise CircuitIRValidationError("Recurrence requires R and C components.")
    if not _all_close(r_values) or not _all_close(c_values):
        raise CircuitIRValidationError("Recurrence requires uniform R and C values.")
    stages = None
    if kind == "domino":
        stages = len(r_values)
    elif kind == "cross":
        if len(r_values) % 2 != 0:
            raise CircuitIRValidationError("Cross ladder should have an even number of resistors.")
        stages = len(r_values) // 2
    return _RecurrenceSpec(
        kind=kind,
        r_value=r_values[0] * r_scale,
        c_value=c_values[0],
        stages=stages,
    )


def _build_recurrence(spec: _RecurrenceSpec) -> RecurrenceFn:
    r = spec.r_value
    c = spec.c_value

    def recurrence(z: complex, s: complex) -> complex:
        return r + 1.0 / (s * c + 1.0 / z)

    return recurrence


def _truncate_ladder(freqs: np.ndarray, spec: _RecurrenceSpec, depth: int) -> dict[str, np.ndarray | int]:
    s = 1j * 2.0 * np.pi * freqs
    z_prev = 1.0 / (s * spec.c_value)
    z_curr = z_prev
    z_prev2 = None
    for _ in range(depth):
        z_curr = spec.r_value + 1.0 / (s * spec.c_value + 1.0 / z_prev)
        z_prev2 = z_prev
        z_prev = z_curr
    error = np.zeros_like(freqs, dtype=float)
    if z_prev2 is not None:
        error = np.abs(z_curr - z_prev2)
    return {"depth": depth, "error": error}


def _truncate_fractal(freqs: np.ndarray, spec: _RecurrenceSpec) -> dict[str, np.ndarray | int]:
    if spec.scale is None or spec.depth is None:
        raise CircuitIRValidationError("Fractal recurrence requires scale and depth.")
    z_curr = _fractal_impedance(freqs, spec.r_value, spec.c_value, spec.scale, spec.depth)
    if spec.depth > 0:
        z_prev = _fractal_impedance(freqs, spec.r_value, spec.c_value, spec.scale, spec.depth - 1)
        error = np.abs(z_curr - z_prev)
    else:
        error = np.zeros_like(freqs, dtype=float)
    return {"depth": spec.depth, "error": error}


def _evaluate_fractal(freqs: np.ndarray, spec: _RecurrenceSpec):
    if spec.scale is None or spec.depth is None:
        raise CircuitIRValidationError("Fractal recurrence requires scale and depth.")
    z_curr = _fractal_impedance(freqs, spec.r_value, spec.c_value, spec.scale, spec.depth)
    converged = np.ones_like(freqs, dtype=bool)
    iterations = np.full_like(freqs, spec.depth + 1, dtype=int)
    residual = np.zeros_like(freqs, dtype=float)
    contraction = np.full_like(freqs, np.nan, dtype=float)
    method = np.array(["depth_recursion"] * freqs.size, dtype=object)
    from fidp.evaluators.recurrence.solver import RecurrenceResult

    return RecurrenceResult(
        freqs_hz=freqs,
        Z=z_curr,
        converged=converged,
        iterations=iterations,
        residual=residual,
        contraction=contraction,
        method=method,
        meta={"depth": spec.depth, "scale": spec.scale},
    )


def _fractal_impedance(
    freqs: np.ndarray, r_value: float, c_value: float, scale: float, depth: int
) -> np.ndarray:
    s = 1j * 2.0 * np.pi * freqs
    if depth == 0:
        return r_value + 1.0 / (s * c_value)
    child = _fractal_impedance(freqs, r_value * scale, c_value * scale, scale, depth - 1)
    return r_value + 1.0 / (s * c_value + 1.0 / child)


def _contractive_intervals(freqs: np.ndarray, contraction: np.ndarray) -> list[tuple[float, float]]:
    mask = np.asarray(contraction) < 1.0
    intervals: list[tuple[float, float]] = []
    if mask.size == 0:
        return intervals
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = float(freqs[idx])
        if start is not None and (not flag or idx == mask.size - 1):
            end = float(freqs[idx] if flag else freqs[idx - 1])
            intervals.append((start, end))
            start = None
    return intervals


def _all_close(values: list[float], tol: float = 1e-9) -> bool:
    base = values[0]
    return all(abs(val - base) <= tol * max(1.0, abs(base)) for val in values)


def _error_result(freqs: np.ndarray, code: str, message: str, details: dict[str, Any]) -> EvalResult:
    Z = np.full_like(freqs, np.nan, dtype=complex)
    return EvalResult(
        freqs_hz=freqs,
        Z=Z,
        status="error",
        errors=[EvalError(code=code, message=message, details=details)],
    )
