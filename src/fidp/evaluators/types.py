"""Typed data contracts for evaluator inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Optional

import numpy as np


def _complex_to_json(value: complex) -> dict[str, float]:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def _array_to_json(values: np.ndarray) -> object:
    arr = np.asarray(values)
    if arr.dtype.kind == "c":
        return _complex_array_to_json(arr)
    return arr.tolist()


def _complex_array_to_json(values: np.ndarray) -> object:
    arr = np.asarray(values, dtype=complex)
    if arr.ndim == 0:
        return _complex_to_json(complex(arr))
    if arr.ndim == 1:
        return [_complex_to_json(item) for item in arr]
    return [_complex_array_to_json(slice_) for slice_ in arr]


@dataclass(frozen=True)
class FrequencyGridSpec:
    """Frequency grid specification with deterministic spacing."""

    f_start_hz: float
    f_stop_hz: float
    points: int
    spacing: Literal["log", "linear"] = "log"
    include_dc: bool = False

    def __post_init__(self) -> None:
        if self.f_start_hz <= 0.0:
            raise ValueError("f_start_hz must be positive.")
        if self.f_stop_hz <= 0.0:
            raise ValueError("f_stop_hz must be positive.")
        if self.f_stop_hz < self.f_start_hz:
            raise ValueError("f_stop_hz must be >= f_start_hz.")
        if self.points <= 0:
            raise ValueError("points must be positive.")
        if self.spacing not in ("log", "linear"):
            raise ValueError("spacing must be 'log' or 'linear'.")

    def make_grid(self) -> np.ndarray:
        if self.spacing == "log":
            grid = np.logspace(
                np.log10(self.f_start_hz),
                np.log10(self.f_stop_hz),
                self.points,
                dtype=float,
            )
        else:
            grid = np.linspace(self.f_start_hz, self.f_stop_hz, self.points, dtype=float)
        if self.include_dc:
            grid = np.concatenate(([0.0], grid))
        return grid

    def to_json_dict(self) -> dict[str, object]:
        return {
            "f_start_hz": float(self.f_start_hz),
            "f_stop_hz": float(self.f_stop_hz),
            "points": int(self.points),
            "spacing": self.spacing,
            "include_dc": bool(self.include_dc),
        }


@dataclass(frozen=True)
class EvalRequest:
    """Evaluation request for a circuit."""

    grid: FrequencyGridSpec
    ports: Optional[list[str]] = None
    fidelity: Literal["fast", "mid", "truth", "mor"] = "mid"
    seed: int = 0
    value_mode: Literal["snapped", "continuous"] = "snapped"
    max_depth: Optional[int] = None
    timeout_s: Optional[float] = None
    spice_simulator: Literal["ngspice", "xyce"] = "ngspice"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.value_mode not in ("snapped", "continuous"):
            raise ValueError("value_mode must be 'snapped' or 'continuous'.")
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be non-negative.")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "grid": self.grid.to_json_dict(),
            "ports": list(self.ports) if self.ports is not None else None,
            "fidelity": self.fidelity,
            "seed": int(self.seed),
            "value_mode": self.value_mode,
            "max_depth": self.max_depth,
            "timeout_s": self.timeout_s,
            "spice_simulator": self.spice_simulator,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvalError:
    """Structured evaluator error."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


@dataclass(frozen=True)
class StateSpaceModel:
    """State-space or descriptor model."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    E: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        A = np.asarray(self.A)
        B = np.asarray(self.B)
        C = np.asarray(self.C)
        D = np.asarray(self.D)
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square.")
        if B.shape[0] != A.shape[0]:
            raise ValueError("B rows must match A.")
        if C.shape[1] != A.shape[1]:
            raise ValueError("C columns must match A.")
        if D.shape[0] != C.shape[0] or D.shape[1] != B.shape[1]:
            raise ValueError("D must match C rows and B cols.")
        if self.E is not None:
            E = np.asarray(self.E)
            if E.shape != A.shape:
                raise ValueError("E must match A dimensions.")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "A": _array_to_json(self.A),
            "B": _array_to_json(self.B),
            "C": _array_to_json(self.C),
            "D": _array_to_json(self.D),
            "E": None if self.E is None else _array_to_json(self.E),
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class StateSpacePassivityReport:
    """Passivity report for state-space models."""

    is_passive: bool
    method: str
    stability_ok: bool
    max_eig: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "is_passive": bool(self.is_passive),
            "method": self.method,
            "stability_ok": bool(self.stability_ok),
            "max_eig": float(self.max_eig),
            "details": dict(self.details),
        }


@dataclass
class EvalResult:
    """Structured evaluation result."""

    freqs_hz: np.ndarray
    Z: np.ndarray
    status: Literal["ok", "error"]
    errors: list[EvalError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    timing_s: dict[str, float] = field(default_factory=dict)
    passivity: Optional[dict[str, Any]] = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "freqs_hz": _array_to_json(self.freqs_hz),
            "Z": _array_to_json(self.Z),
            "status": self.status,
            "errors": [err.to_json_dict() for err in self.errors],
            "warnings": list(self.warnings),
            "meta": dict(self.meta),
            "timing_s": {key: float(val) for key, val in self.timing_s.items()},
            "passivity": None if self.passivity is None else dict(self.passivity),
        }


def summarize_errors(errors: Iterable[EvalError]) -> str:
    """Create a compact summary string for error lists."""
    return "; ".join(f"{err.code}: {err.message}" for err in errors)
