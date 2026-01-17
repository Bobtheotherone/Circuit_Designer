"""Deterministic symbolic regression for simple scaling laws."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class SymbolicRegressionConfig:
    """Configuration for symbolic regression."""

    max_candidates: int = 24
    complexity_penalty: float = 0.02
    min_improvement: float = 0.1
    bootstrap_samples: int = 200
    seed: int | None = 11
    min_r2: float = 0.3

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive.")
        if self.complexity_penalty < 0.0:
            raise ValueError("complexity_penalty must be non-negative.")
        if self.min_improvement < 0.0:
            raise ValueError("min_improvement must be non-negative.")
        if self.bootstrap_samples < 20:
            raise ValueError("bootstrap_samples must be >= 20.")
        if self.min_r2 < -1.0:
            raise ValueError("min_r2 must be >= -1.")


@dataclass
class SymbolicRegressionResult:
    """Symbolic regression outcome."""

    expression: str | None
    parameters: dict[str, float]
    score: float
    confidence_intervals: dict[str, tuple[float, float]]
    rejected: bool
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.expression is None:
            raise ValueError("No expression available for evaluation.")
        return evaluate_expression(self.expression, self.parameters, x)


@dataclass(frozen=True)
class _Candidate:
    name: str
    expression: str
    design_fn: Callable[[np.ndarray], np.ndarray]
    param_names: tuple[str, ...]
    complexity: int


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(x, 1e-12))


def _build_candidates(x: np.ndarray) -> list[_Candidate]:
    n_features = x.shape[1]
    candidates: list[_Candidate] = []

    def add_candidate(name: str, expression: str, design_fn: Callable[[np.ndarray], np.ndarray], params: tuple[str, ...], complexity: int) -> None:
        candidates.append(_Candidate(name=name, expression=expression, design_fn=design_fn, param_names=params, complexity=complexity))

    add_candidate(
        name="constant",
        expression="c",
        design_fn=lambda data: np.ones((data.shape[0], 1), dtype=float),
        params=("c",),
        complexity=1,
    )

    add_candidate(
        name="linear",
        expression="a + b*x",
        design_fn=lambda data: np.column_stack([np.ones(data.shape[0]), data[:, 0]]),
        params=("a", "b"),
        complexity=2,
    )

    add_candidate(
        name="log",
        expression="a + b*log(x)",
        design_fn=lambda data: np.column_stack([np.ones(data.shape[0]), _safe_log(data[:, 0])]),
        params=("a", "b"),
        complexity=2,
    )

    add_candidate(
        name="power",
        expression="a * x^b",
        design_fn=lambda data: data[:, 0].reshape(-1, 1),
        params=("a", "b"),
        complexity=2,
    )

    add_candidate(
        name="log_quad",
        expression="a + b*log(x) + c*log(x)^2",
        design_fn=lambda data: np.column_stack([
            np.ones(data.shape[0]),
            _safe_log(data[:, 0]),
            _safe_log(data[:, 0]) ** 2,
        ]),
        params=("a", "b", "c"),
        complexity=3,
    )

    if n_features > 1:
        add_candidate(
            name="affine",
            expression="a + sum(b_i * x_i)",
            design_fn=lambda data: np.column_stack([np.ones(data.shape[0]), data]),
            params=("a",) + tuple(f"b{i}" for i in range(n_features)),
            complexity=1 + n_features,
        )
        add_candidate(
            name="log_affine",
            expression="a + sum(b_i * log(x_i))",
            design_fn=lambda data: np.column_stack([
                np.ones(data.shape[0]),
                _safe_log(data),
            ]),
            params=("a",) + tuple(f"b{i}" for i in range(n_features)),
            complexity=1 + n_features,
        )

    return candidates


def _fit_candidate(candidate: _Candidate, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    if candidate.name == "power":
        if np.any(x[:, 0] <= 0.0) or np.any(y <= 0.0):
            raise ValueError("Power-law candidate requires positive x and y.")
        logx = _safe_log(x[:, 0])
        logy = _safe_log(y)
        slope, intercept = np.polyfit(logx, logy, 1)
        params = np.array([np.exp(intercept), slope], dtype=float)
        y_hat = params[0] * (x[:, 0] ** params[1])
    else:
        design = candidate.design_fn(x)
        params, *_ = np.linalg.lstsq(design, y, rcond=None)
        y_hat = design @ params
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_hat) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return params, rmse, r2


def _bootstrap_params(candidate: _Candidate, x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_samples: int) -> np.ndarray:
    params_list: list[np.ndarray] = []
    for _ in range(n_samples):
        idx = rng.integers(0, x.shape[0], size=x.shape[0])
        if candidate.name == "power":
            sample_x = x[idx, 0]
            sample_y = y[idx]
            if np.any(sample_x <= 0.0) or np.any(sample_y <= 0.0):
                continue
            logx = _safe_log(sample_x)
            logy = _safe_log(sample_y)
            slope, intercept = np.polyfit(logx, logy, 1)
            params = np.array([np.exp(intercept), slope], dtype=float)
        else:
            params, *_ = np.linalg.lstsq(candidate.design_fn(x[idx]), y[idx], rcond=None)
        params_list.append(params)
    if not params_list:
        raise ValueError("Bootstrap failed; insufficient valid samples.")
    return np.stack(params_list, axis=0)


def evaluate_expression(expression: str, parameters: dict[str, float], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        data = x.reshape(-1, 1)
    else:
        data = x

    if expression == "c":
        return np.full(data.shape[0], parameters["c"], dtype=float)
    if expression == "a + b*x":
        return parameters["a"] + parameters["b"] * data[:, 0]
    if expression == "a + b*log(x)":
        return parameters["a"] + parameters["b"] * _safe_log(data[:, 0])
    if expression == "a * x^b":
        return parameters["a"] * (data[:, 0] ** parameters["b"])
    if expression == "a + b*log(x) + c*log(x)^2":
        logx = _safe_log(data[:, 0])
        return parameters["a"] + parameters["b"] * logx + parameters["c"] * logx**2
    if expression == "a + sum(b_i * x_i)":
        result = np.full(data.shape[0], parameters["a"], dtype=float)
        for idx in range(1, data.shape[1] + 1):
            result += parameters[f"b{idx-1}"] * data[:, idx - 1]
        return result
    if expression == "a + sum(b_i * log(x_i))":
        result = np.full(data.shape[0], parameters["a"], dtype=float)
        logx = _safe_log(data)
        for idx in range(1, data.shape[1] + 1):
            result += parameters[f"b{idx-1}"] * logx[:, idx - 1]
        return result

    raise ValueError(f"Unsupported expression: {expression}")


def symbolic_regression(
    x: np.ndarray,
    y: np.ndarray,
    config: SymbolicRegressionConfig | None = None,
) -> SymbolicRegressionResult:
    """Run a deterministic symbolic regression over a small candidate set."""
    config = config or SymbolicRegressionConfig()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples.")
    if x.shape[0] < 8:
        raise ValueError("At least 8 samples are required for symbolic regression.")

    candidates = _build_candidates(x)
    baseline = float(np.sqrt(np.mean((y - np.mean(y)) ** 2)))

    scored: list[tuple[float, _Candidate, np.ndarray, float]] = []
    for candidate in candidates:
        try:
            params, rmse, r2 = _fit_candidate(candidate, x, y)
        except ValueError:
            continue
        score = rmse + config.complexity_penalty * candidate.complexity
        scored.append((score, candidate, params, r2))

    if not scored:
        return SymbolicRegressionResult(
            expression=None,
            parameters={},
            score=float("inf"),
            confidence_intervals={},
            rejected=True,
            warnings=["No valid candidates for symbolic regression."],
            diagnostics={"baseline_rmse": baseline},
        )

    scored.sort(key=lambda item: (item[0], item[1].complexity, item[1].expression))
    scored = scored[: config.max_candidates]

    best_score, best_candidate, best_params, best_r2 = scored[0]
    improvement = (baseline - best_score) / max(baseline, 1e-12)

    warnings: list[str] = []
    rejected = False
    if improvement < config.min_improvement:
        warnings.append("Insufficient improvement over baseline; rejecting candidate.")
        rejected = True
    if best_r2 < config.min_r2:
        warnings.append("R2 below threshold; rejecting candidate.")
        rejected = True

    rng = np.random.default_rng(config.seed)
    try:
        boot_params = _bootstrap_params(best_candidate, x, y, rng, config.bootstrap_samples)
    except ValueError as exc:
        return SymbolicRegressionResult(
            expression=None,
            parameters={},
            score=float(best_score),
            confidence_intervals={},
            rejected=True,
            warnings=[str(exc)],
            diagnostics={
                "baseline_rmse": baseline,
                "best_candidate": best_candidate.name,
                "best_expression": best_candidate.expression,
                "r2": best_r2,
            },
        )

    ci: dict[str, tuple[float, float]] = {}
    for idx, name in enumerate(best_candidate.param_names):
        low, high = np.percentile(boot_params[:, idx], [2.5, 97.5])
        ci[name] = (float(low), float(high))

    parameters = {name: float(value) for name, value in zip(best_candidate.param_names, best_params)}

    if rejected:
        return SymbolicRegressionResult(
            expression=None,
            parameters={},
            score=float(best_score),
            confidence_intervals={},
            rejected=True,
            warnings=warnings,
            diagnostics={
                "baseline_rmse": baseline,
                "best_candidate": best_candidate.name,
                "best_expression": best_candidate.expression,
                "r2": best_r2,
            },
        )

    return SymbolicRegressionResult(
        expression=best_candidate.expression,
        parameters=parameters,
        score=float(best_score),
        confidence_intervals=ci,
        rejected=False,
        warnings=warnings,
        diagnostics={
            "baseline_rmse": baseline,
            "best_candidate": best_candidate.name,
            "r2": best_r2,
        },
    )
