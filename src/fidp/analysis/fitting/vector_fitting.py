"""Vector fitting for scalar rational macromodels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal
import hashlib
from collections import OrderedDict

import numpy as np

from fidp.errors import (
    VectorFitConvergenceError,
    InvalidFrequencyGridError,
    IllConditionedSolveError,
)


_COMPLEX_JSON = dict[str, float]


def _complex_to_json(value: complex) -> _COMPLEX_JSON:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def _json_to_complex(data: dict[str, Any]) -> complex:
    return complex(float(data["real"]), float(data["imag"]))


@dataclass
class RationalModel:
    """Pole-residue rational model for one-port data."""

    poles: np.ndarray
    residues: np.ndarray
    d: complex
    h: complex
    kind: Literal["impedance", "admittance"]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.poles = np.asarray(self.poles, dtype=complex)
        self.residues = np.asarray(self.residues, dtype=complex)
        if self.poles.shape != self.residues.shape:
            raise ValueError("poles and residues must have the same shape.")
        if self.poles.ndim != 1:
            raise ValueError("poles and residues must be 1D arrays.")
        if self.kind not in ("impedance", "admittance"):
            raise ValueError("kind must be 'impedance' or 'admittance'.")

    def eval_s(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=complex)
        basis = 1.0 / (s[:, None] - self.poles[None, :])
        return self.d + self.h * s + np.sum(self.residues[None, :] * basis, axis=1)

    def eval_freq(self, freq_hz: np.ndarray) -> np.ndarray:
        freq_hz = np.asarray(freq_hz, dtype=float)
        s = 1j * 2.0 * np.pi * freq_hz
        return self.eval_s(s)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "poles": [_complex_to_json(pole) for pole in self.poles],
            "residues": [_complex_to_json(residue) for residue in self.residues],
            "d": _complex_to_json(self.d),
            "h": _complex_to_json(self.h),
            "kind": self.kind,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> "RationalModel":
        poles = np.array([_json_to_complex(item) for item in data["poles"]], dtype=complex)
        residues = np.array([_json_to_complex(item) for item in data["residues"]], dtype=complex)
        d = _json_to_complex(data["d"])
        h = _json_to_complex(data["h"])
        kind = data["kind"]
        metadata = dict(data.get("metadata", {}))
        return RationalModel(poles=poles, residues=residues, d=d, h=h, kind=kind, metadata=metadata)


@dataclass
class VectorFitConfig:
    """Configuration for vector fitting."""

    n_poles: int
    n_iters: int = 12
    pole_shift_tol: float = 1e-6
    min_iters: int = 2
    early_stop: bool = True
    init_pole_scale: float = 1.0
    weighting: Literal["uniform", "inv_mag", "mag_phase", "custom"] = "uniform"
    stabilize_poles: bool = True
    enforce_conjugate_symmetry: bool = True
    ridge_lambda: float = 0.0
    max_condition: float = 1e12
    require_convergence: bool = False
    real_poles: int | None = None
    phase_weight: float = 0.2
    max_weight: float = 1e6
    fail_on_ill_conditioned: bool = False

    def __post_init__(self) -> None:
        if self.n_poles <= 0:
            raise ValueError("n_poles must be positive.")
        if self.n_iters <= 0:
            raise ValueError("n_iters must be positive.")
        if self.min_iters <= 0:
            raise ValueError("min_iters must be positive.")
        if self.pole_shift_tol <= 0.0:
            raise ValueError("pole_shift_tol must be positive.")
        if self.init_pole_scale <= 0.0:
            raise ValueError("init_pole_scale must be positive.")
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be non-negative.")
        if self.max_condition <= 0.0:
            raise ValueError("max_condition must be positive.")
        if not isinstance(self.fail_on_ill_conditioned, bool):
            raise ValueError("fail_on_ill_conditioned must be boolean.")
        if self.real_poles is not None and (self.real_poles < 0 or self.real_poles > self.n_poles):
            raise ValueError("real_poles must be between 0 and n_poles.")
        if self.phase_weight < 0.0:
            raise ValueError("phase_weight must be non-negative.")
        if self.max_weight <= 0.0:
            raise ValueError("max_weight must be positive.")


@dataclass
class VectorFitResult:
    """Vector fitting result."""

    model: RationalModel
    rel_error_rms: float
    abs_error_rms: float
    max_rel_error: float
    max_abs_error: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


_CACHE_LIMIT = 6
_BASIS_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()


def _basis_cache_key(poles: np.ndarray, s: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.asarray(poles, dtype=complex).tobytes())
    hasher.update(np.asarray(s, dtype=complex).tobytes())
    return hasher.hexdigest()


def _compute_basis(poles: np.ndarray, s: np.ndarray) -> np.ndarray:
    key = _basis_cache_key(poles, s)
    cached = _BASIS_CACHE.get(key)
    if cached is not None:
        _BASIS_CACHE.move_to_end(key)
        return cached
    basis = 1.0 / (s[:, None] - poles[None, :])
    _BASIS_CACHE[key] = basis
    if len(_BASIS_CACHE) > _CACHE_LIMIT:
        _BASIS_CACHE.popitem(last=False)
    return basis


def log_frequency_grid(fmin_hz: float, fmax_hz: float, n_points: int) -> np.ndarray:
    """Generate a log-spaced frequency grid."""
    if fmin_hz <= 0.0 or fmax_hz <= 0.0 or fmin_hz >= fmax_hz:
        raise InvalidFrequencyGridError("Frequency bounds must satisfy 0 < fmin < fmax.")
    if n_points < 3:
        raise InvalidFrequencyGridError("n_points must be >= 3 for log spacing.")
    return np.logspace(np.log10(fmin_hz), np.log10(fmax_hz), n_points)


def refine_frequency_grid(
    freq_hz: np.ndarray,
    residual: np.ndarray,
    max_new_points: int = 40,
    hotspot_frac: float = 0.85,
) -> np.ndarray:
    """Suggest refined grid points around residual hotspots."""
    freq_hz = np.asarray(freq_hz, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if freq_hz.ndim != 1 or residual.shape != freq_hz.shape:
        raise InvalidFrequencyGridError("freq_hz and residual must be 1D arrays of equal length.")
    if np.any(np.diff(freq_hz) <= 0.0):
        raise InvalidFrequencyGridError("freq_hz must be strictly increasing for refinement.")
    if max_new_points <= 0:
        return np.array([], dtype=float)

    peak = float(np.max(residual))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.array([], dtype=float)

    threshold = hotspot_frac * peak
    hotspots = np.where(residual >= threshold)[0]
    candidates: list[float] = []
    for idx in hotspots:
        if idx > 0:
            candidates.append(0.5 * (freq_hz[idx - 1] + freq_hz[idx]))
        if idx < freq_hz.size - 1:
            candidates.append(0.5 * (freq_hz[idx] + freq_hz[idx + 1]))
    if not candidates:
        return np.array([], dtype=float)

    candidates = sorted(set(candidates))
    if len(candidates) > max_new_points:
        candidates = candidates[:max_new_points]
    return np.array(candidates, dtype=float)


def _validate_vector_fit_inputs(
    freq_hz: np.ndarray,
    H: np.ndarray,
    cfg: VectorFitConfig,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    freq_hz = np.asarray(freq_hz, dtype=float)
    if freq_hz.ndim != 1:
        raise InvalidFrequencyGridError("freq_hz must be a 1D array.")
    if not np.isfinite(freq_hz).all():
        raise InvalidFrequencyGridError("freq_hz must be finite.")
    if np.any(freq_hz < 0.0):
        raise InvalidFrequencyGridError("freq_hz must be non-negative.")
    if np.any(np.diff(freq_hz) <= 0.0):
        raise InvalidFrequencyGridError("freq_hz must be strictly increasing.")

    H = np.asarray(H, dtype=complex)
    if H.shape != freq_hz.shape:
        raise ValueError("H must have the same shape as freq_hz.")
    if not np.isfinite(H.real).all() or not np.isfinite(H.imag).all():
        raise ValueError("H must be finite.")
    if freq_hz.size < max(3, cfg.n_poles + 1):
        raise InvalidFrequencyGridError("Not enough samples for requested number of poles.")

    if cfg.weighting == "custom":
        if weights is None:
            raise ValueError("weights must be provided when weighting='custom'.")
        weights = np.asarray(weights, dtype=float)
        if weights.shape != freq_hz.shape:
            raise ValueError("weights must have the same shape as freq_hz.")
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("weights must be positive and finite.")
    elif weights is not None:
        weights = None

    return freq_hz, H, weights


def _make_weights(H: np.ndarray, cfg: VectorFitConfig, weights: np.ndarray | None) -> np.ndarray:
    if cfg.weighting == "uniform":
        w = np.ones_like(H.real, dtype=float)
    elif cfg.weighting == "inv_mag":
        w = 1.0 / np.maximum(np.abs(H), 1e-12)
    elif cfg.weighting == "mag_phase":
        mag = np.abs(H)
        phase = np.unwrap(np.angle(H))
        mag_weight = 1.0 / np.maximum(mag, 1e-12)
        phase_dev = np.abs(phase - np.median(phase))
        phase_weight = 1.0 / np.maximum(phase_dev, 1e-3)
        w = mag_weight + cfg.phase_weight * phase_weight
    else:
        if weights is None:
            raise ValueError("Custom weighting selected but no weights provided.")
        w = weights.astype(float)
    w = np.clip(w, 1.0 / cfg.max_weight, cfg.max_weight)
    return w


def _initial_poles(freq_hz: np.ndarray, cfg: VectorFitConfig) -> np.ndarray:
    positive = freq_hz[freq_hz > 0.0]
    if positive.size == 0:
        raise InvalidFrequencyGridError("At least one positive frequency is required.")
    w_min = 2.0 * np.pi * positive.min()
    w_max = 2.0 * np.pi * positive.max()
    if w_min == w_max:
        w_max = w_min * 10.0

    n_real = cfg.real_poles if cfg.real_poles is not None else cfg.n_poles % 2
    n_pairs = (cfg.n_poles - n_real) // 2

    poles: list[complex] = []
    if n_pairs > 0:
        imag_grid = np.logspace(np.log10(w_min), np.log10(w_max), n_pairs)
        for imag in imag_grid:
            sigma = -cfg.init_pole_scale * imag
            pole = complex(sigma, imag)
            poles.append(pole)
            poles.append(np.conj(pole))
    if n_real > 0:
        real_grid = np.logspace(np.log10(w_min), np.log10(w_max), n_real)
        for real in real_grid:
            poles.append(complex(-cfg.init_pole_scale * real, 0.0))

    poles = np.array(poles[: cfg.n_poles], dtype=complex)
    poles, _ = _sort_poles_and_residues(poles)
    return poles


def _stabilize_poles(poles: np.ndarray) -> np.ndarray:
    stabilized = poles.astype(complex, copy=True)
    for idx, pole in enumerate(stabilized):
        if pole.real >= 0.0:
            stabilized[idx] = complex(-abs(pole.real) - 1e-12, pole.imag)
    return stabilized


def _enforce_conjugate_symmetry(
    poles: np.ndarray,
    residues: np.ndarray | None = None,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    poles = poles.astype(complex, copy=True)
    residues_out = None if residues is None else residues.astype(complex, copy=True)

    warnings: list[str] = []
    order = np.lexsort((poles.real, poles.imag))
    used = np.zeros(poles.size, dtype=bool)

    for idx in order:
        if used[idx]:
            continue
        pole = poles[idx]
        if abs(pole.imag) <= tol:
            poles[idx] = complex(pole.real, 0.0)
            if residues_out is not None:
                residues_out[idx] = complex(residues_out[idx].real, 0.0)
            used[idx] = True
            continue
        if pole.imag < -tol:
            continue

        target = np.conj(pole)
        candidates = [j for j in order if (not used[j]) and poles[j].imag < -tol]
        if not candidates:
            warnings.append(f"No conjugate partner available for pole {pole}.")
            used[idx] = True
            continue
        diffs = np.abs(poles[candidates] - target)
        partner = candidates[int(np.argmin(diffs))]
        if diffs.min() > max(1e-8, 1e-3 * abs(pole)):
            warnings.append(f"No conjugate partner within tolerance for pole {pole}.")
            used[idx] = True
            continue

        p_avg = 0.5 * (poles[idx] + np.conj(poles[partner]))
        poles[idx] = p_avg
        poles[partner] = np.conj(p_avg)
        if residues_out is not None:
            r_avg = 0.5 * (residues_out[idx] + np.conj(residues_out[partner]))
            residues_out[idx] = r_avg
            residues_out[partner] = np.conj(r_avg)
        used[idx] = True
        used[partner] = True

    for idx, pole in enumerate(poles):
        if abs(pole.imag) <= tol:
            poles[idx] = complex(pole.real, 0.0)
            if residues_out is not None:
                residues_out[idx] = complex(residues_out[idx].real, 0.0)

    return poles, residues_out, warnings


def _sort_poles_and_residues(
    poles: np.ndarray, residues: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    order = np.lexsort((poles.imag, poles.real))
    poles_sorted = poles[order]
    if residues is None:
        return poles_sorted, None
    return poles_sorted, residues[order]


def _solve_weighted_ls(
    A: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
    max_condition: float,
    fail_on_ill_conditioned: bool,
) -> tuple[np.ndarray, float]:
    w = np.sqrt(weights).astype(float)
    A_w = A * w[:, None]
    y_w = y * w
    if ridge_lambda > 0.0:
        n_params = A_w.shape[1]
        ridge = np.sqrt(ridge_lambda) * np.eye(n_params, dtype=A_w.dtype)
        A_solve = np.vstack([A_w, ridge])
        y_solve = np.concatenate([y_w, np.zeros(n_params, dtype=y_w.dtype)])
    else:
        A_solve = A_w
        y_solve = y_w

    try:
        x, _, _, svals = np.linalg.lstsq(A_solve, y_solve, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise IllConditionedSolveError("Least squares solve failed.") from exc

    if svals is None or svals.size == 0:
        cond = float("inf")
    else:
        cond = float(np.max(svals) / np.maximum(np.min(svals), 1e-16))
    if (not np.isfinite(cond) or cond > max_condition) and fail_on_ill_conditioned:
        raise IllConditionedSolveError(
            f"Ill-conditioned solve detected (cond={cond:.3e} > {max_condition:.3e})."
        )

    return x, cond


def _relocate_poles(poles: np.ndarray, c: np.ndarray) -> np.ndarray:
    diag = np.diag(poles)
    ones = np.ones((poles.size, 1), dtype=complex)
    A = diag - c.reshape(-1, 1) @ ones.T
    return np.linalg.eigvals(A)


def _solve_vf_iteration(
    poles: np.ndarray,
    s: np.ndarray,
    H: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
    max_condition: float,
    fail_on_ill_conditioned: bool,
) -> tuple[np.ndarray, np.ndarray, complex, complex, float]:
    basis = _compute_basis(poles, s)
    A = np.hstack(
        [
            basis,
            -H[:, None] * basis,
            np.ones((s.size, 1), dtype=complex),
            s[:, None],
        ]
    )
    x, cond = _solve_weighted_ls(
        A, H, weights, ridge_lambda, max_condition, fail_on_ill_conditioned
    )
    n_poles = poles.size
    residues = x[:n_poles]
    c = x[n_poles : 2 * n_poles]
    d = x[-2]
    h = x[-1]
    return residues, c, d, h, cond


def _solve_fixed_poles(
    poles: np.ndarray,
    s: np.ndarray,
    H: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
    max_condition: float,
    fail_on_ill_conditioned: bool,
) -> tuple[np.ndarray, complex, complex, float]:
    basis = _compute_basis(poles, s)
    A = np.hstack([basis, np.ones((s.size, 1), dtype=complex), s[:, None]])
    x, cond = _solve_weighted_ls(
        A, H, weights, ridge_lambda, max_condition, fail_on_ill_conditioned
    )
    residues = x[: poles.size]
    d = x[-2]
    h = x[-1]
    return residues, d, h, cond


def _residual_metrics(H: np.ndarray, H_fit: np.ndarray) -> tuple[float, float, float, float]:
    abs_err = np.abs(H_fit - H)
    rel_err = abs_err / np.maximum(np.abs(H), 1e-12)
    abs_rms = float(np.sqrt(np.mean(abs_err**2)))
    rel_rms = float(np.sqrt(np.mean(rel_err**2)))
    max_abs = float(np.max(abs_err))
    max_rel = float(np.max(rel_err))
    return rel_rms, abs_rms, max_rel, max_abs


def vector_fit(
    freq_hz: np.ndarray,
    H: np.ndarray,
    kind: Literal["impedance", "admittance"],
    cfg: VectorFitConfig,
    weights: np.ndarray | None = None,
) -> VectorFitResult:
    """Fit a pole-residue rational model to one-port frequency response data."""
    freq_hz, H, weights = _validate_vector_fit_inputs(freq_hz, H, cfg, weights)
    if kind not in ("impedance", "admittance"):
        raise ValueError("kind must be 'impedance' or 'admittance'.")

    weights = _make_weights(H, cfg, weights)
    s = 1j * 2.0 * np.pi * freq_hz

    poles = _initial_poles(freq_hz, cfg)
    diagnostics: dict[str, Any] = {"iterations": []}
    conjugate_warnings: list[str] = []
    final_pole_shift = float("nan")
    n_iters_run = 0
    converged = False

    for iteration in range(cfg.n_iters):
        residues, c, d, h, cond = _solve_vf_iteration(
            poles,
            s,
            H,
            weights,
            cfg.ridge_lambda,
            cfg.max_condition,
            cfg.fail_on_ill_conditioned,
        )
        new_poles = _relocate_poles(poles, c)
        if cfg.stabilize_poles:
            new_poles = _stabilize_poles(new_poles)
        if cfg.enforce_conjugate_symmetry:
            new_poles, _, iter_warnings = _enforce_conjugate_symmetry(new_poles)
            conjugate_warnings.extend(iter_warnings)
        new_poles, _ = _sort_poles_and_residues(new_poles)

        pole_shift = float(np.max(np.abs(new_poles - poles)))
        diagnostics["iterations"].append(
            {"iter": iteration, "cond": cond, "pole_shift": pole_shift}
        )
        poles = new_poles
        final_pole_shift = pole_shift
        n_iters_run = iteration + 1

        if cfg.early_stop and n_iters_run >= cfg.min_iters and pole_shift < cfg.pole_shift_tol:
            converged = True
            break

    residues, d, h, final_cond = _solve_fixed_poles(
        poles,
        s,
        H,
        weights,
        cfg.ridge_lambda,
        cfg.max_condition,
        cfg.fail_on_ill_conditioned,
    )
    if cfg.enforce_conjugate_symmetry:
        poles, residues, final_warnings = _enforce_conjugate_symmetry(poles, residues)
        conjugate_warnings.extend(final_warnings)
    if cfg.stabilize_poles:
        poles = _stabilize_poles(poles)
    poles, residues = _sort_poles_and_residues(poles, residues)

    model = RationalModel(poles=poles, residues=residues, d=d, h=h, kind=kind)
    H_fit = model.eval_s(s)
    rel_error_rms, abs_error_rms, max_rel_error, max_abs_error = _residual_metrics(H, H_fit)

    diagnostics["final_cond"] = final_cond
    diagnostics["weights"] = cfg.weighting
    diagnostics["converged"] = bool(converged or (n_iters_run >= cfg.min_iters and final_pole_shift < cfg.pole_shift_tol))
    diagnostics["n_iters_run"] = int(n_iters_run)
    diagnostics["final_pole_shift"] = float(final_pole_shift)
    diagnostics["residual_hotspots"] = refine_frequency_grid(freq_hz, np.abs(H_fit - H)).tolist()
    if conjugate_warnings:
        diagnostics["conjugate_warnings"] = conjugate_warnings

    if cfg.require_convergence and not diagnostics["converged"]:
        raise VectorFitConvergenceError(
            "Vector fitting did not converge within the requested iterations.",
            diagnostics=diagnostics,
        )

    return VectorFitResult(
        model=model,
        rel_error_rms=rel_error_rms,
        abs_error_rms=abs_error_rms,
        max_rel_error=max_rel_error,
        max_abs_error=max_abs_error,
        diagnostics=diagnostics,
    )
