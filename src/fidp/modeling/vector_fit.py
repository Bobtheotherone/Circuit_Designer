"""Vector fitting for rational macromodels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


def _complex_to_json(value: complex) -> dict[str, float]:
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
        result = self.d + self.h * s
        for pole, residue in zip(self.poles, self.residues):
            result = result + residue / (s - pole)
        return result

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
        }

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> "RationalModel":
        poles = np.array([_json_to_complex(item) for item in data["poles"]], dtype=complex)
        residues = np.array([_json_to_complex(item) for item in data["residues"]], dtype=complex)
        d = _json_to_complex(data["d"])
        h = _json_to_complex(data["h"])
        kind = data["kind"]
        return RationalModel(poles=poles, residues=residues, d=d, h=h, kind=kind)


@dataclass
class VectorFitConfig:
    """Configuration for vector fitting."""

    n_poles: int
    n_iters: int = 10
    init_pole_scale: float = 1.0
    weighting: Literal["uniform", "inv_mag", "custom"] = "uniform"
    stabilize_poles: bool = True
    enforce_conjugate_symmetry: bool = True
    ridge_lambda: float = 0.0

    def __post_init__(self) -> None:
        if self.n_poles <= 0:
            raise ValueError("n_poles must be positive.")
        if self.n_iters <= 0:
            raise ValueError("n_iters must be positive.")
        if self.init_pole_scale <= 0.0:
            raise ValueError("init_pole_scale must be positive.")
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be non-negative.")


@dataclass
class VectorFitResult:
    """Vector fitting result."""

    model: RationalModel
    rel_error_rms: float
    max_rel_error: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _validate_vector_fit_inputs(
    freq_hz: np.ndarray,
    H: np.ndarray,
    cfg: VectorFitConfig,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq_hz = np.asarray(freq_hz, dtype=float)
    if freq_hz.ndim != 1:
        raise ValueError("freq_hz must be a 1D array.")
    if not np.isfinite(freq_hz).all():
        raise ValueError("freq_hz must be finite.")
    if np.any(freq_hz < 0.0):
        raise ValueError("freq_hz must be non-negative.")

    H = np.asarray(H, dtype=complex)
    if H.shape != freq_hz.shape:
        raise ValueError("H must have the same shape as freq_hz.")
    if not np.isfinite(H.real).all() or not np.isfinite(H.imag).all():
        raise ValueError("H must be finite.")
    if freq_hz.size < max(3, cfg.n_poles + 1):
        raise ValueError("Not enough samples for requested number of poles.")

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
        return np.ones_like(H.real, dtype=float)
    if cfg.weighting == "inv_mag":
        return 1.0 / np.maximum(np.abs(H), 1e-12)
    if weights is None:
        raise ValueError("Custom weighting selected but no weights provided.")
    return weights.astype(float)


def _initial_poles(freq_hz: np.ndarray, cfg: VectorFitConfig) -> np.ndarray:
    positive = freq_hz[freq_hz > 0.0]
    if positive.size == 0:
        raise ValueError("At least one positive frequency is required.")
    w_min = 2.0 * np.pi * positive.min()
    w_max = 2.0 * np.pi * positive.max()
    if w_min == w_max:
        w_max = w_min * 10.0

    poles: list[complex] = []
    n_pairs = cfg.n_poles // 2
    if n_pairs > 0:
        imag_grid = np.logspace(np.log10(w_min), np.log10(w_max), n_pairs)
        for imag in imag_grid:
            pole = complex(-cfg.init_pole_scale * imag, imag)
            poles.append(pole)
            poles.append(np.conj(pole))
    if cfg.n_poles % 2 == 1:
        poles.append(complex(-cfg.init_pole_scale * w_min, 0.0))

    return np.array(poles[: cfg.n_poles], dtype=complex)


def _stabilize_poles(poles: np.ndarray) -> np.ndarray:
    stabilized = poles.astype(complex, copy=True)
    for idx, pole in enumerate(stabilized):
        if pole.real > 0.0:
            stabilized[idx] = complex(-pole.real, pole.imag)
    return stabilized


def _enforce_conjugate_symmetry(
    poles: np.ndarray,
    residues: np.ndarray | None = None,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None]:
    poles = poles.astype(complex, copy=True)
    residues_out = None if residues is None else residues.astype(complex, copy=True)

    used: set[int] = set()
    imag = np.imag(poles)
    indices = np.argsort(imag)
    for idx in indices:
        if idx in used:
            continue
        pole = poles[idx]
        if abs(pole.imag) <= tol:
            poles[idx] = complex(pole.real, 0.0)
            if residues_out is not None:
                residues_out[idx] = complex(residues_out[idx].real, 0.0)
            used.add(idx)

    positive_indices = [idx for idx in indices if poles[idx].imag > tol]
    for idx in positive_indices:
        if idx in used:
            continue
        target = np.conj(poles[idx])
        diffs = np.abs(poles - target)
        diffs[idx] = np.inf
        partner = int(np.argmin(diffs))

        p_avg = 0.5 * (poles[idx] + np.conj(poles[partner]))
        poles[idx] = p_avg
        poles[partner] = np.conj(p_avg)
        if residues_out is not None:
            r_avg = 0.5 * (residues_out[idx] + np.conj(residues_out[partner]))
            residues_out[idx] = r_avg
            residues_out[partner] = np.conj(r_avg)
        used.update({idx, partner})

    return poles, residues_out


def _sort_poles_and_residues(
    poles: np.ndarray, residues: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    order = np.lexsort((poles.imag, poles.real))
    poles_sorted = poles[order]
    if residues is None:
        return poles_sorted, None
    return poles_sorted, residues[order]


def _solve_weighted_ls(
    A: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge_lambda: float
) -> tuple[np.ndarray, float]:
    w = np.sqrt(weights).astype(float)
    A_w = A * w[:, None]
    y_w = y * w
    AhA = A_w.conj().T @ A_w
    if ridge_lambda > 0.0:
        AhA = AhA + ridge_lambda * np.eye(AhA.shape[0], dtype=AhA.dtype)
    try:
        x = np.linalg.solve(AhA, A_w.conj().T @ y_w)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A_w, y_w, rcond=None)[0]
    try:
        cond = float(np.linalg.cond(AhA))
    except np.linalg.LinAlgError:
        cond = float("inf")
    return x, cond


def _relocate_poles(poles: np.ndarray, c: np.ndarray) -> np.ndarray:
    diag = np.diag(poles)
    ones = np.ones((poles.size, 1), dtype=complex)
    A = diag - c.reshape(-1, 1) @ ones.T
    new_poles = np.linalg.eigvals(A)
    return new_poles


def _solve_vf_iteration(
    poles: np.ndarray,
    s: np.ndarray,
    H: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, complex, complex, float]:
    basis = 1.0 / (s[:, None] - poles[None, :])
    A = np.hstack(
        [
            basis,
            -H[:, None] * basis,
            np.ones((s.size, 1), dtype=complex),
            s[:, None],
        ]
    )
    x, cond = _solve_weighted_ls(A, H, weights, ridge_lambda)
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
) -> tuple[np.ndarray, complex, complex, float]:
    basis = 1.0 / (s[:, None] - poles[None, :])
    A = np.hstack([basis, np.ones((s.size, 1), dtype=complex), s[:, None]])
    x, cond = _solve_weighted_ls(A, H, weights, ridge_lambda)
    residues = x[: poles.size]
    d = x[-2]
    h = x[-1]
    return residues, d, h, cond


def vector_fit(
    freq_hz: np.ndarray,
    H: np.ndarray,
    kind: Literal["impedance", "admittance"],
    cfg: VectorFitConfig,
    weights: np.ndarray | None = None,
) -> VectorFitResult:
    """
    Fit a pole-residue rational model to one-port frequency response data.
    """
    freq_hz, H, weights = _validate_vector_fit_inputs(freq_hz, H, cfg, weights)
    if kind not in ("impedance", "admittance"):
        raise ValueError("kind must be 'impedance' or 'admittance'.")

    weights = _make_weights(H, cfg, weights)
    s = 1j * 2.0 * np.pi * freq_hz

    poles = _initial_poles(freq_hz, cfg)
    poles, _ = _sort_poles_and_residues(poles)
    diagnostics: dict[str, Any] = {"iterations": []}

    for iteration in range(cfg.n_iters):
        residues, c, d, h, cond = _solve_vf_iteration(poles, s, H, weights, cfg.ridge_lambda)
        new_poles = _relocate_poles(poles, c)
        if cfg.stabilize_poles:
            new_poles = _stabilize_poles(new_poles)
        if cfg.enforce_conjugate_symmetry:
            new_poles, _ = _enforce_conjugate_symmetry(new_poles)
        new_poles, _ = _sort_poles_and_residues(new_poles)

        pole_shift = float(np.max(np.abs(new_poles - poles)))
        diagnostics["iterations"].append(
            {"iter": iteration, "cond": cond, "pole_shift": pole_shift}
        )
        poles = new_poles

    residues, d, h, final_cond = _solve_fixed_poles(poles, s, H, weights, cfg.ridge_lambda)
    if cfg.enforce_conjugate_symmetry:
        poles, residues = _enforce_conjugate_symmetry(poles, residues)
    if cfg.stabilize_poles:
        poles = _stabilize_poles(poles)
    poles, residues = _sort_poles_and_residues(poles, residues)

    model = RationalModel(poles=poles, residues=residues, d=d, h=h, kind=kind)
    H_fit = model.eval_s(s)
    rel_error = np.abs(H_fit - H) / np.maximum(np.abs(H), 1e-12)
    rel_error_rms = float(np.sqrt(np.mean(rel_error**2)))
    max_rel_error = float(np.max(rel_error))

    diagnostics["final_cond"] = final_cond
    diagnostics["weights"] = cfg.weighting

    return VectorFitResult(
        model=model,
        rel_error_rms=rel_error_rms,
        max_rel_error=max_rel_error,
        diagnostics=diagnostics,
    )
