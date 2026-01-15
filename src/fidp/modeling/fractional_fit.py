"""Fractional-order impedance fitting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CPEFitResult:
    """Result of a constant-phase element fit."""

    alpha: float
    c_alpha: float
    band: tuple[float, float]
    phase_ripple_deg: float
    rmse_logmag: float
    rmse_phase_deg: float
    n_points: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _validate_frequency_data(freq_hz: np.ndarray, Z: np.ndarray) -> None:
    if freq_hz.ndim != 1:
        raise ValueError("freq_hz must be a 1D array.")
    if freq_hz.shape != Z.shape:
        raise ValueError("freq_hz and Z must have the same shape.")
    if np.any(freq_hz <= 0.0):
        raise ValueError("freq_hz must be strictly positive for CPE fitting.")
    if not np.isfinite(freq_hz).all():
        raise ValueError("freq_hz must be finite.")
    if not np.isfinite(Z.real).all() or not np.isfinite(Z.imag).all():
        raise ValueError("Z must be finite.")


def _select_band(
    freq_hz: np.ndarray,
    Z: np.ndarray,
    band: tuple[float, float] | None,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    if band is None:
        mask = np.ones_like(freq_hz, dtype=bool)
        band_used = (float(freq_hz.min()), float(freq_hz.max()))
    else:
        if len(band) != 2:
            raise ValueError("band must be a (fmin, fmax) tuple.")
        fmin, fmax = float(band[0]), float(band[1])
        if fmin <= 0.0 or fmax <= 0.0 or fmin >= fmax:
            raise ValueError("band must have 0 < fmin < fmax.")
        mask = (freq_hz >= fmin) & (freq_hz <= fmax)
        if not np.any(mask):
            raise ValueError("band selection produced no data points.")
        band_used = (fmin, fmax)

    freq_sel = freq_hz[mask]
    Z_sel = Z[mask]
    if weights is None:
        weights_sel = np.ones_like(freq_sel, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != freq_hz.shape:
            raise ValueError("weights must have the same shape as freq_hz.")
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("weights must be positive and finite.")
        weights_sel = weights[mask]
    return freq_sel, Z_sel, weights_sel, band_used


def fit_cpe(
    freq_hz: np.ndarray,
    Z: np.ndarray,
    band: tuple[float, float] | None = None,
    weights: np.ndarray | None = None,
) -> CPEFitResult:
    """
    Fit a constant-phase element impedance model to frequency-domain data.

    Model: Z = 1 / (C_alpha * (j * omega) ** alpha)
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    _validate_frequency_data(freq_hz, Z)

    freq_sel, Z_sel, weights_sel, band_used = _select_band(freq_hz, Z, band, weights)
    omega = 2.0 * np.pi * freq_sel

    mag = np.abs(Z_sel)
    if np.any(mag <= 0.0):
        raise ValueError("Z magnitude must be positive for log fitting.")
    log_mag = np.log(mag)
    phase = np.unwrap(np.angle(Z_sel))

    log_omega = np.log(omega)
    n_points = freq_sel.size

    A_mag = np.column_stack([-np.ones(n_points), -log_omega])
    y_mag = log_mag

    phase_weight = 1.0
    A_phase = np.column_stack([np.zeros(n_points), -0.5 * np.pi * np.ones(n_points)])
    y_phase = phase

    weights_sqrt = np.sqrt(weights_sel.astype(float))
    A_stack = np.vstack(
        [A_mag * weights_sqrt[:, None], A_phase * (weights_sqrt * phase_weight)[:, None]]
    )
    y_stack = np.concatenate([y_mag * weights_sqrt, y_phase * weights_sqrt * phase_weight])

    params, *_ = np.linalg.lstsq(A_stack, y_stack, rcond=None)
    log_c_alpha = params[0]
    alpha = params[1]
    c_alpha = float(np.exp(log_c_alpha))

    s = 1j * omega
    Z_fit = 1.0 / (c_alpha * (s**alpha))
    log_mag_fit = np.log(np.abs(Z_fit))
    phase_fit = np.unwrap(np.angle(Z_fit))

    rmse_logmag = float(np.sqrt(np.mean((log_mag - log_mag_fit) ** 2)))
    rmse_phase = float(np.sqrt(np.mean((phase - phase_fit) ** 2)))
    rmse_phase_deg = rmse_phase * 180.0 / np.pi

    phase_deg = phase * 180.0 / np.pi
    phase_ripple_deg = float(np.max(phase_deg) - np.min(phase_deg))

    alpha_mag = -np.polyfit(log_omega, log_mag, 1)[0]
    alpha_phase = -np.average(phase, weights=weights_sel) / (0.5 * np.pi)

    diagnostics = {
        "alpha_mag": float(alpha_mag),
        "alpha_phase": float(alpha_phase),
        "log_c_alpha": float(log_c_alpha),
        "band_used": band_used,
    }

    return CPEFitResult(
        alpha=float(alpha),
        c_alpha=c_alpha,
        band=band_used,
        phase_ripple_deg=phase_ripple_deg,
        rmse_logmag=rmse_logmag,
        rmse_phase_deg=rmse_phase_deg,
        n_points=n_points,
        diagnostics=diagnostics,
    )


def estimate_alpha_profile(freq_hz: np.ndarray, Z: np.ndarray, window: int = 11) -> np.ndarray:
    """
    Estimate local alpha(omega) via rolling log-magnitude slope.

    Returns an array of alpha estimates with NaNs at the edges where the
    window cannot be centered.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    _validate_frequency_data(freq_hz, Z)

    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3.")
    if np.any(np.diff(freq_hz) <= 0.0):
        raise ValueError("freq_hz must be strictly increasing for alpha profiling.")

    omega = 2.0 * np.pi * freq_hz
    mag = np.abs(Z)
    if np.any(mag <= 0.0):
        raise ValueError("Z magnitude must be positive for log fitting.")

    log_omega = np.log(omega)
    log_mag = np.log(mag)

    n = freq_hz.size
    half = window // 2
    alpha_profile = np.full(n, np.nan, dtype=float)

    for idx in range(half, n - half):
        sl = slice(idx - half, idx + half + 1)
        coeffs = np.polyfit(log_omega[sl], log_mag[sl], 1)
        alpha_profile[idx] = -coeffs[0]

    return alpha_profile
