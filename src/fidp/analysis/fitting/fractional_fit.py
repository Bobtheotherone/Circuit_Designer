"""Fractional-order impedance identification utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fidp.errors import InvalidFrequencyGridError


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


@dataclass(frozen=True)
class FractionalFitConfig:
    """Configuration for fractional-order estimation."""

    window: int = 11
    min_band_points: int = 20
    band_tolerance: float = 0.05
    bootstrap_samples: int = 200
    seed: int | None = 7
    heteroskedastic_power: float = 2.0
    phase_smooth_window: int = 9
    combine_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.window < 3 or self.window % 2 == 0:
            raise ValueError("window must be an odd integer >= 3.")
        if self.min_band_points < 5:
            raise ValueError("min_band_points must be >= 5.")
        if self.band_tolerance <= 0.0:
            raise ValueError("band_tolerance must be positive.")
        if self.bootstrap_samples < 20:
            raise ValueError("bootstrap_samples must be >= 20.")
        if self.heteroskedastic_power <= 0.0:
            raise ValueError("heteroskedastic_power must be positive.")
        if self.phase_smooth_window < 3 or self.phase_smooth_window % 2 == 0:
            raise ValueError("phase_smooth_window must be an odd integer >= 3.")
        if not 0.0 <= self.combine_weight <= 1.0:
            raise ValueError("combine_weight must be between 0 and 1.")


@dataclass(frozen=True)
class BandEstimate:
    """Piecewise alpha estimate for a frequency band."""

    f_start: float
    f_end: float
    alpha: float
    alpha_phase: float
    alpha_mag: float
    r2_mag: float
    phase_ripple_deg: float
    n_points: int
    ci_low: float
    ci_high: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class FractionalOrderReport:
    """Fractional-order estimation report."""

    freq_hz: np.ndarray
    alpha_omega: np.ndarray
    alpha_phase: np.ndarray
    alpha_mag: np.ndarray
    bands: list[BandEstimate]
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "freq_hz": [float(x) for x in self.freq_hz],
            "alpha_omega": [float(x) for x in self.alpha_omega],
            "alpha_phase": [float(x) for x in self.alpha_phase],
            "alpha_mag": [float(x) for x in self.alpha_mag],
            "bands": [
                {
                    "f_start": band.f_start,
                    "f_end": band.f_end,
                    "alpha": band.alpha,
                    "alpha_phase": band.alpha_phase,
                    "alpha_mag": band.alpha_mag,
                    "r2_mag": band.r2_mag,
                    "phase_ripple_deg": band.phase_ripple_deg,
                    "n_points": band.n_points,
                    "ci_low": band.ci_low,
                    "ci_high": band.ci_high,
                    "diagnostics": dict(band.diagnostics),
                }
                for band in self.bands
            ],
            "metrics": dict(self.metrics),
            "warnings": list(self.warnings),
        }


def _validate_frequency_data(freq_hz: np.ndarray, Z: np.ndarray) -> None:
    if freq_hz.ndim != 1:
        raise InvalidFrequencyGridError("freq_hz must be a 1D array.")
    if freq_hz.shape != Z.shape:
        raise ValueError("freq_hz and Z must have the same shape.")
    if np.any(freq_hz <= 0.0):
        raise InvalidFrequencyGridError("freq_hz must be strictly positive.")
    if not np.isfinite(freq_hz).all():
        raise InvalidFrequencyGridError("freq_hz must be finite.")
    if not np.isfinite(Z.real).all() or not np.isfinite(Z.imag).all():
        raise ValueError("Z must be finite.")
    if np.any(np.diff(freq_hz) <= 0.0):
        raise InvalidFrequencyGridError("freq_hz must be strictly increasing.")


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
    """Fit a constant-phase element model to frequency-domain data."""
    freq_hz = np.asarray(freq_hz, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    _validate_frequency_data(freq_hz, Z)

    freq_sel, Z_sel, weights_sel, band_used = _select_band(freq_hz, Z, band, weights)
    omega = 2.0 * np.pi * freq_sel

    mag = np.abs(Z_sel)
    if np.any(mag <= 0.0):
        raise ValueError("Z magnitude must be positive for log fitting.")
    log_mag = np.log(mag)
    phi = np.angle(Z_sel)
    mean_vec = np.mean(np.exp(1j * phi))
    phi_ref = float(np.angle(mean_vec)) if abs(mean_vec) >= 1e-12 else float(np.median(phi))
    phase = phi + 2.0 * np.pi * np.round((phi_ref - phi) / (2.0 * np.pi))

    log_omega = np.log(omega)
    n_points = freq_sel.size

    A_mag = np.column_stack([-np.ones(n_points), -log_omega, np.zeros(n_points)])
    y_mag = log_mag

    phase_weight = 1.0
    A_phase = np.column_stack(
        [np.zeros(n_points), -0.5 * np.pi * np.ones(n_points), np.ones(n_points)]
    )
    y_phase = phase

    weights_sqrt = np.sqrt(weights_sel.astype(float))
    A_stack = np.vstack(
        [A_mag * weights_sqrt[:, None], A_phase * (weights_sqrt * phase_weight)[:, None]]
    )
    y_stack = np.concatenate([y_mag * weights_sqrt, y_phase * weights_sqrt * phase_weight])

    params, *_ = np.linalg.lstsq(A_stack, y_stack, rcond=None)
    log_c_alpha = params[0]
    alpha = params[1]
    phase_offset = params[2]
    c_alpha = float(np.exp(log_c_alpha))

    s = 1j * omega
    Z_fit = 1.0 / (c_alpha * (s**alpha))
    log_mag_fit = np.log(np.abs(Z_fit))
    phase_fit = np.full_like(phase, phase_offset - 0.5 * np.pi * alpha)

    rmse_logmag = float(np.sqrt(np.mean((log_mag - log_mag_fit) ** 2)))
    rmse_phase = float(np.sqrt(np.mean((phase - phase_fit) ** 2)))
    rmse_phase_deg = rmse_phase * 180.0 / np.pi

    phase_deg = phase * 180.0 / np.pi
    phase_ripple_deg = float(np.max(phase_deg) - np.min(phase_deg))

    alpha_mag = -np.polyfit(log_omega, log_mag, 1)[0]
    alpha_phase = -np.average(phase - phase_offset, weights=weights_sel) / (0.5 * np.pi)

    diagnostics = {
        "alpha_mag": float(alpha_mag),
        "alpha_phase": float(alpha_phase),
        "log_c_alpha": float(log_c_alpha),
        "phase_offset_rad": float(phase_offset),
        "phase_ref_rad": float(phi_ref),
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
    """Estimate local alpha(omega) via rolling log-magnitude slope."""
    freq_hz = np.asarray(freq_hz, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    _validate_frequency_data(freq_hz, Z)

    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3.")

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


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    n = values.size
    half = window // 2
    result = np.empty(n, dtype=float)
    for idx in range(n):
        lo = max(0, idx - half)
        hi = min(n, idx + half + 1)
        result[idx] = float(np.median(values[lo:hi]))
    return result


def _rolling_slope(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    n = x.size
    half = window // 2
    slopes = np.full(n, np.nan, dtype=float)
    for idx in range(half, n - half):
        sl = slice(idx - half, idx + half + 1)
        coeffs = np.polyfit(x[sl], y[sl], 1)
        slopes[idx] = coeffs[0]
    if np.isnan(slopes[0]):
        slopes[:half] = slopes[half]
        slopes[-half:] = slopes[-half - 1]
    return slopes


def _segment_alpha(freq_hz: np.ndarray, alpha: np.ndarray, cfg: FractionalFitConfig) -> list[slice]:
    segments: list[slice] = []
    start = 0
    running = alpha[0]
    for idx in range(1, alpha.size):
        running = 0.9 * running + 0.1 * alpha[idx]
        if abs(alpha[idx] - running) > cfg.band_tolerance and (idx - start) >= cfg.min_band_points:
            segments.append(slice(start, idx))
            start = idx
            running = alpha[idx]
    segments.append(slice(start, alpha.size))
    return segments


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    coeffs = np.polyfit(x, y, 1, w=weights)
    slope = coeffs[0]
    intercept = coeffs[1]
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return slope, intercept, r2


def estimate_fractional_order(
    freq_hz: np.ndarray,
    Z: np.ndarray,
    config: FractionalFitConfig | None = None,
) -> FractionalOrderReport:
    """Estimate variable-order alpha(omega) and piecewise bands with confidence intervals."""
    config = config or FractionalFitConfig()
    freq_hz = np.asarray(freq_hz, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    _validate_frequency_data(freq_hz, Z)
    if freq_hz.size < config.window:
        raise InvalidFrequencyGridError("Frequency grid too small for requested window.")

    omega = 2.0 * np.pi * freq_hz
    log_omega = np.log(omega)
    mag = np.abs(Z)
    if np.any(mag <= 0.0):
        raise ValueError("Z magnitude must be positive for log fitting.")
    log_mag = np.log(mag)

    phase = np.unwrap(np.angle(Z))
    phase = phase - 2.0 * np.pi * np.round(np.median(phase) / (2.0 * np.pi))

    alpha_mag = -_rolling_slope(log_omega, log_mag, config.window)
    alpha_phase = -2.0 * phase / np.pi

    alpha_phase_smooth = _rolling_median(alpha_phase, config.phase_smooth_window)
    alpha_mag_smooth = _rolling_median(alpha_mag, config.phase_smooth_window)
    alpha_omega = (
        config.combine_weight * alpha_mag_smooth + (1.0 - config.combine_weight) * alpha_phase_smooth
    )

    segments = _segment_alpha(freq_hz, alpha_omega, config)
    bands: list[BandEstimate] = []
    warnings: list[str] = []

    rng = np.random.default_rng(config.seed)
    weights_full = 1.0 / np.maximum(mag, 1e-12) ** config.heteroskedastic_power
    weights_full = weights_full / np.sum(weights_full)

    for seg in segments:
        idx = np.arange(seg.start, seg.stop)
        if idx.size < config.min_band_points:
            continue
        band_freq = freq_hz[idx]
        band_log_omega = log_omega[idx]
        band_log_mag = log_mag[idx]
        band_phase = phase[idx]
        band_weights = weights_full[idx]
        band_weights = band_weights / np.sum(band_weights)

        slope, intercept, r2_mag = _weighted_linear_fit(band_log_omega, band_log_mag, band_weights)
        alpha_mag_band = -slope
        alpha_phase_band = -2.0 * float(np.median(band_phase)) / np.pi
        alpha_band = 0.5 * (alpha_mag_band + alpha_phase_band)

        phase_dev = np.max(np.abs(band_phase - np.median(band_phase)))
        phase_ripple_deg = float(phase_dev * 180.0 / np.pi)

        if idx.size < config.min_band_points:
            warnings.append("Band skipped due to insufficient points.")
            continue

        samples = []
        for _ in range(config.bootstrap_samples):
            choice = rng.choice(idx, size=idx.size, replace=True, p=band_weights)
            sample_log_omega = log_omega[choice]
            sample_log_mag = log_mag[choice]
            sample_phase = phase[choice]
            sample_weights = weights_full[choice]
            sample_weights = sample_weights / np.sum(sample_weights)
            slope_s, _, _ = _weighted_linear_fit(sample_log_omega, sample_log_mag, sample_weights)
            alpha_mag_s = -slope_s
            alpha_phase_s = -2.0 * float(np.median(sample_phase)) / np.pi
            samples.append(0.5 * (alpha_mag_s + alpha_phase_s))
        sample_arr = np.array(samples, dtype=float)
        ci_low, ci_high = np.percentile(sample_arr, [2.5, 97.5])

        bands.append(
            BandEstimate(
                f_start=float(band_freq[0]),
                f_end=float(band_freq[-1]),
                alpha=float(alpha_band),
                alpha_phase=float(alpha_phase_band),
                alpha_mag=float(alpha_mag_band),
                r2_mag=float(r2_mag),
                phase_ripple_deg=phase_ripple_deg,
                n_points=int(idx.size),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
                diagnostics={"intercept_log_mag": float(intercept)},
            )
        )

    phase_jumps = np.sum(np.abs(np.diff(phase)) > np.pi)
    if phase_jumps > 0:
        warnings.append("Phase wraps detected; check alpha_phase stability.")

    alpha_spikes = np.sum(np.abs(np.diff(alpha_omega)) > 0.5)
    if alpha_spikes > 0:
        warnings.append("Alpha spikes detected; check smoothing or grid density.")

    spacing = np.diff(np.log10(freq_hz))
    if np.max(spacing) / np.maximum(np.min(spacing), 1e-12) > 5.0:
        warnings.append("Non-uniform grid spacing may introduce artifacts.")

    metrics = {
        "alpha_mag_median": float(np.median(alpha_mag)),
        "alpha_phase_median": float(np.median(alpha_phase)),
        "phase_wraps": int(phase_jumps),
        "alpha_spikes": int(alpha_spikes),
    }

    return FractionalOrderReport(
        freq_hz=freq_hz,
        alpha_omega=alpha_omega,
        alpha_phase=alpha_phase,
        alpha_mag=alpha_mag,
        bands=bands,
        metrics=metrics,
        warnings=warnings,
    )
