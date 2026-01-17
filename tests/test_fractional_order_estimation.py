import numpy as np
import pytest

from fidp.analysis.fitting.fractional_fit import FractionalFitConfig, estimate_fractional_order
from fidp.errors import InvalidFrequencyGridError


def _make_cpe(alpha: float, c_alpha: float, freqs: np.ndarray) -> np.ndarray:
    s = 1j * 2.0 * np.pi * freqs
    return 1.0 / (c_alpha * (s**alpha))


def test_fractional_order_estimation_with_ci():
    alpha_true = 0.58
    c_alpha = 2.5e-3
    freqs = np.logspace(2, 5, 140)
    Z = _make_cpe(alpha_true, c_alpha, freqs)

    rng = np.random.default_rng(123)
    noise = 1.0 + 5e-3 * rng.standard_normal(Z.shape)
    Z_noisy = Z * noise

    cfg = FractionalFitConfig(window=11, bootstrap_samples=120, seed=7)
    report = estimate_fractional_order(freqs, Z_noisy, cfg)

    assert report.bands
    band = report.bands[0]
    assert abs(band.alpha - alpha_true) < 0.06
    assert band.ci_low <= alpha_true <= band.ci_high

    report2 = estimate_fractional_order(freqs, Z_noisy, cfg)
    assert report2.bands[0].ci_low == band.ci_low
    assert report2.bands[0].ci_high == band.ci_high


def test_fractional_order_rejects_short_grid():
    freqs = np.logspace(2, 3, 5)
    Z = _make_cpe(0.5, 1e-3, freqs)
    cfg = FractionalFitConfig(window=11)

    with pytest.raises(InvalidFrequencyGridError):
        estimate_fractional_order(freqs, Z, cfg)
