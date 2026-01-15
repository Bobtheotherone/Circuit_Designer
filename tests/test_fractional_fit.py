import numpy as np
import pytest

from fidp.modeling import fit_cpe, estimate_alpha_profile


def _make_cpe(alpha: float, c_alpha: float, freqs: np.ndarray) -> np.ndarray:
    s = 1j * 2.0 * np.pi * freqs
    return 1.0 / (c_alpha * (s**alpha))


def test_fit_cpe_recovers_parameters():
    alpha_true = 0.55
    c_alpha_true = 2.5e-3
    freqs = np.logspace(1, 4, 80)
    Z = _make_cpe(alpha_true, c_alpha_true, freqs)

    result = fit_cpe(freqs, Z)

    assert result.alpha == pytest.approx(alpha_true, abs=1e-2)
    assert result.c_alpha == pytest.approx(c_alpha_true, rel=5e-2)

    Z_fit = _make_cpe(result.alpha, result.c_alpha, freqs)
    rel_err = np.median(np.abs(Z_fit - Z) / np.maximum(np.abs(Z), 1e-12))
    assert rel_err < 2e-2


def test_fit_cpe_handles_mild_noise():
    alpha_true = 0.45
    c_alpha_true = 1.2e-3
    freqs = np.logspace(2, 5, 90)
    Z = _make_cpe(alpha_true, c_alpha_true, freqs)
    phase = np.linspace(0.0, 2.0 * np.pi, freqs.size)
    noise = 2e-3 * (np.sin(phase) + 1j * np.cos(phase))
    Z_noisy = Z * (1.0 + noise)

    result = fit_cpe(freqs, Z_noisy)

    assert result.alpha == pytest.approx(alpha_true, abs=5e-2)


def test_alpha_profile_tracks_constant_order():
    alpha_true = 0.6
    c_alpha_true = 4e-3
    freqs = np.logspace(1, 4, 60)
    Z = _make_cpe(alpha_true, c_alpha_true, freqs)

    profile = estimate_alpha_profile(freqs, Z, window=11)

    center = profile[5:-5]
    assert np.nanmedian(center) == pytest.approx(alpha_true, abs=2e-2)


def test_fit_cpe_rejects_bad_shapes():
    freqs = np.array([10.0, 100.0, 1000.0])
    Z = np.array([1.0 + 0.0j, 2.0 + 0.0j])

    with pytest.raises(ValueError):
        fit_cpe(freqs, Z)
