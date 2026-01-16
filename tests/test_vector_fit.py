import numpy as np

from fidp.modeling import VectorFitConfig, vector_fit


def _parallel_rc_impedance(freqs: np.ndarray, r: float, c: float) -> np.ndarray:
    s = 1j * 2.0 * np.pi * freqs
    return 1.0 / (1.0 / r + s * c)


def _parallel_rlc_impedance(freqs: np.ndarray, r: float, l: float, c: float) -> np.ndarray:
    s = 1j * 2.0 * np.pi * freqs
    return 1.0 / (1.0 / r + s * c + 1.0 / (s * l))


def test_vector_fit_parallel_rc():
    r = 25.0
    c = 2e-3
    freqs = np.logspace(0, 4, 80)
    Z = _parallel_rc_impedance(freqs, r, c)

    cfg = VectorFitConfig(n_poles=1, n_iters=8, init_pole_scale=1.0)
    result = vector_fit(freqs, Z, kind="impedance", cfg=cfg)

    Z_fit = result.model.eval_freq(freqs)
    rel_err = np.median(np.abs(Z_fit - Z) / np.maximum(np.abs(Z), 1e-12))
    assert rel_err < 1e-2
    assert np.all(result.model.poles.real < 0.0)


def test_vector_fit_parallel_rlc_complex_poles():
    r = 8.0
    l = 1.5e-3
    c = 1.2e-6
    freqs = np.logspace(1, 5, 120)
    Z = _parallel_rlc_impedance(freqs, r, l, c)

    cfg = VectorFitConfig(n_poles=4, n_iters=12, init_pole_scale=1.0)
    res1 = vector_fit(freqs, Z, kind="impedance", cfg=cfg)
    res2 = vector_fit(freqs, Z, kind="impedance", cfg=cfg)

    Z_fit = res1.model.eval_freq(freqs)
    rel_err = np.median(np.abs(Z_fit - Z) / np.maximum(np.abs(Z), 1e-12))
    assert rel_err < 5e-2
    assert np.all(res1.model.poles.real < 0.0)
    assert np.any(np.abs(res1.model.poles.imag) > 0.0)

    assert np.allclose(res1.model.poles, res2.model.poles, rtol=1e-8, atol=1e-10)
    assert np.allclose(res1.model.residues, res2.model.residues, rtol=1e-8, atol=1e-10)
    assert np.isclose(res1.model.d, res2.model.d)
    assert np.isclose(res1.model.h, res2.model.h)


def test_vector_fit_deterministic():
    r = 10.0
    c = 5e-4
    freqs = np.logspace(1, 3, 50)
    Z = _parallel_rc_impedance(freqs, r, c)
    cfg = VectorFitConfig(n_poles=1, n_iters=6, init_pole_scale=0.8)

    res1 = vector_fit(freqs, Z, kind="impedance", cfg=cfg)
    res2 = vector_fit(freqs, Z, kind="impedance", cfg=cfg)

    assert np.allclose(res1.model.poles, res2.model.poles, rtol=1e-8, atol=1e-10)
    assert np.allclose(res1.model.residues, res2.model.residues, rtol=1e-8, atol=1e-10)
    assert np.isclose(res1.model.d, res2.model.d)
    assert np.isclose(res1.model.h, res2.model.h)


def test_vector_fit_weighting_modes():
    r = 8.0
    l = 1.5e-3
    c = 1.2e-6
    freqs = np.logspace(1, 5, 120)
    Z = _parallel_rlc_impedance(freqs, r, l, c)

    base_cfg = VectorFitConfig(n_poles=4, n_iters=10, init_pole_scale=1.0)
    uniform = vector_fit(freqs, Z, kind="impedance", cfg=base_cfg)

    inv_cfg = VectorFitConfig(
        n_poles=4, n_iters=10, init_pole_scale=1.0, weighting="inv_mag"
    )
    inv_mag = vector_fit(freqs, Z, kind="impedance", cfg=inv_cfg)

    custom_weights = 1.0 + (freqs / freqs.max()) ** 0.5
    custom_cfg = VectorFitConfig(
        n_poles=4, n_iters=10, init_pole_scale=1.0, weighting="custom"
    )
    custom = vector_fit(freqs, Z, kind="impedance", cfg=custom_cfg, weights=custom_weights)

    baseline = uniform.rel_error_rms
    for res in (inv_mag, custom):
        assert np.isfinite(res.rel_error_rms)
        assert res.rel_error_rms < max(2e-1, 5.0 * baseline)


def test_vector_fit_early_stopping():
    r = 15.0
    c = 8e-4
    freqs = np.logspace(1, 4, 90)
    Z = _parallel_rc_impedance(freqs, r, c)

    cfg = VectorFitConfig(
        n_poles=1,
        n_iters=50,
        init_pole_scale=1.0,
        pole_shift_tol=1e-2,
        min_iters=2,
        early_stop=True,
    )
    result = vector_fit(freqs, Z, kind="impedance", cfg=cfg)

    assert result.diagnostics["n_iters_run"] < cfg.n_iters
    assert result.diagnostics["converged"] is True
