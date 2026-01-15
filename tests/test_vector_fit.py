import numpy as np

from fidp.modeling import VectorFitConfig, vector_fit


def _parallel_rc_impedance(freqs: np.ndarray, r: float, c: float) -> np.ndarray:
    s = 1j * 2.0 * np.pi * freqs
    return 1.0 / (1.0 / r + s * c)


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
