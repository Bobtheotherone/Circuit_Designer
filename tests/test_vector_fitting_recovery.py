import json
from pathlib import Path

import numpy as np
import pytest

from fidp.analysis.fitting.vector_fitting import RationalModel, VectorFitConfig, vector_fit
from fidp.errors import (
    InvalidFrequencyGridError,
    VectorFitConvergenceError,
    IllConditionedSolveError,
)


_FIXTURE = Path("tests/fixtures/vector_fit_recovery.json")


def _load_fixture():
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    freq_hz = np.array(data["freq_hz"], dtype=float)
    poles = np.array([complex(p["real"], p["imag"]) for p in data["model"]["poles"]])
    residues = np.array([complex(r["real"], r["imag"]) for r in data["model"]["residues"]])
    d = complex(data["model"]["d"]["real"], data["model"]["d"]["imag"])
    h = complex(data["model"]["h"]["real"], data["model"]["h"]["imag"])
    model = RationalModel(poles=poles, residues=residues, d=d, h=h, kind="impedance")
    Z = np.array([complex(z[0], z[1]) for z in data["response"]], dtype=complex)
    return freq_hz, Z, model


def test_vector_fit_recovers_rational_fixture():
    freq_hz, Z, model_true = _load_fixture()
    cfg = VectorFitConfig(
        n_poles=model_true.poles.size,
        n_iters=25,
        pole_shift_tol=1e-3,
        min_iters=2,
        require_convergence=True,
    )
    result = vector_fit(freq_hz, Z, kind="impedance", cfg=cfg)

    dense = np.logspace(np.log10(freq_hz.min()), np.log10(freq_hz.max()), 200)
    target_dense = model_true.eval_freq(dense)
    fit_dense = result.model.eval_freq(dense)

    rel_rms = np.sqrt(np.mean(np.abs(target_dense - fit_dense) ** 2)) / np.maximum(
        np.sqrt(np.mean(np.abs(target_dense) ** 2)), 1e-12
    )

    assert rel_rms < 2e-2
    assert np.all(result.model.poles.real < 0.0)
    assert result.diagnostics["converged"] is True


def test_vector_fit_rejects_unsorted_grid():
    freq = np.array([1.0, 10.0, 5.0])
    Z = np.array([1.0 + 0.1j, 0.5 + 0.2j, 0.8 + 0.05j])
    cfg = VectorFitConfig(n_poles=1)

    with pytest.raises(InvalidFrequencyGridError):
        vector_fit(freq, Z, kind="impedance", cfg=cfg)


def test_vector_fit_non_convergence_raises():
    freq = np.logspace(1, 3, 40)
    Z = 1.0 / (1.0 + 1j * 2.0 * np.pi * freq * 1e-3)
    cfg = VectorFitConfig(
        n_poles=2,
        n_iters=1,
        min_iters=1,
        pole_shift_tol=1e-12,
        require_convergence=True,
    )

    with pytest.raises(VectorFitConvergenceError):
        vector_fit(freq, Z, kind="impedance", cfg=cfg)


def test_vector_fit_ill_conditioned_raises():
    freq = np.logspace(1, 3, 40)
    Z = 1.0 / (1.0 + 1j * 2.0 * np.pi * freq * 1e-3)
    cfg = VectorFitConfig(
        n_poles=2,
        n_iters=4,
        max_condition=1e3,
        fail_on_ill_conditioned=True,
    )

    with pytest.raises(IllConditionedSolveError):
        vector_fit(freq, Z, kind="impedance", cfg=cfg)
