"""Micro-benchmark for vector fitting performance."""

from __future__ import annotations

import time

import numpy as np

from fidp.analysis.fitting.vector_fitting import VectorFitConfig, RationalModel, vector_fit, log_frequency_grid


def _make_model() -> RationalModel:
    poles = np.array([
        -20.0 + 80.0j,
        -20.0 - 80.0j,
        -150.0 + 600.0j,
        -150.0 - 600.0j,
        -1200.0,
        -5000.0,
    ])
    residues = np.array([
        12.0 - 4.0j,
        12.0 + 4.0j,
        8.0 - 3.0j,
        8.0 + 3.0j,
        40.0,
        15.0,
    ])
    return RationalModel(poles=poles, residues=residues, d=0.8 + 0.0j, h=0.0 + 0.0j, kind="impedance")


def main() -> None:
    freq_hz = log_frequency_grid(1.0, 1e6, 2000)
    model = _make_model()
    Z = model.eval_freq(freq_hz)

    cfg = VectorFitConfig(n_poles=6, n_iters=12, init_pole_scale=1.0, require_convergence=False)

    start = time.perf_counter()
    result = vector_fit(freq_hz, Z, kind="impedance", cfg=cfg)
    elapsed = time.perf_counter() - start

    print("Vector fitting micro-benchmark")
    print(f"Points: {freq_hz.size} | Poles: {cfg.n_poles} | Iterations: {result.diagnostics['n_iters_run']}")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"RMS rel error: {result.rel_error_rms:.3e}")


if __name__ == "__main__":
    main()
