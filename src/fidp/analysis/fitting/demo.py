"""Demo entrypoint for Step 7 model extraction pipeline."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import subprocess

import numpy as np

from fidp.analysis.fitting.vector_fitting import (
    RationalModel,
    VectorFitConfig,
    vector_fit,
    log_frequency_grid,
)
from fidp.analysis.fitting.passivity.tests import check_passivity
from fidp.analysis.fitting.passivity.enforce_residue_nnls import enforce_passivity_nnls, NNLSPassivityConfig
from fidp.analysis.fitting.passivity.enforce_qp import enforce_passivity_qp, QPPassivityConfig
from fidp.analysis.fitting.fractional_fit import FractionalFitConfig, estimate_fractional_order
from fidp.analysis.fitting.symbolic_regression import SymbolicRegressionConfig, symbolic_regression


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _lock_hash() -> str:
    lock_path = Path("requirements/requirements-cpu.lock")
    if not lock_path.exists():
        return "unknown"
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


def _artifact_dir(payload: dict) -> Path:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return Path("artifacts") / "step7" / f"demo_{digest}"


def _make_synthetic_rational() -> RationalModel:
    poles = np.array([-40.0 + 120.0j, -40.0 - 120.0j, -800.0], dtype=complex)
    residues = np.array([15.0 - 6.0j, 15.0 + 6.0j, 50.0], dtype=complex)
    return RationalModel(poles=poles, residues=residues, d=1.0 + 0.0j, h=0.0 + 0.0j, kind="impedance")


def _make_cpe(alpha: float, c_alpha: float, freqs: np.ndarray) -> np.ndarray:
    s = 1j * 2.0 * np.pi * freqs
    return 1.0 / (c_alpha * (s**alpha))


def run_demo(seed: int, enable_symbolic: bool) -> Path:
    rng = np.random.default_rng(seed)
    freq_hz = log_frequency_grid(1.0, 1e5, 200)

    true_model = _make_synthetic_rational()
    Z = true_model.eval_freq(freq_hz)
    noise = 1.0 + 1e-3 * rng.standard_normal(Z.shape)
    Z_noisy = Z * noise

    cfg = VectorFitConfig(n_poles=3, n_iters=18, init_pole_scale=1.0)
    fit_result = vector_fit(freq_hz, Z_noisy, kind="impedance", cfg=cfg)

    passivity_report = check_passivity(freq_hz, model=fit_result.model, tol=1e-9)

    violating = fit_result.model
    if passivity_report.margin > 0.0:
        violating = RationalModel(
            poles=fit_result.model.poles.copy(),
            residues=fit_result.model.residues.copy(),
            d=fit_result.model.d - (passivity_report.margin + 0.2),
            h=fit_result.model.h,
            kind=fit_result.model.kind,
            metadata=dict(fit_result.model.metadata),
        )

    nnls_model, nnls_report = enforce_passivity_nnls(violating, freq_hz, NNLSPassivityConfig(tol=1e-9))
    qp_model, qp_report = enforce_passivity_qp(violating, freq_hz, QPPassivityConfig(tol=1e-9))

    cpe_freq = log_frequency_grid(10.0, 1e4, 160)
    alpha_true = 0.62
    c_alpha_true = 2.2e-3
    cpe = _make_cpe(alpha_true, c_alpha_true, cpe_freq)
    cpe_noise = 1.0 + 5e-3 * rng.standard_normal(cpe.shape)
    cpe_noisy = cpe * cpe_noise
    frac_cfg = FractionalFitConfig(seed=seed)
    frac_report = estimate_fractional_order(cpe_freq, cpe_noisy, frac_cfg)

    sym_result = None
    if enable_symbolic:
        x = cpe_freq
        y = 0.8 * np.log(cpe_freq) + 1.2
        sym_cfg = SymbolicRegressionConfig(seed=seed)
        sym_result = symbolic_regression(x, y, sym_cfg)
    else:
        sym_cfg = None

    payload = {
        "seed": seed,
        "vector_fit": cfg.__dict__,
        "fractional": frac_cfg.__dict__,
        "symbolic": None if sym_cfg is None else sym_cfg.__dict__,
    }
    out_dir = _artifact_dir(payload)
    if out_dir.exists():
        counter = 1
        while (out_dir.parent / f"{out_dir.name}_v{counter}").exists():
            counter += 1
        out_dir = out_dir.parent / f"{out_dir.name}_v{counter}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "seed": seed,
        "git_sha": _git_sha(),
        "lock_hash": _lock_hash(),
        "vector_fit": {
            "metrics": {
                "rel_error_rms": fit_result.rel_error_rms,
                "abs_error_rms": fit_result.abs_error_rms,
                "max_rel_error": fit_result.max_rel_error,
                "max_abs_error": fit_result.max_abs_error,
            },
            "diagnostics": fit_result.diagnostics,
        },
        "passivity": {
            "initial": passivity_report.to_dict(),
            "nnls": nnls_report.to_dict(),
            "qp": qp_report.to_dict(),
        },
        "fractional": frac_report.to_dict(),
    }

    if sym_result is not None:
        summary["symbolic_regression"] = {
            "expression": sym_result.expression,
            "parameters": sym_result.parameters,
            "score": sym_result.score,
            "confidence_intervals": sym_result.confidence_intervals,
            "rejected": sym_result.rejected,
            "warnings": sym_result.warnings,
            "diagnostics": sym_result.diagnostics,
        }

    (out_dir / "model_fit.json").write_text(
        json.dumps(fit_result.model.to_json_dict(), sort_keys=True, indent=2), encoding="utf-8"
    )
    (out_dir / "model_nnls.json").write_text(
        json.dumps(nnls_model.to_json_dict(), sort_keys=True, indent=2), encoding="utf-8"
    )
    (out_dir / "model_qp.json").write_text(
        json.dumps(qp_model.to_json_dict(), sort_keys=True, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8"
    )

    np.savez(
        out_dir / "responses.npz",
        freq_hz=freq_hz,
        Z_target=Z_noisy,
        Z_fit=fit_result.model.eval_freq(freq_hz),
        cpe_freq=cpe_freq,
        cpe=cpe_noisy,
    )

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model extraction demo pipeline.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--symbolic",
        action="store_true",
        help="Enable symbolic regression stage.",
    )
    args = parser.parse_args()

    out_dir = run_demo(args.seed, args.symbolic)
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
