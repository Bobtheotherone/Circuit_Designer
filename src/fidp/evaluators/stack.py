"""Evaluator stack orchestrating multi-fidelity impedance evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Protocol
import hashlib
import json
import subprocess
import time

import numpy as np

from fidp.circuits import CircuitIR
from fidp.circuits.canonical import canonicalize_circuit
from fidp.dsl.generators.ladder import domino_ladder
from fidp.evaluators.mna import MnaEvaluator
from fidp.evaluators.mor import PrimaEvaluator, MorOptions
from fidp.evaluators.passivity import check_impedance_passivity
from fidp.evaluators.recurrence import RecurrenceEvaluator, RecurrenceOptions
from fidp.evaluators.spice import SpiceEvaluator, SpiceOptions
from fidp.evaluators.types import EvalError, EvalRequest, EvalResult, FrequencyGridSpec, summarize_errors


class Evaluator(Protocol):
    """Protocol for evaluators in the stack."""

    def evaluate(self, circuit: CircuitIR, request: EvalRequest) -> EvalResult:
        ...


@dataclass
class EvaluatorStack:
    """Stacked evaluator orchestrating recurrence, MNA, MOR, and SPICE."""

    recurrence: RecurrenceEvaluator = field(default_factory=RecurrenceEvaluator)
    mna: MnaEvaluator = field(default_factory=MnaEvaluator)
    spice: SpiceEvaluator = field(default_factory=SpiceEvaluator)
    mor: PrimaEvaluator = field(default_factory=PrimaEvaluator)

    def evaluate(self, circuit: CircuitIR, request: EvalRequest) -> EvalResult:
        start = time.perf_counter()
        result: EvalResult
        if request.fidelity == "fast":
            result = self.recurrence.evaluate(circuit, request)
            if _is_not_applicable(result):
                result = self.mna.evaluate(circuit, request)
        elif request.fidelity == "mid":
            result = self.mna.evaluate(circuit, request)
        elif request.fidelity == "truth":
            result = self.spice.evaluate(circuit, request)
        elif request.fidelity == "mor":
            result = self.mor.evaluate(circuit, request)
        else:
            result = _error_result(
                request.grid.make_grid(),
                "invalid_fidelity",
                f"Unknown fidelity: {request.fidelity}",
                {},
            )

        if result.status == "ok":
            passivity = check_impedance_passivity(result.freqs_hz, result.Z)
            result.passivity = {
                "is_passive": passivity.is_passive,
                "min_eig": passivity.min_eig,
                "worst_freq_hz": passivity.worst_freq_hz,
                "n_violations": passivity.n_violations,
            }
            if not passivity.is_passive:
                result.status = "error"
                result.errors.append(
                    EvalError(
                        code="passivity_violation",
                        message="Impedance violates passivity on the evaluation grid.",
                        details={
                            "min_eig": passivity.min_eig,
                            "worst_freq_hz": passivity.worst_freq_hz,
                        },
                    )
                )
            else:
                _apply_grid_refinement_check(circuit, request, result, self)

        provenance = _collect_provenance()
        provenance["seed"] = str(request.seed)
        provenance["fidelity"] = request.fidelity
        provenance["evaluators"] = {
            "recurrence": type(self.recurrence).__name__,
            "mna": type(self.mna).__name__,
            "spice": type(self.spice).__name__,
            "mor": type(self.mor).__name__,
        }
        result.meta.setdefault("provenance", provenance)
        result.timing_s.setdefault("stack_total", time.perf_counter() - start)
        result.meta.setdefault("fidelity", request.fidelity)
        return result


def _is_not_applicable(result: EvalResult) -> bool:
    return any(err.code == "recurrence_not_applicable" for err in result.errors)


def _collect_provenance() -> dict[str, str]:
    git_sha = "unknown"
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_sha = completed.stdout.strip()
    except Exception:
        pass

    lock_hash = "unknown"
    lock_path = Path("requirements/requirements-cpu.lock")
    if lock_path.exists():
        lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()

    return {"git_sha": git_sha, "lock_hash": lock_hash}


def _error_result(freqs: np.ndarray, code: str, message: str, details: dict[str, object]) -> EvalResult:
    Z = np.full_like(freqs, np.nan, dtype=complex)
    return EvalResult(
        freqs_hz=freqs,
        Z=Z,
        status="error",
        errors=[EvalError(code=code, message=message, details=details)],
    )


def _write_eval_artifact(out_dir: Path, label: str, result: EvalResult) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_json_dict()
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    path = out_dir / f"{label}_{digest}.json"
    if path.exists():
        return path
    path.write_bytes(content)
    return path


def _apply_grid_refinement_check(
    circuit: CircuitIR,
    request: EvalRequest,
    result: EvalResult,
    stack: EvaluatorStack,
) -> None:
    sanity = result.meta.setdefault("sanity_check", {})
    if request.metadata.get("grid_refinement", True) is False:
        sanity["grid_refinement"] = {"status": "skipped", "reason": "disabled"}
        return
    if request.fidelity == "truth":
        sanity["grid_refinement"] = {"status": "skipped", "reason": "truth_fidelity"}
        return
    if request.grid.points < 3:
        sanity["grid_refinement"] = {"status": "skipped", "reason": "insufficient_points"}
        return

    refined_grid = FrequencyGridSpec(
        f_start_hz=request.grid.f_start_hz,
        f_stop_hz=request.grid.f_stop_hz,
        points=2 * request.grid.points - 1,
        spacing=request.grid.spacing,
        include_dc=request.grid.include_dc,
    )
    ref_request = replace(request, grid=refined_grid)

    if request.fidelity == "fast":
        if "family" in result.meta:
            evaluator = stack.recurrence
        else:
            evaluator = stack.mna
    elif request.fidelity == "mid":
        evaluator = stack.mna
    elif request.fidelity == "mor":
        evaluator = stack.mor
    else:
        sanity["grid_refinement"] = {"status": "skipped", "reason": "unsupported_fidelity"}
        return

    ref_result = evaluator.evaluate(circuit, ref_request)
    if ref_result.status != "ok":
        sanity["grid_refinement"] = {
            "status": "skipped",
            "reason": "refined_eval_failed",
            "error": summarize_errors(ref_result.errors),
        }
        result.warnings.append("Grid refinement check skipped due to refined evaluation failure.")
        return

    coarse_freqs = request.grid.make_grid()
    refined_freqs = refined_grid.make_grid()
    if request.grid.include_dc:
        coarse_eval = coarse_freqs[1:]
        refined_eval = refined_freqs[1:]
        offset = 1
    else:
        coarse_eval = coarse_freqs
        refined_eval = refined_freqs
        offset = 0

    if coarse_eval.size < 2:
        sanity["grid_refinement"] = {"status": "skipped", "reason": "insufficient_points"}
        return

    expected_refined = 2 * coarse_eval.size - 1
    if refined_eval.size != expected_refined:
        sanity["grid_refinement"] = {
            "status": "skipped",
            "reason": "grid_mismatch",
            "expected_points": int(expected_refined),
            "actual_points": int(refined_eval.size),
        }
        result.warnings.append("Grid refinement check skipped due to grid mismatch.")
        return

    coarse_indices = np.arange(coarse_eval.size) * 2
    mid_indices = coarse_indices[:-1] + 1
    left_indices = coarse_indices[:-1] + offset
    right_indices = coarse_indices[1:] + offset
    mid_full_indices = mid_indices + offset

    Z_left = result.Z[left_indices]
    Z_right = result.Z[right_indices]
    Z_mid = ref_result.Z[mid_full_indices]

    if request.grid.spacing == "log":
        x_left = np.log(coarse_eval[:-1])
        x_right = np.log(coarse_eval[1:])
        x_mid = np.log(refined_eval[mid_indices])
    else:
        x_left = coarse_eval[:-1]
        x_right = coarse_eval[1:]
        x_mid = refined_eval[mid_indices]

    denom = x_right - x_left
    if np.any(denom <= 0):
        sanity["grid_refinement"] = {"status": "skipped", "reason": "non_monotonic_grid"}
        result.warnings.append("Grid refinement check skipped due to non-monotonic grid.")
        return

    t = (x_mid - x_left) / denom
    t = t.reshape((t.size,) + (1,) * (Z_mid.ndim - 1))
    Z_interp = (1.0 - t) * Z_left + t * Z_right
    diff = Z_mid - Z_interp
    abs_diff = np.abs(diff)
    abs_mid = np.abs(Z_mid)
    rel_err = abs_diff / np.maximum(abs_mid, 1e-12)

    max_rel = float(np.max(rel_err))
    max_abs = float(np.max(abs_diff))
    rel_tol = float(request.metadata.get("grid_refinement_rel_tol", 0.1))
    status = "ok" if max_rel <= rel_tol else "warn"
    sanity["grid_refinement"] = {
        "status": status,
        "max_rel_err": max_rel,
        "max_abs_err": max_abs,
        "rel_tol": rel_tol,
        "refined_points": refined_grid.points,
    }
    if status == "warn":
        result.warnings.append(
            f"Grid refinement check exceeded tolerance: max_rel_err={max_rel:.3g}, rel_tol={rel_tol:.3g}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FIDP evaluator stack demo.")
    parser.add_argument("--demo", action="store_true", help="Run the demo evaluation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/eval_demo"))
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return

    circuit = domino_ladder(stages=6, r_value=100.0, c_value=1e-6, seed=args.seed)
    grid = FrequencyGridSpec(f_start_hz=10.0, f_stop_hz=1e5, points=40)
    stack = EvaluatorStack()

    results = {}
    for fidelity in ("fast", "mid", "truth", "mor"):
        request = EvalRequest(grid=grid, fidelity=fidelity, seed=args.seed)
        results[fidelity] = stack.evaluate(circuit, request)

    canonical = canonicalize_circuit(circuit, mode="snapped")
    base_dir = args.out_dir / canonical.canonical_hash
    for fidelity, result in results.items():
        path = _write_eval_artifact(base_dir, fidelity, result)
        status = result.status
        print(f"{fidelity}: {status} -> {path}")


if __name__ == "__main__":
    main()
