"""Fixed-point solvers for self-similar impedance recurrences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List

import numpy as np


RecurrenceFn = Callable[[complex, complex], complex]


@dataclass
class RecurrenceResult:
    """Solver output with per-frequency diagnostics."""

    freqs_hz: np.ndarray
    Z: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    residual: np.ndarray
    contraction: np.ndarray
    method: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FixedPointImpedanceSolver:
    """Fixed-point solver with optional Anderson mixing and damping."""

    max_iter: int = 200
    tol: float = 1e-8
    residual_tol: float = 1e-8
    damping: float = 0.6
    anderson_m: int = 0
    anderson_start: int = 5
    stall_iter: int = 4

    def solve(
        self,
        freqs_hz: np.ndarray,
        recurrence: RecurrenceFn,
        initial_z: complex | None = None,
        warm_start: bool = True,
    ) -> RecurrenceResult:
        freqs_hz = np.asarray(freqs_hz, dtype=float)
        Z = np.zeros_like(freqs_hz, dtype=complex)
        converged = np.zeros_like(freqs_hz, dtype=bool)
        iterations = np.zeros_like(freqs_hz, dtype=int)
        residual = np.zeros_like(freqs_hz, dtype=float)
        contraction = np.full_like(freqs_hz, np.nan, dtype=float)
        method = np.empty_like(freqs_hz, dtype=object)

        prev_z = initial_z
        for idx, freq in enumerate(freqs_hz):
            s = 1j * 2.0 * np.pi * freq
            z = prev_z if (warm_start and prev_z is not None) else (initial_z or 1.0 + 0j)
            if warm_start and prev_z is not None and initial_z is not None:
                res_prev = abs(recurrence(prev_z, s) - prev_z)
                res_init = abs(recurrence(initial_z, s) - initial_z)
                if res_init < res_prev:
                    z = initial_z
            history_z: List[complex] = []
            history_g: List[complex] = []
            last_residual = None
            use_anderson = False
            restarted = False
            for it in range(1, self.max_iter + 1):
                g = recurrence(z, s)
                history_z.append(z)
                history_g.append(g)
                if self.anderson_m > 0 and use_anderson and len(history_z) >= 2:
                    z_accel = _anderson_mix(history_z, history_g, self.anderson_m)
                    z_new = (1.0 - self.damping) * z + self.damping * z_accel
                else:
                    z_new = (1.0 - self.damping) * z + self.damping * g
                if z_new.real < 0.0 and g.real >= 0.0:
                    z_new = g
                delta = abs(z_new - z)
                res = abs(g - z)
                if len(history_z) >= 2:
                    dz = abs(history_z[-1] - history_z[-2])
                    dg = abs(history_g[-1] - history_g[-2])
                    if dz > 0.0:
                        contraction[idx] = min(1e6, dg / dz)

                if (
                    warm_start
                    and not restarted
                    and initial_z is not None
                    and it >= self.stall_iter
                    and last_residual is not None
                    and res > last_residual * 0.99
                ):
                    z = initial_z
                    history_z.clear()
                    history_g.clear()
                    last_residual = None
                    use_anderson = False
                    restarted = True
                    continue

                if delta <= self.tol * (1.0 + abs(z_new)) or res <= self.residual_tol:
                    Z[idx] = z_new
                    converged[idx] = True
                    iterations[idx] = it
                    residual[idx] = res
                    method[idx] = "anderson" if use_anderson else "fixed_point"
                    prev_z = z_new
                    break
                if last_residual is not None and it >= self.stall_iter:
                    if res >= last_residual * 0.98:
                        use_anderson = self.anderson_m > 0 and it >= self.anderson_start
                last_residual = res
                z = z_new
                if it == self.max_iter:
                    Z[idx] = z
                    iterations[idx] = it
                    residual[idx] = res
                    method[idx] = "anderson" if use_anderson else "fixed_point"
                    prev_z = z
            else:
                Z[idx] = z
                method[idx] = "fixed_point"

        meta = {
            "max_iter": self.max_iter,
            "tol": self.tol,
            "residual_tol": self.residual_tol,
            "damping": self.damping,
            "anderson_m": self.anderson_m,
            "anderson_start": self.anderson_start,
            "stall_iter": self.stall_iter,
        }
        return RecurrenceResult(
            freqs_hz=freqs_hz,
            Z=Z,
            converged=converged,
            iterations=iterations,
            residual=residual,
            contraction=contraction,
            method=method,
            meta=meta,
        )


def _anderson_mix(history_z: List[complex], history_g: List[complex], m: int) -> complex:
    """Anderson mixing for scalar complex fixed-point iterations."""
    m = min(m, len(history_z))
    z_hist = np.array(history_z[-m:], dtype=complex)
    g_hist = np.array(history_g[-m:], dtype=complex)
    f_hist = g_hist - z_hist

    # Build real-valued least squares for stability.
    F = np.vstack([f_hist.real, f_hist.imag])
    FtF = F.T @ F
    ones = np.ones((m, 1))
    lhs = np.block([[FtF, ones], [ones.T, np.zeros((1, 1))]])
    rhs = np.zeros((m + 1, 1))
    rhs[-1, 0] = 1.0

    try:
        sol = np.linalg.solve(lhs, rhs)
        coeffs = sol[:-1, 0]
        z_new = np.sum(coeffs * g_hist)
        return z_new
    except np.linalg.LinAlgError:
        return g_hist[-1]
