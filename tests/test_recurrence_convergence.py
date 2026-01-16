import numpy as np

from fidp.dsl.generators.ladder import domino_ladder
from fidp.evaluators.recurrence import FixedPointImpedanceSolver, RecurrenceEvaluator, RecurrenceOptions
from fidp.evaluators.types import EvalRequest, FrequencyGridSpec


def _analytic_infinite_ladder(s: complex, r: float, c: float) -> complex:
    a = s * c
    b = -s * c * r
    c0 = -r
    disc = b * b - 4.0 * a * c0
    sqrt_disc = np.sqrt(disc)
    z1 = (-b + sqrt_disc) / (2.0 * a)
    z2 = (-b - sqrt_disc) / (2.0 * a)
    return z1 if z1.real >= z2.real else z2


def test_recurrence_converges_on_domino() -> None:
    circuit = domino_ladder(stages=6, r_value=100.0, c_value=1e-6)
    grid = FrequencyGridSpec(f_start_hz=1.0, f_stop_hz=1000.0, points=6, spacing="linear")
    request = EvalRequest(grid=grid, fidelity="fast")
    evaluator = RecurrenceEvaluator()

    result = evaluator.evaluate(circuit, request)

    freqs = grid.make_grid()
    expected = np.array([_analytic_infinite_ladder(1j * 2.0 * np.pi * f, 100.0, 1e-6) for f in freqs])
    assert result.status == "ok"
    assert np.allclose(result.Z, expected, rtol=1e-3, atol=1e-6)


def test_recurrence_nonconvergence_returns_error() -> None:
    circuit = domino_ladder(stages=3, r_value=100.0, c_value=1e-6)
    grid = FrequencyGridSpec(f_start_hz=1.0, f_stop_hz=100.0, points=4, spacing="linear")
    request = EvalRequest(grid=grid, fidelity="fast")
    options = RecurrenceOptions(max_iter=1, anderson_m=0)
    evaluator = RecurrenceEvaluator(options)

    result = evaluator.evaluate(circuit, request)

    assert result.status == "error"
    assert any(err.code == "recurrence_nonconverged" for err in result.errors)


def test_anderson_fallback_triggers() -> None:
    freqs = np.array([10.0])

    def recurrence(z: complex, s: complex) -> complex:
        _ = s
        return z + 1.0

    solver = FixedPointImpedanceSolver(
        max_iter=6,
        tol=1e-10,
        residual_tol=1e-10,
        damping=0.9,
        anderson_m=2,
        anderson_start=2,
        stall_iter=1,
    )
    result = solver.solve(freqs, recurrence, initial_z=2.0 + 0j, warm_start=False)

    assert result.method[0] == "anderson"
