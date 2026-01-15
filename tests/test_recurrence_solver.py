import numpy as np

from fidp.evaluators.recurrence import FixedPointImpedanceSolver


def analytic_infinite_ladder(s: complex, r: float, c: float) -> complex:
    a = s * c
    b = -s * c * r
    c0 = -r
    disc = b * b - 4.0 * a * c0
    sqrt_disc = np.sqrt(disc)
    z1 = (-b + sqrt_disc) / (2.0 * a)
    z2 = (-b - sqrt_disc) / (2.0 * a)
    return z1 if z1.real >= z2.real else z2


def test_infinite_ladder_converges_and_matches():
    r = 100.0
    c = 1e-6
    freqs = np.array([1.0, 10.0, 100.0, 1000.0])

    def recurrence(z: complex, s: complex) -> complex:
        return r + 1.0 / (s * c + 1.0 / z)

    solver = FixedPointImpedanceSolver(
        max_iter=200,
        tol=1e-8,
        residual_tol=1e-8,
        damping=0.6,
        anderson_m=3,
    )
    result = solver.solve(freqs, recurrence, initial_z=r, warm_start=True)

    expected = np.array([analytic_infinite_ladder(1j * 2.0 * np.pi * f, r, c) for f in freqs])

    assert result.converged.all()
    assert np.allclose(result.Z, expected, rtol=1e-4, atol=1e-6)
    assert (result.iterations > 0).all()
    assert (result.residual >= 0.0).all()
