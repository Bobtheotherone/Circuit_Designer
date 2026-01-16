import numpy as np

from fidp.search.botorch_mobo import propose_next_botorch, propose_next_random


def test_random_proposer_bounds_and_determinism() -> None:
    bounds = [[0.0, -1.0], [1.0, 2.0]]
    first = propose_next_random(bounds, batch_size=4, seed=123)
    second = propose_next_random(bounds, batch_size=4, seed=123)

    assert np.allclose(first, second)
    assert first.shape == (4, 2)
    assert np.all(first[:, 0] >= 0.0) and np.all(first[:, 0] <= 1.0)
    assert np.all(first[:, 1] >= -1.0) and np.all(first[:, 1] <= 2.0)


def test_botorch_proposer_bounds() -> None:
    bounds = [[0.0, 0.0], [1.0, 1.0]]
    X = np.array(
        [
            [0.1, 0.2],
            [0.4, 0.6],
            [0.8, 0.1],
        ],
        dtype=np.float64,
    )
    Y = np.array(
        [
            [1.0, 0.5],
            [0.8, 0.7],
            [0.2, 0.9],
        ],
        dtype=np.float64,
    )

    candidates = propose_next_botorch(bounds, X, Y, batch_size=2, seed=1)

    assert candidates.shape == (2, 2)
    assert np.all(candidates >= 0.0) and np.all(candidates <= 1.0)


def test_botorch_proposer_constraints() -> None:
    bounds = [[0.0, 0.0], [1.0, 1.0]]
    X = np.array(
        [
            [0.1, 0.2],
            [0.4, 0.6],
            [0.8, 0.1],
        ],
        dtype=np.float64,
    )
    Y = np.array(
        [
            [1.0, 0.5],
            [0.8, 0.7],
            [0.2, 0.9],
        ],
        dtype=np.float64,
    )

    constraints = [(np.array([1.0, 1.0], dtype=np.float64), 1.0)]
    candidates = propose_next_botorch(bounds, X, Y, batch_size=2, seed=2, constraints=constraints)

    assert np.all(candidates.sum(axis=1) <= 1.0 + 1e-8)
