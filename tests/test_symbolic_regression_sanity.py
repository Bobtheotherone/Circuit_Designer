import numpy as np

from fidp.analysis.fitting.symbolic_regression import (
    SymbolicRegressionConfig,
    symbolic_regression,
)


def test_symbolic_regression_recovers_log_law():
    rng = np.random.default_rng(42)
    x = np.logspace(1, 3, 60)
    a_true = 0.9
    b_true = -1.2
    y = a_true + b_true * np.log(x)
    y_noisy = y + 0.02 * rng.standard_normal(y.shape)

    cfg = SymbolicRegressionConfig(seed=3, bootstrap_samples=120, min_improvement=0.05)
    result = symbolic_regression(x, y_noisy, cfg)

    assert result.rejected is False
    assert result.expression in {"a + b*log(x)", "a + b*log(x) + c*log(x)^2"}
    assert "b" in result.parameters
    assert result.confidence_intervals["b"][0] <= b_true <= result.confidence_intervals["b"][1]


def test_symbolic_regression_rejects_noise():
    rng = np.random.default_rng(7)
    x = np.logspace(1, 2, 50)
    y = rng.standard_normal(x.shape)

    cfg = SymbolicRegressionConfig(seed=4, bootstrap_samples=80, min_improvement=0.3)
    result = symbolic_regression(x, y, cfg)

    assert result.rejected is True
    assert result.expression is None
