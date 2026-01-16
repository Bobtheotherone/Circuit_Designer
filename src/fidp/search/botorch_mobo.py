"""Multi-objective Bayesian optimization helpers."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import normalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from fidp.search.utils import set_seed


def _validate_bounds(bounds: Sequence[Sequence[float]]) -> np.ndarray:
    bounds_array = np.asarray(bounds, dtype=np.float64)
    if bounds_array.shape[0] != 2:
        raise ValueError("Bounds must have shape (2, d).")
    if bounds_array.ndim != 2:
        raise ValueError("Bounds must be 2D.")
    lower, upper = bounds_array
    if lower.shape != upper.shape:
        raise ValueError("Lower/upper bounds must match in shape.")
    if np.any(lower >= upper):
        raise ValueError("Each lower bound must be less than upper bound.")
    return bounds_array


def _validate_observations(X: np.ndarray, Y: np.ndarray, dim: int) -> None:
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")
    if X.shape[1] != dim:
        raise ValueError("X dimension must match bounds.")


def propose_next_random(
    bounds: Sequence[Sequence[float]],
    batch_size: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Fallback random proposer for multi-objective search."""
    bounds_array = _validate_bounds(bounds)
    rng = np.random.default_rng(seed)
    lower, upper = bounds_array
    samples = rng.random((batch_size, lower.shape[0]))
    return lower + samples * (upper - lower)


def propose_next_botorch(
    bounds: Sequence[Sequence[float]],
    X: Optional[np.ndarray],
    Y: Optional[np.ndarray],
    batch_size: int,
    seed: Optional[int] = None,
    objective_signs: Optional[Sequence[float]] = None,
    constraints: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
) -> np.ndarray:
    """Propose next batch using BoTorch qNEHVI."""
    bounds_array = _validate_bounds(bounds)
    dim = bounds_array.shape[1]

    if X is None or Y is None or len(X) == 0:
        return propose_next_random(bounds_array, batch_size, seed)

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    _validate_observations(X, Y, dim)

    signs = np.ones(Y.shape[1], dtype=np.float64)
    if objective_signs is not None:
        signs = np.asarray(objective_signs, dtype=np.float64)
        if signs.shape[0] != Y.shape[1]:
            raise ValueError("objective_signs must match number of objectives.")

    Y_signed = Y * signs

    set_seed(seed)

    train_X = torch.tensor(X, dtype=torch.double)
    train_Y = torch.tensor(Y_signed, dtype=torch.double)
    bounds_tensor = torch.tensor(bounds_array, dtype=torch.double)

    train_X = normalize(train_X, bounds_tensor)

    models = []
    for idx in range(train_Y.shape[1]):
        models.append(SingleTaskGP(train_X, train_Y[:, idx : idx + 1]))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    ref_point = train_Y.min(dim=0).values - 0.1 * train_Y.std(dim=0)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=seed)

    acquisition = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=train_X,
        sampler=sampler,
        prune_baseline=True,
    )

    candidates, _ = optimize_acqf(
        acq_function=acquisition,
        bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]),
        q=batch_size,
        num_restarts=5,
        raw_samples=64,
        options={"batch_limit": 5, "maxiter": 200},
    )

    candidates = torch.clamp(candidates, 0.0, 1.0)
    candidates = candidates * (bounds_tensor[1] - bounds_tensor[0]) + bounds_tensor[0]

    if constraints:
        candidates = _apply_constraints(candidates.detach().cpu().numpy(), constraints, bounds_array, batch_size, seed)
        return candidates

    return candidates.detach().cpu().numpy()


def _apply_constraints(
    candidates: np.ndarray,
    constraints: Sequence[Tuple[np.ndarray, float]],
    bounds_array: np.ndarray,
    batch_size: int,
    seed: Optional[int],
) -> np.ndarray:
    dim = bounds_array.shape[1]
    filtered = _filter_candidates(candidates, constraints, dim)
    if filtered.shape[0] >= batch_size:
        return filtered[:batch_size]

    rng = np.random.default_rng(seed)
    lower, upper = bounds_array
    attempts = 0
    extra: List[np.ndarray] = []
    while len(extra) + filtered.shape[0] < batch_size and attempts < 50:
        sample = rng.random((batch_size, dim)) * (upper - lower) + lower
        sample = _filter_candidates(sample, constraints, dim)
        if sample.size:
            extra.append(sample)
        attempts += 1

    if extra:
        filtered = np.vstack([filtered] + extra)
    if filtered.shape[0] < batch_size:
        raise ValueError("Unable to satisfy constraints with available candidates.")
    return filtered[:batch_size]


def _filter_candidates(
    candidates: np.ndarray,
    constraints: Sequence[Tuple[np.ndarray, float]],
    dim: int,
) -> np.ndarray:
    filtered = np.asarray(candidates, dtype=np.float64)
    for weights, threshold in constraints:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != dim:
            raise ValueError("Constraint weights must match candidate dimension.")
        mask = filtered @ weights <= threshold
        filtered = filtered[mask]
        if filtered.size == 0:
            break
    return filtered
