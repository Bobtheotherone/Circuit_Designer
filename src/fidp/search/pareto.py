"""Pareto ranking utilities for multi-objective optimization."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def dominates(
    obj_a: np.ndarray,
    obj_b: np.ndarray,
    objective_signs: Optional[Sequence[float]] = None,
) -> bool:
    """Return True if obj_a dominates obj_b."""
    a = np.asarray(obj_a, dtype=np.float64)
    b = np.asarray(obj_b, dtype=np.float64)
    if objective_signs is not None:
        signs = np.asarray(objective_signs, dtype=np.float64)
        a = a * signs
        b = b * signs
    return np.all(a >= b) and np.any(a > b)


def non_dominated_sort(
    objectives: np.ndarray,
    objective_signs: Optional[Sequence[float]] = None,
) -> List[List[int]]:
    """Perform non-dominated sorting and return Pareto fronts."""
    objs = np.asarray(objectives, dtype=np.float64)
    population_size = objs.shape[0]
    dominates_list: List[List[int]] = [[] for _ in range(population_size)]
    dominated_count = np.zeros(population_size, dtype=int)
    fronts: List[List[int]] = [[]]

    for i in range(population_size):
        for j in range(population_size):
            if i == j:
                continue
            if dominates(objs[i], objs[j], objective_signs):
                dominates_list[i].append(j)
            elif dominates(objs[j], objs[i], objective_signs):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front: List[int] = []
        for i in fronts[current]:
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(
    objectives: np.ndarray,
    front: Iterable[int],
    objective_signs: Optional[Sequence[float]] = None,
) -> Dict[int, float]:
    """Compute crowding distance for a front."""
    objs = np.asarray(objectives, dtype=np.float64)
    indices = list(front)
    if not indices:
        return {}

    distance = {idx: 0.0 for idx in indices}

    values = objs[indices]
    if objective_signs is not None:
        values = values * np.asarray(objective_signs, dtype=np.float64)

    num_objectives = values.shape[1]
    for m in range(num_objectives):
        order = np.argsort(values[:, m])
        sorted_indices = [indices[i] for i in order]
        distance[sorted_indices[0]] = float("inf")
        distance[sorted_indices[-1]] = float("inf")
        min_val = values[order[0], m]
        max_val = values[order[-1], m]
        denom = max_val - min_val
        if denom == 0:
            denom = 1.0
        for i in range(1, len(sorted_indices) - 1):
            prev_val = values[order[i - 1], m]
            next_val = values[order[i + 1], m]
            distance[sorted_indices[i]] += (next_val - prev_val) / denom

    return distance


def pareto_rank(
    objectives: np.ndarray,
    objective_signs: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Return rank array and Pareto fronts."""
    fronts = non_dominated_sort(objectives, objective_signs)
    ranks = np.zeros(len(objectives), dtype=int)
    for rank, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank
    return ranks, fronts
