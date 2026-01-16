"""Evolutionary search operators for circuit designs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

from fidp.search.features import CircuitGraph, validate_circuit_graph
from fidp.search.pareto import crowding_distance, pareto_rank

if TYPE_CHECKING:
    from fidp.metrics.novelty import NoveltyMetrics


@dataclass(frozen=True)
class DesignRecord:
    """Container for a candidate circuit design."""

    graph: CircuitGraph
    parameters: np.ndarray
    spec: Optional[str] = None
    global_features: Optional[np.ndarray] = None
    novelty: Optional["NoveltyMetrics"] = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PopulationMember:
    """Population member with associated objectives."""

    design: DesignRecord
    objectives: np.ndarray


@dataclass(frozen=True)
class EvolutionConfig:
    """Configuration for evolutionary search."""

    population_size: int = 16
    elite_fraction: float = 0.25
    mutation_rate: float = 0.4
    crossover_rate: float = 0.7
    parameter_sigma: float = 0.1
    parameter_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    objective_signs: Optional[Sequence[float]] = None
    mutate_graph_fn: Optional[Callable[[CircuitGraph, np.random.Generator], CircuitGraph]] = None
    crossover_graph_fn: Optional[
        Callable[[CircuitGraph, CircuitGraph, np.random.Generator], CircuitGraph]
    ] = None
    validate_graph_fn: Optional[Callable[[CircuitGraph], None]] = None
    snap_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    max_attempts: int = 25


@dataclass
class EvolutionResult:
    """Result of a population evolution step."""

    population: List[PopulationMember]
    fronts: List[List[int]]
    ranks: np.ndarray


def evolve_population(
    population: Sequence[Union[DesignRecord, PopulationMember]],
    evaluate_fn: Callable[[DesignRecord], np.ndarray],
    config: EvolutionConfig,
    seed: Optional[int] = None,
) -> EvolutionResult:
    """Evolve a population using NSGA-II style selection."""
    rng = np.random.default_rng(seed)

    members = _ensure_population_members(population, evaluate_fn)
    if len(members) < 2:
        raise ValueError("Population must contain at least two members.")

    parent_objectives = np.stack([member.objectives for member in members], axis=0)
    parent_ranks, parent_fronts = pareto_rank(parent_objectives, config.objective_signs)
    parent_crowding = _crowding_map(parent_objectives, parent_fronts, config.objective_signs)

    offspring: List[PopulationMember] = []
    while len(offspring) < config.population_size:
        parent_a = _tournament_select(members, parent_ranks, parent_crowding, rng)
        parent_b = _tournament_select(members, parent_ranks, parent_crowding, rng)
        child = _crossover_design(parent_a.design, parent_b.design, config, rng)
        child = _mutate_design(child, config, rng)
        _validate_design(child, config)
        objectives = np.asarray(evaluate_fn(child), dtype=np.float64)
        offspring.append(PopulationMember(design=child, objectives=objectives))

    combined = members + offspring
    combined_objectives = np.stack([member.objectives for member in combined], axis=0)
    ranks, fronts = pareto_rank(combined_objectives, config.objective_signs)
    crowding = _crowding_map(combined_objectives, fronts, config.objective_signs)

    next_population = _select_next_generation(combined, fronts, crowding, config.population_size)

    return EvolutionResult(population=next_population, fronts=fronts, ranks=ranks)


def _ensure_population_members(
    population: Sequence[Union[DesignRecord, PopulationMember]],
    evaluate_fn: Callable[[DesignRecord], np.ndarray],
) -> List[PopulationMember]:
    members: List[PopulationMember] = []
    for item in population:
        if isinstance(item, PopulationMember):
            members.append(item)
        else:
            objectives = np.asarray(evaluate_fn(item), dtype=np.float64)
            members.append(PopulationMember(design=item, objectives=objectives))
    return members


def _tournament_select(
    members: Sequence[PopulationMember],
    ranks: np.ndarray,
    crowding: dict,
    rng: np.random.Generator,
) -> PopulationMember:
    idx_a, idx_b = rng.integers(0, len(members), size=2)
    if ranks[idx_a] < ranks[idx_b]:
        return members[idx_a]
    if ranks[idx_b] < ranks[idx_a]:
        return members[idx_b]
    if crowding.get(idx_a, 0.0) > crowding.get(idx_b, 0.0):
        return members[idx_a]
    if crowding.get(idx_b, 0.0) > crowding.get(idx_a, 0.0):
        return members[idx_b]
    return members[idx_a]


def _crowding_map(
    objectives: np.ndarray,
    fronts: Sequence[Sequence[int]],
    objective_signs: Optional[Sequence[float]],
) -> dict:
    crowding: dict = {}
    for front in fronts:
        crowding.update(crowding_distance(objectives, front, objective_signs))
    return crowding


def _select_next_generation(
    population: Sequence[PopulationMember],
    fronts: Sequence[Sequence[int]],
    crowding: dict,
    population_size: int,
) -> List[PopulationMember]:
    next_population: List[PopulationMember] = []
    for front in fronts:
        if len(next_population) + len(front) <= population_size:
            next_population.extend(population[idx] for idx in front)
        else:
            sorted_front = sorted(front, key=lambda idx: crowding.get(idx, 0.0), reverse=True)
            remaining = population_size - len(next_population)
            next_population.extend(population[idx] for idx in sorted_front[:remaining])
            break
    return next_population


def _mutate_design(
    design: DesignRecord,
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> DesignRecord:
    params = np.array(design.parameters, dtype=np.float64)
    if rng.random() < config.mutation_rate:
        params = _mutate_parameters(params, config, rng)
    if config.snap_fn is not None:
        params = np.asarray(config.snap_fn(params), dtype=np.float64)
    graph = design.graph
    if config.mutate_graph_fn is not None:
        graph = config.mutate_graph_fn(graph, rng)
    return DesignRecord(
        graph=graph,
        parameters=params,
        spec=design.spec,
        global_features=None,
        novelty=None,
        metadata=dict(design.metadata),
    )


def _crossover_design(
    parent_a: DesignRecord,
    parent_b: DesignRecord,
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> DesignRecord:
    if rng.random() >= config.crossover_rate:
        return parent_a
    params = _crossover_parameters(parent_a.parameters, parent_b.parameters, rng)
    graph = parent_a.graph
    if config.crossover_graph_fn is not None:
        graph = config.crossover_graph_fn(parent_a.graph, parent_b.graph, rng)
    return DesignRecord(
        graph=graph,
        parameters=params,
        spec=parent_a.spec,
        global_features=None,
        novelty=None,
        metadata=dict(parent_a.metadata),
    )


def _mutate_parameters(
    params: np.ndarray,
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.normal(scale=config.parameter_sigma, size=params.shape)
    mutated = params + noise
    if config.parameter_bounds is not None:
        lower, upper = config.parameter_bounds
        mutated = np.clip(mutated, lower, upper)
    mutated = np.maximum(mutated, 1e-12)
    return mutated


def _crossover_parameters(
    params_a: np.ndarray,
    params_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    params_a = np.asarray(params_a, dtype=np.float64)
    params_b = np.asarray(params_b, dtype=np.float64)
    mask = rng.random(params_a.shape) < 0.5
    offspring = params_a.copy()
    offspring[mask] = params_b[mask]
    return offspring


def _validate_design(design: DesignRecord, config: EvolutionConfig) -> None:
    if config.validate_graph_fn is not None:
        config.validate_graph_fn(design.graph)
    else:
        validate_circuit_graph(design.graph)
    params = np.asarray(design.parameters, dtype=np.float64)
    if np.any(params <= 0):
        raise ValueError("Design parameters must be positive.")
    if config.parameter_bounds is not None:
        lower, upper = config.parameter_bounds
        if np.any(params < lower) or np.any(params > upper):
            raise ValueError("Design parameters exceed provided bounds.")
