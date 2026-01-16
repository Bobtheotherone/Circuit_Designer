import numpy as np

from fidp.search.evolution import DesignRecord, EvolutionConfig, evolve_population
from fidp.search.features import CircuitGraph, Component


def _make_design(scale: float) -> DesignRecord:
    components = [
        Component(kind="R", node_a="1", node_b="0", value=1.0),
        Component(kind="PORT", node_a="1", node_b="0"),
    ]
    graph = CircuitGraph(components=components, ports=[])
    params = np.array([scale, scale + 0.5], dtype=np.float64)
    return DesignRecord(graph=graph, parameters=params)


def _evaluate(design: DesignRecord) -> np.ndarray:
    value = float(np.sum(design.parameters))
    return np.array([value, -value], dtype=np.float64)


def test_evolution_validity_and_determinism() -> None:
    population = [_make_design(1.0), _make_design(2.0)]
    bounds = (np.array([0.1, 0.1]), np.array([5.0, 5.0]))
    config = EvolutionConfig(
        population_size=4,
        elite_fraction=0.5,
        mutation_rate=1.0,
        crossover_rate=1.0,
        parameter_sigma=0.05,
        parameter_bounds=bounds,
    )

    result_a = evolve_population(population, _evaluate, config, seed=42)
    result_b = evolve_population(population, _evaluate, config, seed=42)

    params_a = np.stack([member.design.parameters for member in result_a.population])
    params_b = np.stack([member.design.parameters for member in result_b.population])

    assert params_a.shape == params_b.shape
    assert np.allclose(params_a, params_b)
    assert np.all(params_a > 0.0)
    assert np.all(params_a >= bounds[0]) and np.all(params_a <= bounds[1])
