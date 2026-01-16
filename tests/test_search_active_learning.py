import numpy as np
import torch

from fidp.search.active_learning import ActiveLearningLoop
from fidp.search.evolution import DesignRecord
from fidp.search.features import COMPONENT_TYPES, CircuitGraph, Component, FeatureConfig
from fidp.search.surrogate import SurrogateConfig
from fidp.search.train_surrogate import TrainingConfig


def _feature_dim(config: FeatureConfig) -> int:
    dim = len(COMPONENT_TYPES) + 1
    if config.include_depth:
        dim += 1
    if config.include_distance_to_port:
        dim += 1
    return dim


class DummyEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, design: DesignRecord) -> np.ndarray:
        self.calls += 1
        value = float(np.sum(design.parameters))
        return np.array([value, value + 1.0], dtype=np.float64)


def _make_design(scale: float) -> DesignRecord:
    components = [
        Component(kind="R", node_a="1", node_b="0", value=1.0),
        Component(kind="PORT", node_a="1", node_b="0"),
    ]
    graph = CircuitGraph(components=components, ports=[])
    params = np.array([scale, scale + 0.2], dtype=np.float64)
    return DesignRecord(graph=graph, parameters=params)


def test_active_learning_iteration_and_caching() -> None:
    evaluator = DummyEvaluator()
    feature_config = FeatureConfig()
    surrogate_config = SurrogateConfig(
        node_feature_dim=_feature_dim(feature_config),
        output_dim=2,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
    )
    training_config = TrainingConfig(epochs=1, batch_size=2, seed=0, patience=1)

    loop = ActiveLearningLoop(
        evaluator=evaluator,
        feature_config=feature_config,
        surrogate_config=surrogate_config,
        training_config=training_config,
    )

    initial_designs = [_make_design(1.0), _make_design(2.0)]
    loop.add_designs(initial_designs)
    assert evaluator.calls == 2

    loop.add_designs(initial_designs)
    assert evaluator.calls == 2

    result = loop.run_iteration(
        initial_designs=initial_designs,
        bo_bounds=[[0.1, 0.1], [3.0, 3.0]],
        batch_size=2,
        seed=1,
    )

    assert result.dataset_size >= 2
    assert evaluator.calls >= 2
    assert len(result.new_designs) <= 2
