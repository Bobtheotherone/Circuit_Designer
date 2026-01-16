import numpy as np
import torch

from fidp.search.dataset import CircuitSample
from fidp.search.features import COMPONENT_TYPES, CircuitGraph, Component, FeatureConfig
from fidp.search.surrogate import SurrogateConfig
from fidp.search.train_surrogate import TrainingConfig, train_surrogate


def _feature_dim(config: FeatureConfig) -> int:
    dim = len(COMPONENT_TYPES) + 1
    if config.include_depth:
        dim += 1
    if config.include_distance_to_port:
        dim += 1
    return dim


def _make_sample(scale: float) -> CircuitSample:
    components = [
        Component(kind="R", node_a="1", node_b="0", value=10.0 * scale, depth=1),
        Component(kind="C", node_a="1", node_b="2", value=1e-6 * scale, depth=2),
        Component(kind="PORT", node_a="1", node_b="0"),
    ]
    graph = CircuitGraph(components=components, ports=[])
    targets = np.array([scale, scale + 0.5], dtype=np.float32)
    return CircuitSample(graph=graph, targets=targets)


def test_train_surrogate_smoke() -> None:
    feature_config = FeatureConfig()
    samples = [_make_sample(1.0), _make_sample(2.0), _make_sample(3.0)]
    surrogate_config = SurrogateConfig(
        node_feature_dim=_feature_dim(feature_config),
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        output_dim=2,
    )
    training_config = TrainingConfig(epochs=2, batch_size=2, seed=0, patience=1)

    result = train_surrogate(samples, surrogate_config, feature_config, training_config)

    assert result.loss_history
    assert result.best_loss <= max(result.loss_history)
    assert result.model is not None
