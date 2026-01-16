import numpy as np
import pytest

from fidp.errors import CircuitValidationError
from fidp.search.features import (
    COMPONENT_TYPES,
    CircuitGraph,
    Component,
    FeatureConfig,
    extract_graph_features,
)


def _feature_dim(config: FeatureConfig) -> int:
    dim = len(COMPONENT_TYPES) + 1
    if config.include_depth:
        dim += 1
    if config.include_distance_to_port:
        dim += 1
    return dim


def test_extract_graph_features_shapes_and_values() -> None:
    components = [
        Component(kind="R", node_a="1", node_b="0", value=100.0, depth=1),
        Component(kind="C", node_a="1", node_b="2", value=1e-6, depth=2),
        Component(kind="PORT", node_a="1", node_b="0", value=None),
    ]
    graph = CircuitGraph(components=components, ports=[])
    config = FeatureConfig(include_depth=True, include_distance_to_port=True, include_edge_features=True)

    graph_tensors = extract_graph_features(graph, config=config, global_features=[1.0, 2.0])

    assert graph_tensors.node_features.shape == (
        len(components),
        _feature_dim(config),
    )
    assert graph_tensors.edge_index.shape[0] == 2
    assert graph_tensors.edge_features is not None
    assert graph_tensors.edge_features.shape[1] == 1
    assert graph_tensors.global_features is not None
    assert graph_tensors.global_features.shape == (2,)

    log_value = graph_tensors.node_features[0, len(COMPONENT_TYPES)]
    assert log_value == pytest.approx(np.log10(100.0))


def test_missing_port_raises() -> None:
    components = [Component(kind="R", node_a="1", node_b="0", value=10.0)]
    with pytest.raises(CircuitValidationError):
        CircuitGraph(components=components, ports=[])


def test_negative_value_raises() -> None:
    with pytest.raises(CircuitValidationError):
        Component(kind="R", node_a="1", node_b="0", value=-1.0)
