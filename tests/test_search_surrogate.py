import numpy as np
import pytest

import fidp.search.features as features
from fidp.search.dataset import CircuitDataset, CircuitSample, collate_fn
from fidp.search.surrogate import GraphSurrogateModel, SurrogateConfig
from fidp.search.utils import set_seed


torch = pytest.importorskip("torch")


def _make_sample(scale: float) -> CircuitSample:
    components = [
        features.Component(kind="R", node_a="1", node_b="0", value=1.0 * scale, depth=1),
        features.Component(kind="L", node_a="1", node_b="2", value=1e-3 * scale, depth=2),
        features.Component(kind="PORT", node_a="1", node_b="0"),
    ]
    graph = features.CircuitGraph(components=components, ports=[])
    targets = np.array([scale, scale + 1.0], dtype=np.float32)
    return CircuitSample(graph=graph, targets=targets)


def test_surrogate_forward_shape_and_determinism() -> None:
    config = features.FeatureConfig(include_edge_features=True)
    samples = [_make_sample(1.0), _make_sample(2.0)]
    dataset = CircuitDataset(samples=samples, config=config)
    batch = collate_fn([dataset[0], dataset[1]])

    model_config = SurrogateConfig(
        node_feature_dim=batch.node_features.shape[-1],
        edge_feature_dim=batch.edge_features.shape[-1] if batch.edge_features is not None else 0,
        global_feature_dim=0,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        output_dim=2,
    )

    set_seed(123)
    model_a = GraphSurrogateModel(model_config)
    out_a1 = model_a(batch)
    out_a2 = model_a(batch)

    set_seed(123)
    model_b = GraphSurrogateModel(model_config)
    out_b = model_b(batch)

    assert out_a1.shape == (2, 2)
    assert torch.allclose(out_a1, out_a2)
    assert torch.allclose(out_a1, out_b)
