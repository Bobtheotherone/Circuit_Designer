"""Search and AI engine utilities for FIDP."""

from fidp.search.active_learning import ActiveLearningLoop
from fidp.search.botorch_mobo import propose_next_botorch, propose_next_random
from fidp.search.dataset import CircuitDataset, CircuitSample, GraphBatch, collate_fn
from fidp.search.evolution import DesignRecord, EvolutionConfig, evolve_population
from fidp.search.features import (
    CircuitGraph,
    Component,
    FeatureConfig,
    GraphTensors,
    extract_graph_features,
    from_core_circuit,
    validate_circuit_graph,
)
from fidp.search.surrogate import GraphSurrogateModel, SurrogateConfig, predict
from fidp.search.train_surrogate import TrainingConfig, TrainingResult, train_surrogate

__all__ = [
    "ActiveLearningLoop",
    "propose_next_botorch",
    "propose_next_random",
    "CircuitDataset",
    "CircuitSample",
    "GraphBatch",
    "collate_fn",
    "DesignRecord",
    "EvolutionConfig",
    "evolve_population",
    "CircuitGraph",
    "Component",
    "FeatureConfig",
    "GraphTensors",
    "extract_graph_features",
    "from_core_circuit",
    "validate_circuit_graph",
    "GraphSurrogateModel",
    "SurrogateConfig",
    "predict",
    "TrainingConfig",
    "TrainingResult",
    "train_surrogate",
]
