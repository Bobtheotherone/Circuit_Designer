"""Datasets and batching for surrogate training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fidp.search.features import CircuitGraph, FeatureConfig, GraphTensors, extract_graph_features

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    Dataset = object  # type: ignore


@dataclass(frozen=True)
class CircuitSample:
    """Container for a circuit sample and its targets."""

    graph: CircuitGraph
    targets: np.ndarray
    global_features: Optional[np.ndarray] = None


@dataclass(frozen=True)
class GraphBatch:
    """Batched graph tensors for model ingestion."""

    node_features: "torch.Tensor"
    node_mask: "torch.Tensor"
    adjacency: "torch.Tensor"
    edge_features: Optional["torch.Tensor"]
    global_features: Optional["torch.Tensor"]
    targets: "torch.Tensor"


class CircuitDataset(Dataset):
    """Dataset of circuit graphs and target metrics."""

    def __init__(self, samples: Sequence[CircuitSample], config: FeatureConfig) -> None:
        if torch is None:
            raise RuntimeError("Torch is required for CircuitDataset.")
        self._samples = list(samples)
        self._config = config

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[GraphTensors, np.ndarray]:
        sample = self._samples[idx]
        graph_tensors = extract_graph_features(
            sample.graph,
            config=self._config,
            global_features=sample.global_features,
        )
        targets = np.asarray(sample.targets, dtype=np.float32)
        return graph_tensors, targets


def collate_fn(batch: Iterable[Tuple[GraphTensors, np.ndarray]]) -> GraphBatch:
    """Collate a batch of graph tensors into padded torch tensors."""
    if torch is None:
        raise RuntimeError("Torch is required for collate_fn.")
    batch_list = list(batch)
    if not batch_list:
        raise ValueError("Empty batch provided to collate_fn.")

    node_feature_dim = batch_list[0][0].node_features.shape[1]
    max_nodes = max(item[0].node_features.shape[0] for item in batch_list)

    has_edge_features = any(item[0].edge_features is not None for item in batch_list)
    edge_feature_dim = 0
    if has_edge_features:
        for item in batch_list:
            if item[0].edge_features is not None:
                edge_feature_dim = item[0].edge_features.shape[1]
                break

    global_dim = 0
    if any(item[0].global_features is not None for item in batch_list):
        for item in batch_list:
            if item[0].global_features is not None:
                global_dim = item[0].global_features.shape[0]
                break

    node_features = torch.zeros((len(batch_list), max_nodes, node_feature_dim), dtype=torch.float32)
    node_mask = torch.zeros((len(batch_list), max_nodes), dtype=torch.bool)
    adjacency = torch.zeros((len(batch_list), max_nodes, max_nodes), dtype=torch.bool)

    edge_features = None
    if has_edge_features:
        edge_features = torch.zeros(
            (len(batch_list), max_nodes, max_nodes, edge_feature_dim), dtype=torch.float32
        )

    global_features = None
    if global_dim > 0:
        global_features = torch.zeros((len(batch_list), global_dim), dtype=torch.float32)

    targets_list: List[torch.Tensor] = []

    for batch_idx, (graph, targets) in enumerate(batch_list):
        node_count = graph.node_features.shape[0]
        node_features[batch_idx, :node_count, :] = torch.from_numpy(graph.node_features)
        node_mask[batch_idx, :node_count] = True

        edge_index = graph.edge_index
        if edge_index.shape[1] > 0:
            src = edge_index[0]
            dst = edge_index[1]
            adjacency[batch_idx, dst, src] = True
            if edge_features is not None and graph.edge_features is not None:
                for edge_idx in range(edge_index.shape[1]):
                    edge_features[batch_idx, dst[edge_idx], src[edge_idx], :] = torch.from_numpy(
                        graph.edge_features[edge_idx]
                    )

        # Always include self loops for real nodes.
        adjacency[batch_idx, :node_count, :node_count] |= torch.eye(
            node_count, dtype=torch.bool
        )

        if global_features is not None:
            if graph.global_features is not None:
                global_features[batch_idx, :] = torch.from_numpy(graph.global_features)

        targets_list.append(torch.from_numpy(np.asarray(targets, dtype=np.float32)))

    targets_tensor = torch.stack(targets_list, dim=0)

    return GraphBatch(
        node_features=node_features,
        node_mask=node_mask,
        adjacency=adjacency,
        edge_features=edge_features,
        global_features=global_features,
        targets=targets_tensor,
    )


def move_batch_to_device(batch: GraphBatch, device: "torch.device") -> GraphBatch:
    """Move a GraphBatch to the specified device."""
    return GraphBatch(
        node_features=batch.node_features.to(device),
        node_mask=batch.node_mask.to(device),
        adjacency=batch.adjacency.to(device),
        edge_features=batch.edge_features.to(device) if batch.edge_features is not None else None,
        global_features=batch.global_features.to(device)
        if batch.global_features is not None
        else None,
        targets=batch.targets.to(device),
    )
