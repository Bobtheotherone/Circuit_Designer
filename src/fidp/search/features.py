"""Feature extraction for circuit graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fidp.circuits.core import (
    Capacitor,
    CircuitGraph as CoreCircuitGraph,
    Inductor,
    Port as CorePort,
    Resistor,
)
from fidp.errors import CircuitValidationError


COMPONENT_TYPES: Tuple[str, ...] = ("R", "C", "L", "PORT", "WIRE", "OTHER")
_TYPE_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(COMPONENT_TYPES)}


@dataclass(frozen=True)
class Component:
    """Minimal component representation for graph search."""

    kind: str
    node_a: str
    node_b: str
    value: Optional[float] = None
    depth: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.node_a or not self.node_b:
            raise CircuitValidationError("Component nodes must be provided.")
        if self.value is not None and self.value <= 0:
            raise CircuitValidationError("Component value must be positive.")
        if self.depth is not None and self.depth < 0:
            raise CircuitValidationError("Component depth must be non-negative.")


@dataclass
class CircuitGraph:
    """Lightweight circuit graph for search and surrogate modeling."""

    components: List[Component] = field(default_factory=list)
    ports: List[CorePort] = field(default_factory=list)
    ground: str = "0"
    nodes: set[str] = field(default_factory=set)
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ground:
            raise CircuitValidationError("Ground node must be provided.")
        self.nodes.add(self.ground)
        for comp in self.components:
            self._register_component_nodes(comp)
        for port in self.ports:
            if not port.pos or not port.neg:
                raise CircuitValidationError("Port nodes must be provided.")
            self.nodes.add(port.pos)
            self.nodes.add(port.neg)
        if not self.ports and not any(comp.kind.upper() == "PORT" for comp in self.components):
            raise CircuitValidationError("At least one port must be defined.")

    def _register_component_nodes(self, comp: Component) -> None:
        self.nodes.add(comp.node_a)
        self.nodes.add(comp.node_b)

    def iter_components(self, include_ports: bool = True) -> Iterable[Component]:
        if include_ports:
            return iter(self._components_with_ports())
        return iter(self.components)

    def _components_with_ports(self) -> List[Component]:
        comps = list(self.components)
        has_port = any(comp.kind.upper() == "PORT" for comp in comps)
        if self.ports and not has_port:
            for port in self.ports:
                comps.append(Component(kind="PORT", node_a=port.pos, node_b=port.neg))
        return comps


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature extraction."""

    include_depth: bool = True
    include_distance_to_port: bool = True
    include_edge_features: bool = False


@dataclass(frozen=True)
class GraphTensors:
    """Graph tensor bundle for a single circuit graph."""

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: Optional[np.ndarray] = None
    global_features: Optional[np.ndarray] = None


def validate_circuit_graph(graph: CircuitGraph) -> None:
    """Validate a circuit graph before feature extraction."""
    if not graph.components and not graph.ports:
        raise CircuitValidationError("Circuit graph must include components or ports.")
    if not graph.ports and not any(comp.kind.upper() == "PORT" for comp in graph.components):
        raise CircuitValidationError("Circuit graph is missing ports.")
    for comp in graph.components:
        if comp.value is not None and comp.value <= 0:
            raise CircuitValidationError("Component value must be positive.")


def extract_graph_features(
    graph: CircuitGraph,
    config: Optional[FeatureConfig] = None,
    global_features: Optional[Sequence[float]] = None,
) -> GraphTensors:
    """Convert a circuit graph into tensor features."""
    config = config or FeatureConfig()
    validate_circuit_graph(graph)

    components = list(graph.iter_components(include_ports=True))
    if not components:
        raise CircuitValidationError("Circuit graph has no components.")

    type_count = len(COMPONENT_TYPES)
    node_features: List[List[float]] = []

    adjacency = _build_adjacency(components)
    if config.include_distance_to_port:
        distances = _distance_to_ports(components, adjacency)
    else:
        distances = [0.0 for _ in components]

    for idx, comp in enumerate(components):
        kind = comp.kind.upper()
        if kind not in _TYPE_TO_INDEX:
            kind = "OTHER"
        one_hot = [0.0] * type_count
        one_hot[_TYPE_TO_INDEX[kind]] = 1.0
        if comp.value is None:
            log_value = 0.0
        else:
            if comp.value <= 0:
                raise CircuitValidationError("Component value must be positive.")
            log_value = float(np.log10(comp.value))
        feats = one_hot + [log_value]
        if config.include_depth:
            feats.append(float(comp.depth or 0))
        if config.include_distance_to_port:
            feats.append(float(distances[idx]))
        node_features.append(feats)

    node_array = np.asarray(node_features, dtype=np.float32)

    edge_index, edge_features = _build_edges(components, config.include_edge_features)

    if global_features is None:
        global_array = None
    else:
        global_array = np.asarray(global_features, dtype=np.float32)
        if global_array.ndim != 1:
            raise CircuitValidationError("Global features must be a 1D array.")

    return GraphTensors(
        node_features=node_array,
        edge_index=edge_index,
        edge_features=edge_features,
        global_features=global_array,
    )


def _build_adjacency(components: Sequence[Component]) -> Dict[int, List[int]]:
    node_to_components: Dict[str, List[int]] = {}
    for idx, comp in enumerate(components):
        node_to_components.setdefault(comp.node_a, []).append(idx)
        node_to_components.setdefault(comp.node_b, []).append(idx)

    adjacency: Dict[int, List[int]] = {idx: [] for idx in range(len(components))}
    for comp_indices in node_to_components.values():
        for i in comp_indices:
            for j in comp_indices:
                if i == j:
                    continue
                adjacency[i].append(j)

    return adjacency


def _distance_to_ports(components: Sequence[Component], adjacency: Dict[int, List[int]]) -> List[float]:
    from collections import deque

    port_indices = [idx for idx, comp in enumerate(components) if comp.kind.upper() == "PORT"]
    if not port_indices:
        raise CircuitValidationError("Ports are required to compute distance-to-port.")

    distances = [float("inf")] * len(components)
    queue = deque()
    for idx in port_indices:
        distances[idx] = 0.0
        queue.append(idx)

    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if distances[neighbor] == float("inf"):
                distances[neighbor] = distances[current] + 1.0
                queue.append(neighbor)

    if any(dist == float("inf") for dist in distances):
        raise CircuitValidationError("Some components are unreachable from ports.")

    return distances


def _build_edges(
    components: Sequence[Component],
    include_edge_features: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    edge_list: List[Tuple[int, int]] = []
    edge_feat_list: List[List[float]] = []

    node_to_components: Dict[str, List[int]] = {}
    for idx, comp in enumerate(components):
        node_to_components.setdefault(comp.node_a, []).append(idx)
        node_to_components.setdefault(comp.node_b, []).append(idx)

    seen = set()
    for comp_indices in node_to_components.values():
        for i in comp_indices:
            for j in comp_indices:
                if i == j:
                    continue
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                edge_list.append((i, j))
                if include_edge_features:
                    shared = _shared_node_count(components[i], components[j])
                    edge_feat_list.append([float(shared)])

    if edge_list:
        edge_index = np.array(edge_list, dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    if include_edge_features:
        edge_features = np.asarray(edge_feat_list, dtype=np.float32)
    else:
        edge_features = None

    return edge_index, edge_features


def _shared_node_count(comp_a: Component, comp_b: Component) -> int:
    nodes_a = {comp_a.node_a, comp_a.node_b}
    nodes_b = {comp_b.node_a, comp_b.node_b}
    return len(nodes_a.intersection(nodes_b))


def from_core_circuit(
    circuit: CoreCircuitGraph,
    ports: Optional[Sequence[CorePort]] = None,
    depth_by_index: Optional[Dict[int, int]] = None,
) -> CircuitGraph:
    """Convert a core circuit graph into a search circuit graph."""
    components: List[Component] = []
    for idx, comp in enumerate(circuit.components):
        if isinstance(comp, Resistor):
            value = comp.resistance_ohms
            kind = "R"
        elif isinstance(comp, Capacitor):
            value = comp.capacitance_f
            kind = "C"
        elif isinstance(comp, Inductor):
            value = comp.inductance_h
            kind = "L"
        elif isinstance(comp, CorePort):
            value = None
            kind = "PORT"
            comp = Component(kind=kind, node_a=comp.pos, node_b=comp.neg, value=value)
            components.append(comp)
            continue
        else:
            value = _extract_value(comp)
            kind = _infer_kind(comp)
        depth = None if depth_by_index is None else depth_by_index.get(idx)
        components.append(
            Component(
                kind=kind,
                node_a=getattr(comp, "node_a", getattr(comp, "pos", "")),
                node_b=getattr(comp, "node_b", getattr(comp, "neg", "")),
                value=value,
                depth=depth,
            )
        )

    return CircuitGraph(
        components=components,
        ports=list(ports) if ports is not None else [],
        ground=circuit.ground,
    )


def _extract_value(comp: object) -> Optional[float]:
    for attr in ("resistance_ohms", "capacitance_f", "inductance_h", "value"):
        if hasattr(comp, attr):
            value = getattr(comp, attr)
            if value is None:
                return None
            return float(value)
    return None


def _infer_kind(comp: object) -> str:
    name = comp.__class__.__name__.upper()
    if "RES" in name:
        return "R"
    if "CAP" in name:
        return "C"
    if "IND" in name:
        return "L"
    if "PORT" in name:
        return "PORT"
    return "OTHER"
