"""Canonicalization and deduplication utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from fidp.circuits.ir import CircuitIR, Component
from fidp.circuits.ops import flatten_circuit
from fidp.errors import CircuitIRValidationError


DEFAULT_MAX_DEPTH = 12
DEFAULT_MAX_NODES = 5000
DEFAULT_MAX_COMPONENTS = 10000


@dataclass(frozen=True)
class CanonicalCircuit:
    canonical_hash: str
    canonical_serialization: str
    node_mapping: Dict[str, str]
    mode: str


class CanonicalizationCache:
    """Simple LRU cache for canonicalization results."""

    def __init__(self, max_items: int = 1024) -> None:
        self.max_items = max_items
        self._cache: OrderedDict[str, CanonicalCircuit] = OrderedDict()

    def get(self, key: str) -> Optional[CanonicalCircuit]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: CanonicalCircuit) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


class DedupeIndex:
    """Incremental deduplication index for CircuitIR objects."""

    def __init__(
        self,
        mode: str = "continuous",
        cache: Optional[CanonicalizationCache] = None,
        max_depth: int | None = DEFAULT_MAX_DEPTH,
        max_nodes: int | None = DEFAULT_MAX_NODES,
        max_components: int | None = DEFAULT_MAX_COMPONENTS,
    ) -> None:
        self.mode = mode
        self.cache = cache or CanonicalizationCache()
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_components = max_components
        self._seen: Dict[str, CanonicalCircuit] = {}

    def add(self, circuit: CircuitIR) -> Tuple[str, bool]:
        canonical = canonicalize_circuit(
            circuit,
            mode=self.mode,
            cache=self.cache,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
            max_components=self.max_components,
        )
        if canonical.canonical_hash in self._seen:
            return canonical.canonical_hash, False
        self._seen[canonical.canonical_hash] = canonical
        return canonical.canonical_hash, True


def canonicalize_circuit(
    circuit: CircuitIR,
    mode: str = "continuous",
    cache: Optional[CanonicalizationCache] = None,
    max_depth: int | None = DEFAULT_MAX_DEPTH,
    max_nodes: int | None = DEFAULT_MAX_NODES,
    max_components: int | None = DEFAULT_MAX_COMPONENTS,
) -> CanonicalCircuit:
    """Canonicalize a circuit and return hash/serialization/mapping."""
    flat = _flatten_for_canonicalization(circuit, max_depth, max_nodes, max_components)
    graph = _build_graph(flat, mode)
    signature = _prehash_signature(graph)
    cache_key = f"{mode}:{signature}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    order, mapping = _canonical_order(graph)
    serialization = _serialize_graph(graph, order, mapping)
    canonical_hash = hashlib.sha256(serialization.encode("utf-8")).hexdigest()
    canonical = CanonicalCircuit(
        canonical_hash=canonical_hash,
        canonical_serialization=serialization,
        node_mapping=mapping,
        mode=mode,
    )
    if cache is not None:
        cache.set(cache_key, canonical)
    return canonical


def are_isomorphic(
    circuit_a: CircuitIR,
    circuit_b: CircuitIR,
    mode: str = "continuous",
    max_depth: int | None = DEFAULT_MAX_DEPTH,
    max_nodes: int | None = DEFAULT_MAX_NODES,
    max_components: int | None = DEFAULT_MAX_COMPONENTS,
) -> bool:
    """Check isomorphism between two circuits using VF2 matching."""
    graph_a = _build_graph(_flatten_for_canonicalization(circuit_a, max_depth, max_nodes, max_components), mode)
    graph_b = _build_graph(_flatten_for_canonicalization(circuit_b, max_depth, max_nodes, max_components), mode)
    if not _cheap_filter(graph_a, graph_b):
        return False
    node_match = nx.algorithms.isomorphism.categorical_node_match("port_role", ())
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("edge_label", None)
    matcher = nx.algorithms.isomorphism.MultiGraphMatcher(graph_a, graph_b, node_match=node_match, edge_match=edge_match)
    return matcher.is_isomorphic()


def _build_graph(circuit: CircuitIR, mode: str) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    port_roles: Dict[str, List[str]] = {}
    for port in circuit.ports:
        port_roles.setdefault(port.pos, []).append(f"{port.name}+")
        port_roles.setdefault(port.neg, []).append(f"{port.name}-")

    nodes = set(port_roles.keys())
    for comp in circuit.components:
        nodes.add(comp.node_a)
        nodes.add(comp.node_b)

    for node in nodes:
        roles = tuple(sorted(port_roles.get(node, [])))
        graph.add_node(node, port_role=roles)

    for comp in circuit.components:
        label = _edge_label(comp, mode)
        graph.add_edge(
            comp.node_a,
            comp.node_b,
            edge_label=label,
        )

    return graph


def _edge_label(component: Component, mode: str) -> str:
    value = component.value.resolved("snapped" if mode == "snapped" else "continuous")
    return f"{component.kind}:{_format_float(value)}"


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _cheap_filter(graph_a: nx.MultiGraph, graph_b: nx.MultiGraph) -> bool:
    if graph_a.number_of_nodes() != graph_b.number_of_nodes():
        return False
    if graph_a.number_of_edges() != graph_b.number_of_edges():
        return False
    if sorted(dict(graph_a.degree()).values()) != sorted(dict(graph_b.degree()).values()):
        return False
    if _port_signature(graph_a) != _port_signature(graph_b):
        return False
    if _edge_histogram(graph_a) != _edge_histogram(graph_b):
        return False
    if _wl_hash(graph_a) != _wl_hash(graph_b):
        return False
    return True


def _port_signature(graph: nx.MultiGraph) -> Tuple[Tuple[str, ...], ...]:
    roles = [graph.nodes[node].get("port_role", ()) for node in graph.nodes]
    return tuple(sorted(tuple(role) for role in roles))


def _edge_histogram(graph: nx.MultiGraph) -> Tuple[Tuple[str, int], ...]:
    counts: Dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        label = data.get("edge_label", "")
        counts[label] = counts.get(label, 0) + 1
    return tuple(sorted(counts.items()))


def _prehash_signature(graph: nx.MultiGraph) -> str:
    signature = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "degree": sorted(dict(graph.degree()).values()),
        "ports": _port_signature(graph),
        "edgesig": _edge_histogram(graph),
        "wl": _wl_hash(graph),
    }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _flatten_for_canonicalization(
    circuit: CircuitIR,
    max_depth: int | None,
    max_nodes: int | None,
    max_components: int | None,
) -> CircuitIR:
    flat = flatten_circuit(
        circuit,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_components=max_components,
    )
    if flat.subcircuits:
        depth = "None" if max_depth is None else str(max_depth)
        raise CircuitIRValidationError(
            "Canonicalization requires full flattening; "
            f"max_depth={depth} stopped expansion."
        )
    return flat


def _wl_hash(graph: nx.MultiGraph, iterations: int = 3) -> str:
    labels = {
        node: hashlib.sha256(
            (str(graph.nodes[node].get("port_role", ())) + str(graph.degree(node))).encode("utf-8")
        ).hexdigest()
        for node in graph.nodes
    }
    for _ in range(iterations):
        new_labels = {}
        for node in graph.nodes:
            neighbor_info = []
            for neighbor in graph.neighbors(node):
                for key in graph[node][neighbor]:
                    edge_label = graph[node][neighbor][key].get("edge_label", "")
                    neighbor_info.append(f"{edge_label}:{labels[neighbor]}")
            neighbor_info.sort()
            payload = "|".join([labels[node]] + neighbor_info)
            new_labels[node] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        labels = new_labels
    graph_payload = "|".join(sorted(labels.values()))
    return hashlib.sha256(graph_payload.encode("utf-8")).hexdigest()


def _canonical_order(graph: nx.MultiGraph) -> Tuple[List[str], Dict[str, str]]:
    nodes = list(graph.nodes)
    colors = _initial_colors(graph, nodes)
    colors = _refine_colors(graph, colors)
    best_serialization: Optional[str] = None
    best_order: List[str] = []

    def search(current_colors: Dict[str, int]) -> None:
        nonlocal best_serialization, best_order
        refined = _refine_colors(graph, current_colors)
        if _is_discrete(refined):
            order = _order_from_colors(refined)
            mapping = {node: f"n{idx}" for idx, node in enumerate(order)}
            serialization = _serialize_graph(graph, order, mapping)
            if best_serialization is None or serialization < best_serialization:
                best_serialization = serialization
                best_order = order
            return
        color_classes = _color_classes(refined)
        candidates = [(color, nodes) for color, nodes in color_classes.items() if len(nodes) > 1]
        color_id, target_class = min(candidates, key=lambda item: (len(item[1]), item[0]))
        max_color = max(refined.values())
        for node in sorted(target_class):
            individualized = dict(refined)
            individualized[node] = max_color + 1
            search(individualized)

    search(colors)
    mapping = {node: f"n{idx}" for idx, node in enumerate(best_order)}
    return best_order, mapping


def _initial_colors(graph: nx.MultiGraph, nodes: Iterable[str]) -> Dict[str, int]:
    signatures = {}
    for node in nodes:
        edge_labels = []
        for neighbor in graph.neighbors(node):
            for key in graph[node][neighbor]:
                edge_labels.append(graph[node][neighbor][key].get("edge_label", ""))
        edge_labels.sort()
        signature = (
            graph.nodes[node].get("port_role", ()),
            graph.degree(node),
            tuple(edge_labels),
        )
        signatures[node] = signature
    return _assign_colors(signatures)


def _refine_colors(graph: nx.MultiGraph, colors: Dict[str, int]) -> Dict[str, int]:
    current = dict(colors)
    while True:
        signatures = {}
        for node in graph.nodes:
            neighbor_info = []
            for neighbor in graph.neighbors(node):
                for key in graph[node][neighbor]:
                    edge_label = graph[node][neighbor][key].get("edge_label", "")
                    neighbor_info.append((edge_label, current[neighbor]))
            neighbor_info.sort()
            signatures[node] = (
                current[node],
                graph.nodes[node].get("port_role", ()),
                tuple(neighbor_info),
            )
        new_colors = _assign_colors(signatures)
        if new_colors == current:
            return current
        current = new_colors


def _assign_colors(signatures: Dict[str, Tuple[object, ...]]) -> Dict[str, int]:
    unique = sorted(set(signatures.values()))
    color_map = {sig: idx for idx, sig in enumerate(unique)}
    return {node: color_map[signature] for node, signature in signatures.items()}


def _is_discrete(colors: Dict[str, int]) -> bool:
    return len(set(colors.values())) == len(colors)


def _color_classes(colors: Dict[str, int]) -> Dict[int, List[str]]:
    classes: Dict[int, List[str]] = {}
    for node, color in colors.items():
        classes.setdefault(color, []).append(node)
    return classes


def _order_from_colors(colors: Dict[str, int]) -> List[str]:
    return [node for node, _ in sorted(colors.items(), key=lambda item: item[1])]


def _serialize_graph(graph: nx.MultiGraph, order: List[str], mapping: Dict[str, str]) -> str:
    node_data = [
        {
            "id": mapping[node],
            "port_role": graph.nodes[node].get("port_role", ()),
        }
        for node in order
    ]
    edge_data = []
    for u, v, data in graph.edges(data=True):
        u_idx = int(mapping[u][1:])
        v_idx = int(mapping[v][1:])
        if u_idx > v_idx:
            u_idx, v_idx = v_idx, u_idx
        edge_data.append({"u": u_idx, "v": v_idx, "label": data.get("edge_label", "")})
    edge_data.sort(key=lambda item: (item["u"], item["v"], item["label"]))
    payload = {
        "nodes": node_data,
        "edges": edge_data,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
