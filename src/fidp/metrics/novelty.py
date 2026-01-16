"""Novelty scoring utilities for topology and impedance response."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import hashlib
import json
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np

from fidp.circuits.core import Port as CorePort
from fidp.modeling.fractional_fit import fit_cpe
from fidp.search.features import COMPONENT_TYPES, CircuitGraph as SearchCircuitGraph


@dataclass(frozen=True)
class NoveltyConfig:
    """Configuration for novelty scoring and feature extraction."""

    topology_weight: float = 0.5
    response_weight: float = 0.5
    knn_k: int = 5
    distance_metric: str = "cosine"
    knn_aggregation: str = "mean"
    max_degree_bin: int = 6
    spectral_k: int = 6
    include_triangle_count: bool = True
    include_four_cycles: bool = False
    component_kinds: Tuple[str, ...] = COMPONENT_TYPES
    normalize_component_hist: bool = True
    normalize_degree_hist: bool = True
    normalize_cycle_counts: bool = True
    response_grid_size: int = 64
    include_phase: bool = True
    include_alpha: bool = True
    include_mc_variability: bool = False
    cpe_distance_weight: float = 0.2
    cpe_distance_scale: float = 0.2
    target_cpe: bool = False
    normalization: str = "zscore"
    zscore_clip: float = 3.0
    stats_sample_size: int = 512
    stats_seed: int = 0
    store_features: bool = False
    store_neighbors: bool = True
    feature_version: int = 1

    def __post_init__(self) -> None:
        if self.knn_k <= 0:
            raise ValueError("knn_k must be positive.")
        if self.max_degree_bin < 0:
            raise ValueError("max_degree_bin must be non-negative.")
        if self.spectral_k < 0:
            raise ValueError("spectral_k must be non-negative.")
        if self.response_grid_size < 4:
            raise ValueError("response_grid_size must be >= 4.")
        if self.distance_metric not in {"cosine", "l2"}:
            raise ValueError("distance_metric must be 'cosine' or 'l2'.")
        if self.knn_aggregation not in {"mean", "min"}:
            raise ValueError("knn_aggregation must be 'mean' or 'min'.")
        if self.cpe_distance_scale <= 0.0:
            raise ValueError("cpe_distance_scale must be positive.")
        if self.normalization not in {"zscore"}:
            raise ValueError("normalization must be 'zscore'.")
        if self.stats_sample_size < 2:
            raise ValueError("stats_sample_size must be >= 2.")

    def feature_fingerprint(self) -> dict[str, Any]:
        return {
            "feature_version": self.feature_version,
            "max_degree_bin": self.max_degree_bin,
            "spectral_k": self.spectral_k,
            "include_triangle_count": self.include_triangle_count,
            "include_four_cycles": self.include_four_cycles,
            "component_kinds": list(self.component_kinds),
            "normalize_component_hist": self.normalize_component_hist,
            "normalize_degree_hist": self.normalize_degree_hist,
            "normalize_cycle_counts": self.normalize_cycle_counts,
            "response_grid_size": self.response_grid_size,
            "include_phase": self.include_phase,
            "include_alpha": self.include_alpha,
            "include_mc_variability": self.include_mc_variability,
        }

    def feature_hash(self) -> str:
        payload = json.dumps(self.feature_fingerprint(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CPEBaselineFit:
    """Compact summary of an ideal CPE fit."""

    alpha: float
    c_alpha: float
    rmse_logmag: float
    rmse_phase_deg: float
    residual: float


@dataclass(frozen=True)
class NoveltyMetrics:
    """Novelty scores and optional diagnostics."""

    topology_novelty: float
    response_novelty: float
    overall_novelty: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class NoveltyCorpus:
    """Feature cache for novelty scoring."""

    topology_features: np.ndarray
    response_features: np.ndarray
    design_ids: list[str]
    config_hash: str
    stats: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    @classmethod
    def empty(cls, config: NoveltyConfig) -> NoveltyCorpus:
        topo_dim = topology_feature_dim(config)
        resp_dim = response_feature_dim(config)
        return cls(
            topology_features=np.zeros((0, topo_dim), dtype=np.float64),
            response_features=np.zeros((0, resp_dim), dtype=np.float64),
            design_ids=[],
            config_hash=config.feature_hash(),
        )

    @property
    def size(self) -> int:
        return len(self.design_ids)

    def add(
        self,
        design_id: str,
        topo_vec: np.ndarray,
        resp_vec: np.ndarray,
        allow_duplicate: bool = False,
    ) -> None:
        if not design_id:
            raise ValueError("design_id must be provided.")
        if not allow_duplicate and design_id in self.design_ids:
            raise ValueError(f"design_id '{design_id}' already exists in corpus.")
        topo_vec = np.asarray(topo_vec, dtype=np.float64).reshape(1, -1)
        resp_vec = np.asarray(resp_vec, dtype=np.float64).reshape(1, -1)
        if self.topology_features.size and topo_vec.shape[1] != self.topology_features.shape[1]:
            raise ValueError("Topology feature dimension mismatch.")
        if self.response_features.size and resp_vec.shape[1] != self.response_features.shape[1]:
            raise ValueError("Response feature dimension mismatch.")
        if self.topology_features.size == 0:
            self.topology_features = topo_vec
        else:
            self.topology_features = np.vstack([self.topology_features, topo_vec])
        if self.response_features.size == 0:
            self.response_features = resp_vec
        else:
            self.response_features = np.vstack([self.response_features, resp_vec])
        self.design_ids.append(design_id)
        self.stats = {}

    def knn_topology(
        self,
        topo_vec: np.ndarray,
        *,
        k: int,
        metric: str,
        exclude_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, list[str]]:
        return _knn(self.topology_features, self.design_ids, topo_vec, k, metric, exclude_id)

    def knn_response(
        self,
        resp_vec: np.ndarray,
        *,
        k: int,
        metric: str,
        exclude_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, list[str]]:
        return _knn(self.response_features, self.design_ids, resp_vec, k, metric, exclude_id)

    def ensure_stats(self, config: NoveltyConfig) -> dict[str, Any]:
        topo_stats = self.stats.get("topology")
        resp_stats = self.stats.get("response")
        if (
            topo_stats is None
            or resp_stats is None
            or topo_stats.get("metric") != config.distance_metric
            or resp_stats.get("metric") != config.distance_metric
            or topo_stats.get("k") != config.knn_k
            or resp_stats.get("k") != config.knn_k
            or topo_stats.get("aggregation") != config.knn_aggregation
            or resp_stats.get("aggregation") != config.knn_aggregation
        ):
            topo_stats = _compute_knn_stats(
                self.topology_features,
                config,
            )
            resp_stats = _compute_knn_stats(
                self.response_features,
                config,
            )
            self.stats["topology"] = topo_stats
            self.stats["response"] = resp_stats
        return {"topology": topo_stats, "response": resp_stats}

    def save(self, path: str) -> None:
        metadata = {
            "version": self.version,
            "config_hash": self.config_hash,
            "stats": self.stats,
        }
        np.savez(
            path,
            topology_features=self.topology_features,
            response_features=self.response_features,
            design_ids=np.asarray(self.design_ids, dtype=str),
            metadata=json.dumps(metadata, separators=(",", ":")),
        )

    @classmethod
    def load(cls, path: str, config: NoveltyConfig) -> NoveltyCorpus:
        with np.load(path) as data:
            metadata = json.loads(str(data["metadata"].item()))
            config_hash = metadata.get("config_hash")
            if config_hash != config.feature_hash():
                raise ValueError(
                    "Novelty corpus config hash mismatch. Rebuild corpus with matching feature settings."
                )
            corpus = cls(
                topology_features=np.asarray(data["topology_features"], dtype=np.float64),
                response_features=np.asarray(data["response_features"], dtype=np.float64),
                design_ids=[str(item) for item in data["design_ids"].tolist()],
                config_hash=config_hash,
                stats=metadata.get("stats", {}),
                version=int(metadata.get("version", 1)),
            )
        return corpus


@dataclass(frozen=True)
class _ComponentView:
    kind: str
    node_a: str
    node_b: str


def topology_feature_dim(config: NoveltyConfig) -> int:
    base = 10
    return (
        base
        + len(config.component_kinds)
        + (config.max_degree_bin + 2)
        + 2
        + config.spectral_k
    )


def response_feature_dim(config: NoveltyConfig) -> int:
    total = config.response_grid_size
    if config.include_phase:
        total += config.response_grid_size
    if config.include_alpha:
        total += config.response_grid_size
    total += 3
    if config.include_mc_variability:
        total += 1
    return total


def extract_topology_features(graph: object, *, config: NoveltyConfig) -> np.ndarray:
    """Extract fixed-length topology features from a circuit graph."""
    components, port_pairs, nodes = _collect_components(graph)
    node_list = sorted(nodes)
    node_index = {node: idx for idx, node in enumerate(node_list)}

    adjacency = _build_adjacency(components, node_index)
    n_nodes = float(len(node_list))
    n_edges = float(sum(1 for comp in components if comp.kind != "PORT"))
    n_components = float(_count_components(adjacency))
    disconnected_flag = 1.0 if n_components > 1.0 else 0.0

    port_stats = _port_distance_stats(port_pairs, node_index, adjacency)

    kind_hist = _component_histogram(components, config)
    degree_hist = _degree_histogram(adjacency, config)
    triangle_count = _triangle_count(adjacency) if config.include_triangle_count else 0.0
    four_cycle_count = _four_cycle_count(adjacency) if config.include_four_cycles else 0.0
    if config.normalize_cycle_counts:
        denom = max(1.0, n_nodes)
        triangle_count /= denom
        four_cycle_count /= denom

    spectral = _spectral_signature(adjacency, config.spectral_k)

    features = np.concatenate(
        [
            np.array(
                [
                    n_nodes,
                    n_edges,
                    n_components,
                    disconnected_flag,
                    float(port_stats["port_present"]),
                    float(port_stats["port_count"]),
                    float(port_stats["dist_min"]),
                    float(port_stats["dist_mean"]),
                    float(port_stats["dist_max"]),
                    float(port_stats["port_unreachable"]),
                ],
                dtype=np.float64,
            ),
            kind_hist,
            degree_hist,
            np.array([triangle_count, four_cycle_count], dtype=np.float64),
            spectral,
        ]
    )

    expected = topology_feature_dim(config)
    if features.size != expected:
        raise ValueError(f"Topology feature size {features.size} != expected {expected}.")
    return features


def extract_response_features(freq_hz: np.ndarray, Z: np.ndarray, *, config: NoveltyConfig) -> np.ndarray:
    """Extract fixed-length response features from impedance data."""
    features, _ = _extract_response_features(freq_hz, Z, config)
    return features


def score_novelty(
    design_or_eval: object,
    *,
    corpus: NoveltyCorpus,
    config: NoveltyConfig,
) -> NoveltyMetrics:
    """Compute novelty scores for a design or evaluation bundle."""
    graph, freq_hz, Z, design_id = _unpack_design(design_or_eval)
    topo_vec = extract_topology_features(graph, config=config)
    resp_vec, resp_diag = _extract_response_features(freq_hz, Z, config)
    return score_novelty_from_features(
        topo_vec,
        resp_vec,
        corpus=corpus,
        config=config,
        design_id=design_id,
        response_diag=resp_diag,
    )


def score_novelty_from_features(
    topo_vec: np.ndarray,
    resp_vec: np.ndarray,
    *,
    corpus: NoveltyCorpus,
    config: NoveltyConfig,
    design_id: Optional[str] = None,
    response_diag: Optional[dict[str, Any]] = None,
) -> NoveltyMetrics:
    """Score novelty given precomputed feature vectors."""
    flags: list[str] = []
    topo_score = 0.0
    resp_score = 0.0
    topo_neighbors: list[str] = []
    resp_neighbors: list[str] = []
    topo_distances = np.array([], dtype=np.float64)
    resp_distances = np.array([], dtype=np.float64)

    if corpus.size == 0:
        flags.append("empty_corpus")
    else:
        if corpus.size < config.knn_k:
            flags.append("insufficient_corpus")
        k = min(config.knn_k, corpus.size)
        topo_distances, topo_neighbors = corpus.knn_topology(
            topo_vec, k=k, metric=config.distance_metric, exclude_id=design_id
        )
        resp_distances, resp_neighbors = corpus.knn_response(
            resp_vec, k=k, metric=config.distance_metric, exclude_id=design_id
        )
        if topo_distances.size == 0 or resp_distances.size == 0:
            flags.append("no_neighbors")
        else:
            stats = corpus.ensure_stats(config)
            topo_raw = _aggregate_knn(topo_distances, config.knn_aggregation)
            resp_raw = _aggregate_knn(resp_distances, config.knn_aggregation)
            topo_score = _normalize_distance(topo_raw, stats["topology"], config)
            resp_score = _normalize_distance(resp_raw, stats["response"], config)

    if config.cpe_distance_weight > 0.0:
        cpe_distance = None if response_diag is None else response_diag.get("cpe_distance")
        if cpe_distance is not None:
            baseline = _baseline_score(float(cpe_distance), config)
            if config.target_cpe:
                baseline = 1.0 - baseline
            resp_score = (1.0 - config.cpe_distance_weight) * resp_score + config.cpe_distance_weight * baseline

    resp_score = float(np.clip(resp_score, 0.0, 1.0))
    topo_score = float(np.clip(topo_score, 0.0, 1.0))
    weight_sum = config.topology_weight + config.response_weight
    if weight_sum <= 0.0:
        overall = 0.0
    else:
        overall = (
            config.topology_weight * topo_score + config.response_weight * resp_score
        ) / weight_sum
    overall = float(np.clip(overall, 0.0, 1.0))

    diagnostics: dict[str, Any] = {}
    if flags:
        diagnostics["flags"] = flags
    if config.store_neighbors:
        diagnostics["topology_neighbors"] = topo_neighbors
        diagnostics["response_neighbors"] = resp_neighbors
        diagnostics["topology_distances"] = topo_distances
        diagnostics["response_distances"] = resp_distances
    if config.store_features:
        diagnostics["topology_features"] = topo_vec
        diagnostics["response_features"] = resp_vec
    if response_diag:
        diagnostics.update(response_diag)

    return NoveltyMetrics(
        topology_novelty=topo_score,
        response_novelty=resp_score,
        overall_novelty=overall,
        diagnostics=diagnostics,
    )


def fit_cpe_baseline(freq_hz: np.ndarray, Z: np.ndarray) -> CPEBaselineFit:
    """Fit an ideal CPE baseline model and summarize the residual."""
    fit = fit_cpe(freq_hz, Z)
    residual = float(
        np.sqrt(fit.rmse_logmag**2 + (fit.rmse_phase_deg / 180.0) ** 2)
    )
    return CPEBaselineFit(
        alpha=float(fit.alpha),
        c_alpha=float(fit.c_alpha),
        rmse_logmag=float(fit.rmse_logmag),
        rmse_phase_deg=float(fit.rmse_phase_deg),
        residual=residual,
    )


def _collect_components(
    graph: object,
) -> Tuple[list[_ComponentView], list[Tuple[str, str]], set[str]]:
    components: list[_ComponentView] = []
    port_pairs: list[Tuple[str, str]] = []
    nodes: set[str] = set()

    if isinstance(graph, SearchCircuitGraph):
        raw_components = list(graph.iter_components(include_ports=True))
        for comp in raw_components:
            kind = comp.kind.upper()
            node_a = comp.node_a
            node_b = comp.node_b
            components.append(_ComponentView(kind=kind, node_a=node_a, node_b=node_b))
            nodes.add(node_a)
            nodes.add(node_b)
            if kind == "PORT":
                port_pairs.append((node_a, node_b))
        nodes.update(graph.nodes)
        return components, port_pairs, nodes

    raw_iter: Iterable[object]
    if hasattr(graph, "iter_components"):
        raw_iter = graph.iter_components()
    elif hasattr(graph, "components"):
        raw_iter = getattr(graph, "components")
    else:
        raise TypeError("graph must provide components.")

    for comp in raw_iter:
        node_a, node_b = _extract_nodes(comp)
        kind = _infer_kind(comp)
        components.append(_ComponentView(kind=kind, node_a=node_a, node_b=node_b))
        nodes.add(node_a)
        nodes.add(node_b)
        if kind == "PORT":
            port_pairs.append((node_a, node_b))

    if hasattr(graph, "ports"):
        for port in getattr(graph, "ports"):
            node_a, node_b = _extract_port_nodes(port)
            nodes.add(node_a)
            nodes.add(node_b)
            if (node_a, node_b) not in port_pairs and (node_b, node_a) not in port_pairs:
                port_pairs.append((node_a, node_b))
                components.append(_ComponentView(kind="PORT", node_a=node_a, node_b=node_b))

    if hasattr(graph, "nodes"):
        nodes.update(getattr(graph, "nodes"))

    return components, port_pairs, nodes


def _extract_nodes(comp: object) -> Tuple[str, str]:
    if hasattr(comp, "node_a") and hasattr(comp, "node_b"):
        node_a = getattr(comp, "node_a")
        node_b = getattr(comp, "node_b")
        return str(node_a), str(node_b)
    if hasattr(comp, "pos") and hasattr(comp, "neg"):
        node_a = getattr(comp, "pos")
        node_b = getattr(comp, "neg")
        return str(node_a), str(node_b)
    raise ValueError("Component lacks node endpoints.")


def _extract_port_nodes(port: object) -> Tuple[str, str]:
    if isinstance(port, CorePort):
        return port.pos, port.neg
    if hasattr(port, "pos") and hasattr(port, "neg"):
        return str(getattr(port, "pos")), str(getattr(port, "neg"))
    raise ValueError("Port lacks pos/neg nodes.")


def _infer_kind(comp: object) -> str:
    if hasattr(comp, "kind"):
        return str(getattr(comp, "kind")).upper()
    name = comp.__class__.__name__.upper()
    if "RES" in name:
        return "R"
    if "CAP" in name:
        return "C"
    if "IND" in name:
        return "L"
    if "PORT" in name:
        return "PORT"
    if "VOLT" in name or name.startswith("V"):
        return "V"
    if "CURR" in name or name.startswith("I"):
        return "I"
    return "OTHER"


def _build_adjacency(
    components: Sequence[_ComponentView],
    node_index: dict[str, int],
) -> np.ndarray:
    n_nodes = len(node_index)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for comp in components:
        if comp.kind == "PORT":
            continue
        if comp.node_a not in node_index or comp.node_b not in node_index:
            continue
        i = node_index[comp.node_a]
        j = node_index[comp.node_b]
        if i == j:
            continue
        adjacency[i, j] += 1.0
        adjacency[j, i] += 1.0
    return adjacency


def _count_components(adjacency: np.ndarray) -> int:
    n = adjacency.shape[0]
    if n == 0:
        return 0
    visited = np.zeros(n, dtype=bool)
    count = 0
    for start in range(n):
        if visited[start]:
            continue
        count += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            neighbors = np.nonzero(adjacency[node])[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
    return count


def _port_distance_stats(
    port_pairs: Sequence[Tuple[str, str]],
    node_index: dict[str, int],
    adjacency: np.ndarray,
) -> dict[str, float]:
    port_count = len(port_pairs)
    if port_count == 0:
        return {
            "port_present": 0.0,
            "port_count": 0.0,
            "dist_min": 0.0,
            "dist_mean": 0.0,
            "dist_max": 0.0,
            "port_unreachable": 0.0,
        }
    neighbors = [np.nonzero(adjacency[idx])[0] for idx in range(adjacency.shape[0])]
    distances: list[float] = []
    unreachable = False
    for node_a, node_b in port_pairs:
        if node_a not in node_index or node_b not in node_index:
            unreachable = True
            continue
        dist = _shortest_path_length(
            neighbors, node_index[node_a], node_index[node_b]
        )
        if dist == float("inf"):
            unreachable = True
        else:
            distances.append(float(dist))
    if not distances:
        dist_min = dist_mean = dist_max = 0.0
    else:
        dist_min = float(np.min(distances))
        dist_mean = float(np.mean(distances))
        dist_max = float(np.max(distances))
    return {
        "port_present": 1.0,
        "port_count": float(port_count),
        "dist_min": dist_min,
        "dist_mean": dist_mean,
        "dist_max": dist_max,
        "port_unreachable": 1.0 if unreachable else 0.0,
    }


def _shortest_path_length(
    neighbors: Sequence[np.ndarray],
    start: int,
    goal: int,
) -> float:
    if start == goal:
        return 0.0
    queue: deque[int] = deque([start])
    visited = {start}
    distances = {start: 0}
    while queue:
        current = queue.popleft()
        next_dist = distances[current] + 1
        for neighbor in neighbors[current]:
            if neighbor in visited:
                continue
            if neighbor == goal:
                return float(next_dist)
            visited.add(neighbor)
            distances[neighbor] = next_dist
            queue.append(int(neighbor))
    return float("inf")


def _component_histogram(
    components: Sequence[_ComponentView], config: NoveltyConfig
) -> np.ndarray:
    kind_map = {kind: idx for idx, kind in enumerate(config.component_kinds)}
    counts = np.zeros(len(config.component_kinds), dtype=np.float64)
    for comp in components:
        kind = comp.kind.upper()
        if kind not in kind_map:
            kind = "OTHER" if "OTHER" in kind_map else config.component_kinds[-1]
        counts[kind_map[kind]] += 1.0
    if config.normalize_component_hist and counts.sum() > 0:
        counts /= counts.sum()
    return counts


def _degree_histogram(adjacency: np.ndarray, config: NoveltyConfig) -> np.ndarray:
    degrees = adjacency.sum(axis=1)
    bins = np.zeros(config.max_degree_bin + 2, dtype=np.float64)
    for degree in degrees:
        idx = int(round(degree))
        if idx > config.max_degree_bin:
            idx = config.max_degree_bin + 1
        bins[idx] += 1.0
    if config.normalize_degree_hist and bins.sum() > 0:
        bins /= bins.sum()
    return bins


def _triangle_count(adjacency: np.ndarray) -> float:
    n = adjacency.shape[0]
    if n < 3:
        return 0.0
    A = (adjacency > 0).astype(np.int64)
    tri = np.trace(A @ A @ A) / 6.0
    return float(tri)


def _four_cycle_count(adjacency: np.ndarray) -> float:
    n = adjacency.shape[0]
    if n < 4:
        return 0.0
    A = (adjacency > 0).astype(np.int64)
    A2 = A @ A
    upper = np.triu(A2, k=1)
    comb = upper * (upper - 1) / 2.0
    return float(np.sum(comb) / 2.0)


def _spectral_signature(adjacency: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros(0, dtype=np.float64)
    n = adjacency.shape[0]
    if n == 0:
        return np.zeros(k, dtype=np.float64)
    A = (adjacency > 0).astype(np.float64)
    degrees = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    L = np.eye(n) - (inv_sqrt[:, None] * A * inv_sqrt[None, :])
    eigs = np.linalg.eigvalsh(L)
    eigs = np.sort(eigs)
    if eigs.size >= k:
        return eigs[:k].astype(np.float64)
    padding = np.zeros(k - eigs.size, dtype=np.float64)
    return np.concatenate([eigs, padding])


def _validate_response_inputs(freq_hz: np.ndarray, Z: np.ndarray) -> None:
    if freq_hz.ndim != 1:
        raise ValueError("freq_hz must be a 1D array.")
    if np.any(freq_hz <= 0.0):
        raise ValueError("freq_hz must be strictly positive.")
    if np.any(np.diff(freq_hz) <= 0.0):
        raise ValueError("freq_hz must be strictly increasing.")
    if Z.ndim == 1:
        if Z.shape != freq_hz.shape:
            raise ValueError("Z shape must match freq_hz.")
    elif Z.ndim == 2:
        if Z.shape[1] != freq_hz.shape[0]:
            raise ValueError("Z second dimension must match freq_hz length.")
    else:
        raise ValueError("Z must be 1D or 2D.")
    if not np.isfinite(Z.real).all() or not np.isfinite(Z.imag).all():
        raise ValueError("Z must be finite.")


def _extract_response_features(
    freq_hz: np.ndarray,
    Z: np.ndarray,
    config: NoveltyConfig,
) -> Tuple[np.ndarray, dict[str, Any]]:
    freq_hz = np.asarray(freq_hz, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    _validate_response_inputs(freq_hz, Z)

    if Z.ndim == 1:
        Z_samples = [Z]
    else:
        Z_samples = [Z[idx] for idx in range(Z.shape[0])]

    feature_list: list[np.ndarray] = []
    cpe_residuals: list[float] = []
    cpe_alphas: list[float] = []
    cpe_calphas: list[float] = []
    cpe_rmse_logmag: list[float] = []
    cpe_rmse_phase: list[float] = []

    for sample in Z_samples:
        features, fit = _extract_single_response(freq_hz, sample, config)
        feature_list.append(features)
        cpe_residuals.append(fit.residual)
        cpe_alphas.append(fit.alpha)
        cpe_calphas.append(fit.c_alpha)
        cpe_rmse_logmag.append(fit.rmse_logmag)
        cpe_rmse_phase.append(fit.rmse_phase_deg)

    stacked = np.stack(feature_list, axis=0)
    mean_features = np.mean(stacked, axis=0)
    if config.include_mc_variability:
        variability = float(np.mean(np.std(stacked, axis=0)))
        mean_features = np.concatenate(
            [mean_features, np.array([variability], dtype=np.float64)]
        )
    else:
        variability = 0.0

    diagnostics = {
        "cpe_alpha": float(np.mean(cpe_alphas)),
        "cpe_c_alpha": float(np.mean(cpe_calphas)),
        "cpe_rmse_logmag": float(np.mean(cpe_rmse_logmag)),
        "cpe_rmse_phase_deg": float(np.mean(cpe_rmse_phase)),
        "cpe_distance": float(np.mean(cpe_residuals)),
        "mc_variability": float(variability),
    }

    expected = response_feature_dim(config)
    if mean_features.size != expected:
        raise ValueError(f"Response feature size {mean_features.size} != expected {expected}.")
    return mean_features.astype(np.float64), diagnostics


def _extract_single_response(
    freq_hz: np.ndarray, Z: np.ndarray, config: NoveltyConfig
) -> Tuple[np.ndarray, CPEBaselineFit]:
    log_freq = np.log10(freq_hz)
    magnitude = np.abs(Z)
    if np.any(magnitude <= 0.0):
        raise ValueError("Z magnitude must be positive.")
    log_mag = np.log10(magnitude)
    phase = np.unwrap(np.angle(Z))
    alpha = -np.gradient(log_mag, log_freq)

    grid = np.linspace(log_freq.min(), log_freq.max(), config.response_grid_size)
    log_mag_s = np.interp(grid, log_freq, log_mag)
    phase_s = np.interp(grid, log_freq, phase)
    alpha_s = np.interp(grid, log_freq, alpha)

    fit = fit_cpe_baseline(freq_hz, Z)
    log_c_alpha = np.log10(fit.c_alpha)

    parts: list[np.ndarray] = [log_mag_s]
    if config.include_phase:
        parts.append(phase_s)
    if config.include_alpha:
        parts.append(alpha_s)
    parts.append(
        np.array([fit.alpha, log_c_alpha, fit.residual], dtype=np.float64)
    )
    return np.concatenate(parts).astype(np.float64), fit


def _unpack_design(
    design_or_eval: object,
) -> Tuple[object, np.ndarray, np.ndarray, Optional[str]]:
    if isinstance(design_or_eval, tuple):
        if len(design_or_eval) == 3:
            graph, freq_hz, Z = design_or_eval
            return graph, np.asarray(freq_hz), np.asarray(Z), None
        if len(design_or_eval) == 2:
            graph, sweep = design_or_eval
            freq_hz = getattr(sweep, "freqs_hz", None)
            Z = getattr(sweep, "Z", None)
            if freq_hz is None or Z is None:
                raise ValueError("Sweep must provide freqs_hz and Z.")
            return graph, np.asarray(freq_hz), np.asarray(Z), None
        raise ValueError("Tuple inputs must be (graph, freq_hz, Z) or (graph, sweep).")

    if isinstance(design_or_eval, dict):
        graph = design_or_eval.get("graph")
        freq_hz = design_or_eval.get("freq_hz", design_or_eval.get("freqs_hz"))
        Z = design_or_eval.get("Z")
        design_id = design_or_eval.get("design_id")
        if graph is None or freq_hz is None or Z is None:
            raise ValueError("Design dict must provide graph, freq_hz, and Z.")
        return graph, np.asarray(freq_hz), np.asarray(Z), design_id

    graph = getattr(design_or_eval, "graph", None)
    if graph is None:
        raise ValueError("design_or_eval must provide a graph.")
    freq_hz = getattr(design_or_eval, "freq_hz", None)
    if freq_hz is None:
        freq_hz = getattr(design_or_eval, "freqs_hz", None)
    Z = getattr(design_or_eval, "Z", None)
    if freq_hz is None or Z is None:
        raise ValueError("design_or_eval must include freq_hz/freqs_hz and Z.")
    design_id = getattr(design_or_eval, "design_id", None)
    if design_id is None and hasattr(design_or_eval, "metadata"):
        design_id = design_or_eval.metadata.get("design_id")
    return graph, np.asarray(freq_hz), np.asarray(Z), design_id


def _knn(
    matrix: np.ndarray,
    design_ids: Sequence[str],
    vector: np.ndarray,
    k: int,
    metric: str,
    exclude_id: Optional[str],
) -> Tuple[np.ndarray, list[str]]:
    if matrix.size == 0:
        return np.array([], dtype=np.float64), []
    distances = _compute_distances(matrix, vector, metric)
    if exclude_id is not None:
        for idx, design_id in enumerate(design_ids):
            if design_id == exclude_id:
                distances[idx] = np.inf
    if not np.isfinite(distances).any():
        return np.array([], dtype=np.float64), []
    k = min(k, distances.size)
    indices = np.argpartition(distances, k - 1)[:k]
    sorted_idx = indices[np.argsort(distances[indices])]
    return distances[sorted_idx], [design_ids[idx] for idx in sorted_idx]


def _compute_distances(
    matrix: np.ndarray, vector: np.ndarray, metric: str
) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).ravel()
    matrix = np.asarray(matrix, dtype=np.float64)
    if metric == "l2":
        diff = matrix - vector
        return np.linalg.norm(diff, axis=1)
    if metric == "cosine":
        vec_norm = np.linalg.norm(vector)
        mat_norm = np.linalg.norm(matrix, axis=1)
        denom = mat_norm * vec_norm
        dot = matrix @ vector
        cos = np.zeros_like(dot, dtype=np.float64)
        mask = denom > 0
        cos[mask] = dot[mask] / denom[mask]
        dist = 1.0 - cos
        if vec_norm == 0.0:
            dist = np.where(mat_norm == 0.0, 0.0, 1.0)
        return dist
    raise ValueError(f"Unknown distance metric: {metric}")


def _compute_knn_stats(matrix: np.ndarray, config: NoveltyConfig) -> dict[str, Any]:
    n = matrix.shape[0]
    if n < 2:
        return {
            "mean": 0.0,
            "std": 1.0,
            "metric": config.distance_metric,
            "k": config.knn_k,
            "aggregation": config.knn_aggregation,
            "n": n,
        }
    rng = np.random.default_rng(config.stats_seed)
    sample_size = min(n, config.stats_sample_size)
    if sample_size < n:
        indices = rng.choice(n, size=sample_size, replace=False)
    else:
        indices = np.arange(n)
    k = min(config.knn_k, n - 1)
    aggregates = []
    for idx in indices:
        distances = _compute_distances(matrix, matrix[idx], config.distance_metric)
        distances[idx] = np.inf
        nearest = np.partition(distances, k - 1)[:k]
        aggregates.append(_aggregate_knn(nearest, config.knn_aggregation))
    aggregates = np.asarray(aggregates, dtype=np.float64)
    mean = float(np.mean(aggregates))
    std = float(np.std(aggregates))
    if std <= 1e-12:
        std = 1.0
    return {
        "mean": mean,
        "std": std,
        "metric": config.distance_metric,
        "k": config.knn_k,
        "aggregation": config.knn_aggregation,
        "n": n,
    }


def _aggregate_knn(values: np.ndarray, mode: str) -> float:
    if values.size == 0:
        return 0.0
    if mode == "min":
        return float(np.min(values))
    return float(np.mean(values))


def _normalize_distance(value: float, stats: dict[str, Any], config: NoveltyConfig) -> float:
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std", 1.0))
    if std <= 1e-12:
        z = 0.0
    else:
        z = (value - mean) / std
    clip = config.zscore_clip
    if clip > 0.0:
        z = float(np.clip(z, -clip, clip))
        return (z + clip) / (2.0 * clip)
    return 0.5


def _baseline_score(distance: float, config: NoveltyConfig) -> float:
    return float(distance / (distance + config.cpe_distance_scale))
