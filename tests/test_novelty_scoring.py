import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pytest

from fidp.circuits.core import Port
from fidp.metrics.novelty import (
    NoveltyConfig,
    NoveltyCorpus,
    extract_response_features,
    extract_topology_features,
    fit_cpe_baseline,
    response_feature_dim,
    score_novelty,
)
from fidp.novelty_score import main as novelty_main
from fidp.search.features import CircuitGraph, Component


def _make_triangle_graph(node_a: str, node_b: str, node_c: str, ground: str) -> CircuitGraph:
    components = [
        Component(kind="R", node_a=node_a, node_b=node_b, value=100.0),
        Component(kind="C", node_a=node_b, node_b=node_c, value=1e-6),
        Component(kind="L", node_a=node_c, node_b=node_a, value=1e-3),
    ]
    ports = [Port(pos=node_a, neg=ground)]
    return CircuitGraph(components=components, ports=ports, ground=ground)


def _make_cpe_response(alpha: float, c_alpha: float, n_points: int = 64) -> tuple[np.ndarray, np.ndarray]:
    freq = np.logspace(1, 4, n_points)
    omega = 2.0 * np.pi * freq
    Z = 1.0 / (c_alpha * (1j * omega) ** alpha)
    return freq, Z


def test_topology_features_permutation_invariant() -> None:
    config = NoveltyConfig()
    graph_a = _make_triangle_graph("1", "2", "3", "0")
    graph_b = _make_triangle_graph("a", "b", "c", "g")

    feats_a = extract_topology_features(graph_a, config=config)
    feats_b = extract_topology_features(graph_b, config=config)

    assert np.allclose(feats_a, feats_b)


def test_topology_distance_sanity() -> None:
    config = NoveltyConfig(distance_metric="l2", knn_k=1)
    graph_base = _make_triangle_graph("1", "2", "3", "0")
    graph_extra = _make_triangle_graph("1", "2", "3", "0")
    graph_extra.components.append(
        Component(kind="R", node_a="2", node_b="0", value=50.0)
    )

    topo_base = extract_topology_features(graph_base, config=config)
    topo_extra = extract_topology_features(graph_extra, config=config)
    resp_dummy = np.zeros(response_feature_dim(config), dtype=np.float64)

    corpus = NoveltyCorpus.empty(config)
    corpus.add("base", topo_base, resp_dummy)

    dist_same, _ = corpus.knn_topology(topo_base, k=1, metric="l2")
    dist_extra, _ = corpus.knn_topology(topo_extra, k=1, metric="l2")

    assert dist_same[0] == pytest.approx(0.0)
    assert dist_extra[0] > 0.0


def test_response_feature_determinism() -> None:
    config = NoveltyConfig()
    freq, Z = _make_cpe_response(alpha=0.65, c_alpha=5e-4, n_points=48)

    features_a = extract_response_features(freq, Z, config=config)
    features_b = extract_response_features(freq, Z, config=config)

    assert np.allclose(features_a, features_b)


def test_cpe_fit_sanity() -> None:
    alpha = 0.55
    c_alpha = 2e-4
    freq, Z = _make_cpe_response(alpha=alpha, c_alpha=c_alpha, n_points=80)

    fit = fit_cpe_baseline(freq, Z)

    assert fit.alpha == pytest.approx(alpha, rel=1e-2, abs=1e-3)
    assert fit.residual < 1e-3


def test_corpus_round_trip(tmp_path: Path) -> None:
    config = NoveltyConfig()
    graph_a = _make_triangle_graph("1", "2", "3", "0")
    graph_b = _make_triangle_graph("1", "2", "4", "0")
    freq, Z = _make_cpe_response(alpha=0.6, c_alpha=1e-3, n_points=32)

    topo_a = extract_topology_features(graph_a, config=config)
    topo_b = extract_topology_features(graph_b, config=config)
    resp = extract_response_features(freq, Z, config=config)

    corpus = NoveltyCorpus.empty(config)
    corpus.add("a", topo_a, resp)
    corpus.add("b", topo_b, resp)

    path = tmp_path / "corpus.npz"
    corpus.save(str(path))
    loaded = NoveltyCorpus.load(str(path), config)

    assert loaded.design_ids == ["a", "b"]
    assert np.allclose(loaded.topology_features, corpus.topology_features)
    assert np.allclose(loaded.response_features, corpus.response_features)

    dist_orig, ids_orig = corpus.knn_topology(topo_a, k=1, metric=config.distance_metric)
    dist_loaded, ids_loaded = loaded.knn_topology(topo_a, k=1, metric=config.distance_metric)

    assert np.allclose(dist_orig, dist_loaded)
    assert ids_orig == ids_loaded


def test_score_novelty_end_to_end_and_mismatch(tmp_path: Path) -> None:
    config = NoveltyConfig(knn_k=1)
    graph_a = _make_triangle_graph("1", "2", "3", "0")
    graph_b = _make_triangle_graph("1", "2", "4", "0")
    freq, Z = _make_cpe_response(alpha=0.7, c_alpha=8e-4, n_points=48)

    topo_a = extract_topology_features(graph_a, config=config)
    resp = extract_response_features(freq, Z, config=config)

    corpus = NoveltyCorpus.empty(config)
    corpus.add("base", topo_a, resp)

    metrics = score_novelty(
        {"design_id": "candidate", "graph": graph_b, "freq_hz": freq, "Z": Z},
        corpus=corpus,
        config=config,
    )

    assert 0.0 <= metrics.topology_novelty <= 1.0
    assert 0.0 <= metrics.response_novelty <= 1.0
    assert 0.0 <= metrics.overall_novelty <= 1.0

    path = tmp_path / "corpus.npz"
    corpus.save(str(path))
    mismatch_config = NoveltyConfig(response_grid_size=32)
    with pytest.raises(ValueError):
        NoveltyCorpus.load(str(path), mismatch_config)


def test_cli_jsonl_round_trip(tmp_path: Path) -> None:
    graph_payload = {
        "ground": "0",
        "components": [
            {"kind": "R", "node_a": "1", "node_b": "2", "value": 100.0},
            {"kind": "C", "node_a": "2", "node_b": "3", "value": 1e-6},
        ],
        "ports": [{"pos": "1", "neg": "0"}],
    }
    freq, Z = _make_cpe_response(alpha=0.6, c_alpha=1e-3, n_points=24)
    z_pairs = np.column_stack([Z.real, Z.imag]).tolist()

    records = [
        {"design_id": "d1", "graph": graph_payload, "freq_hz": freq.tolist(), "Z": z_pairs},
        {"design_id": "d2", "graph": graph_payload, "freq_hz": freq.tolist(), "Z": z_pairs},
    ]

    input_path = tmp_path / "designs.jsonl"
    output_path = tmp_path / "out.jsonl"
    corpus_path = tmp_path / "corpus.npz"

    with open(input_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    result = novelty_main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--corpus",
            str(corpus_path),
            "--update-corpus",
        ]
    )

    assert result == 0
    output_records = []
    with open(output_path, "r", encoding="utf-8") as handle:
        for line in handle:
            output_records.append(json.loads(line))

    assert len(output_records) == 2
    for rec in output_records:
        assert 0.0 <= rec["topology_novelty"] <= 1.0
        assert 0.0 <= rec["response_novelty"] <= 1.0
        assert 0.0 <= rec["overall_novelty"] <= 1.0

    loaded = NoveltyCorpus.load(str(corpus_path), NoveltyConfig())
    assert loaded.size == 2


def test_cli_parquet_round_trip(tmp_path: Path) -> None:
    graph_payload = {
        "ground": "0",
        "components": [
            {"kind": "R", "node_a": "1", "node_b": "2", "value": 50.0},
            {"kind": "L", "node_a": "2", "node_b": "3", "value": 1e-3},
        ],
        "ports": [{"pos": "1", "neg": "0"}],
    }
    freq, Z = _make_cpe_response(alpha=0.7, c_alpha=5e-4, n_points=16)
    z_pairs = np.column_stack([Z.real, Z.imag]).tolist()

    records = [
        {
            "design_id": "p1",
            "graph": json.dumps(graph_payload),
            "freq_hz": freq.tolist(),
            "Z": z_pairs,
        },
        {
            "design_id": "p2",
            "graph": json.dumps(graph_payload),
            "freq_hz": freq.tolist(),
            "Z": z_pairs,
        },
    ]

    input_path = tmp_path / "designs.parquet"
    output_path = tmp_path / "out.parquet"
    corpus_path = tmp_path / "corpus.npz"

    table = pa.Table.from_pylist(records)
    pq.write_table(table, input_path)

    result = novelty_main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--corpus",
            str(corpus_path),
            "--update-corpus",
        ]
    )

    assert result == 0

    out_table = pq.read_table(output_path)
    out_rows = out_table.to_pylist()
    assert len(out_rows) == 2
    for row in out_rows:
        assert 0.0 <= row["topology_novelty"] <= 1.0
        assert 0.0 <= row["response_novelty"] <= 1.0
        assert 0.0 <= row["overall_novelty"] <= 1.0

    loaded = NoveltyCorpus.load(str(corpus_path), NoveltyConfig())
    assert loaded.size == 2
