"""CLI for novelty scoring."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fidp.circuits.core import Port
from fidp.metrics.novelty import (
    NoveltyConfig,
    NoveltyCorpus,
    NoveltyMetrics,
    score_novelty,
)
from fidp.search.features import CircuitGraph, Component


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FIDP novelty scoring CLI.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--corpus", required=True, help="Novelty corpus path (.npz).")
    parser.add_argument("--config", help="Optional JSON config override.")
    parser.add_argument(
        "--update-corpus",
        action="store_true",
        help="Append scored designs to the novelty corpus.",
    )
    return parser.parse_args(argv)


def _load_config(path: Optional[str]) -> NoveltyConfig:
    if not path:
        return NoveltyConfig()
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return NoveltyConfig(**data)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _read_parquet(path: str) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def _read_records(path: str) -> list[dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return _read_jsonl(path)
    if ext == ".parquet":
        return _read_parquet(path)
    raise ValueError("Input must be .jsonl or .parquet.")


def _write_jsonl(path: str, records: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _write_parquet(path: str, records: Iterable[dict[str, Any]]) -> None:
    rows = list(records)
    if not rows:
        table = pa.Table.from_pydict(
            {
                "design_id": [],
                "topology_novelty": [],
                "response_novelty": [],
                "overall_novelty": [],
            }
        )
        pq.write_table(table, path)
        return
    has_flags = any("flags" in row for row in rows)
    data = {
        "design_id": [row.get("design_id") for row in rows],
        "topology_novelty": [row.get("topology_novelty") for row in rows],
        "response_novelty": [row.get("response_novelty") for row in rows],
        "overall_novelty": [row.get("overall_novelty") for row in rows],
    }
    if has_flags:
        data["flags"] = [row.get("flags") for row in rows]
    table = pa.Table.from_pydict(data)
    pq.write_table(table, path)


def _write_records(path: str, records: Iterable[dict[str, Any]]) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        _write_jsonl(path, records)
        return
    if ext == ".parquet":
        _write_parquet(path, records)
        return
    raise ValueError("Output must be .jsonl or .parquet.")


def _parse_graph(payload: dict[str, Any]) -> CircuitGraph:
    components = []
    for comp in payload.get("components", []):
        components.append(
            Component(
                kind=str(comp["kind"]),
                node_a=str(comp["node_a"]),
                node_b=str(comp["node_b"]),
                value=comp.get("value"),
                depth=comp.get("depth"),
            )
        )
    ports = []
    for port in payload.get("ports", []):
        ports.append(Port(pos=str(port["pos"]), neg=str(port["neg"])))
    ground = str(payload.get("ground", "0"))
    return CircuitGraph(components=components, ports=ports, ground=ground)


def _parse_impedance(record: dict[str, Any]) -> np.ndarray:
    if "Z" in record:
        arr = np.asarray(record["Z"], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Z must be an array of [real, imag] pairs.")
        return arr[:, 0] + 1j * arr[:, 1]
    if "Z_real" in record and "Z_imag" in record:
        real = np.asarray(record["Z_real"], dtype=float)
        imag = np.asarray(record["Z_imag"], dtype=float)
        if real.shape != imag.shape:
            raise ValueError("Z_real and Z_imag must have the same shape.")
        return real + 1j * imag
    raise ValueError("Impedance data must include Z or Z_real/Z_imag.")


def _coerce_graph_payload(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        return json.loads(entry)
    if isinstance(entry, (bytes, bytearray)):
        return json.loads(entry.decode("utf-8"))
    if isinstance(entry, dict):
        return entry
    raise ValueError("graph must be a JSON string or dict.")


def _load_designs(path: str) -> list[dict[str, Any]]:
    records = _read_records(path)
    designs: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        design_id = record.get("design_id")
        if not design_id:
            raise ValueError(f"design_id is required for record {idx}.")
        graph_payload = record.get("graph")
        if graph_payload is None:
            raise ValueError("Each record must contain a graph payload.")
        graph_payload = _coerce_graph_payload(graph_payload)
        freq_hz = np.asarray(record.get("freq_hz"), dtype=float)
        if freq_hz.size == 0:
            raise ValueError("freq_hz must be provided.")
        Z = _parse_impedance(record)
        designs.append(
            {
                "design_id": design_id,
                "graph": _parse_graph(graph_payload),
                "freq_hz": freq_hz,
                "Z": Z,
            }
        )
    return designs


def _format_metrics(design_id: Optional[str], metrics: NoveltyMetrics) -> dict[str, Any]:
    record = {
        "design_id": design_id,
        "topology_novelty": float(metrics.topology_novelty),
        "response_novelty": float(metrics.response_novelty),
        "overall_novelty": float(metrics.overall_novelty),
    }
    flags = metrics.diagnostics.get("flags") if metrics.diagnostics else None
    if flags:
        record["flags"] = flags
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = _load_config(args.config)
    if args.update_corpus:
        config = replace(config, store_features=True)

    if os.path.exists(args.corpus):
        corpus = NoveltyCorpus.load(args.corpus, config)
    else:
        corpus = NoveltyCorpus.empty(config)

    designs = _load_designs(args.input)
    outputs: list[dict[str, Any]] = []

    for design in designs:
        metrics = score_novelty(design, corpus=corpus, config=config)
        outputs.append(_format_metrics(design.get("design_id"), metrics))
        if args.update_corpus:
            topo_vec = metrics.diagnostics.get("topology_features")
            resp_vec = metrics.diagnostics.get("response_features")
            if topo_vec is None or resp_vec is None:
                raise ValueError("Feature vectors missing from diagnostics.")
            try:
                corpus.add(design.get("design_id"), topo_vec, resp_vec)
            except ValueError:
                continue

    _write_records(args.output, outputs)
    if args.update_corpus:
        corpus.save(args.corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
