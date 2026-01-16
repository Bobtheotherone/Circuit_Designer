"""Sparse MNA evaluator for CircuitIR inputs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence
import time

import numpy as np
import scipy.sparse.linalg as spla

from fidp.circuits import CircuitIR, Port, Resistor, Capacitor, Inductor
from fidp.circuits.canonical import canonicalize_circuit
from fidp.circuits.ir import PortDef
from fidp.circuits.ops import flatten_circuit
from fidp.data import DescriptorSystem
from fidp.errors import CircuitIRValidationError
from fidp.evaluators.mna.descriptor import assemble_descriptor_system
from fidp.evaluators.types import EvalError, EvalRequest, EvalResult, FrequencyGridSpec


@dataclass(frozen=True)
class MnaOptions:
    """Configuration for sparse MNA evaluation."""

    permc_spec: str = "COLAMD"
    cond_threshold: float = 1e12
    max_cache_items: int = 128
    enable_diagnostics: bool = True


@dataclass
class _CacheEntry:
    key: str
    system: DescriptorSystem
    port_names: tuple[str, ...]
    metadata: dict[str, Any]


class _DescriptorCache:
    def __init__(self, max_items: int) -> None:
        self._max_items = max_items
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()

    def get(self, key: str) -> Optional[_CacheEntry]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        self._cache.move_to_end(key)
        return entry

    def set(self, key: str, entry: _CacheEntry) -> None:
        self._cache[key] = entry
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_items:
            self._cache.popitem(last=False)


class MnaEvaluator:
    """Evaluator using sparse MNA for CircuitIR circuits."""

    def __init__(self, options: Optional[MnaOptions] = None) -> None:
        self.options = options or MnaOptions()
        self._cache = _DescriptorCache(max_items=self.options.max_cache_items)

    def evaluate(self, circuit: CircuitIR, request: EvalRequest) -> EvalResult:
        start = time.perf_counter()
        freqs = request.grid.make_grid()

        try:
            circuit.validate()
        except CircuitIRValidationError as exc:
            return _error_result(freqs, "circuit_invalid", str(exc), {})

        try:
            flat = flatten_circuit(circuit, max_depth=request.max_depth)
        except CircuitIRValidationError as exc:
            return _error_result(freqs, "flatten_failed", str(exc), {"stage": "flatten"})

        if flat.subcircuits:
            return _error_result(
                freqs,
                "flatten_incomplete",
                "Circuit requires full flattening for MNA evaluation.",
                {"remaining_subcircuits": len(flat.subcircuits)},
            )

        try:
            ports = _resolve_ports(flat, request.ports)
            port_names = tuple(port.name for port in ports)
            graph = _circuit_ir_to_graph(flat, request.value_mode, ports)
        except CircuitIRValidationError as exc:
            return _error_result(freqs, "port_resolution_failed", str(exc), {"stage": "ports"})

        canonical = canonicalize_circuit(flat, mode=request.value_mode)
        cache_key = f"{canonical.canonical_hash}:{port_names}"
        entry = self._cache.get(cache_key)
        if entry is None:
            system = assemble_descriptor_system(graph, [_portdef_to_port(port) for port in ports])
            entry = _CacheEntry(
                key=cache_key,
                system=system,
                port_names=port_names,
                metadata={
                    "canonical_hash": canonical.canonical_hash,
                    "n_ports": len(ports),
                    "n_nodes": system.G.shape[0],
                    "matrix_nnz": int(system.G.nnz + system.C.nnz),
                },
            )
            self._cache.set(cache_key, entry)

        solve_start = time.perf_counter()
        Z, diag, errors = _solve_descriptor_system(entry.system, freqs, self.options)
        solve_time = time.perf_counter() - solve_start

        if errors:
            return EvalResult(
                freqs_hz=freqs,
                Z=Z,
                status="error",
                errors=errors,
                meta=entry.metadata,
                timing_s={"total": time.perf_counter() - start, "solve": solve_time},
            )

        meta = dict(entry.metadata)
        meta.update(diag)
        return EvalResult(
            freqs_hz=freqs,
            Z=Z,
            status="ok",
            meta=meta,
            timing_s={"total": time.perf_counter() - start, "solve": solve_time},
        )


def _resolve_ports(circuit: CircuitIR, ports: Optional[Sequence[str]]) -> list[PortDef]:
    by_name = {port.name: port for port in circuit.ports}
    if ports is None:
        return list(circuit.ports)
    resolved = []
    for name in ports:
        if name not in by_name:
            raise CircuitIRValidationError(f"Unknown port: {name}")
        resolved.append(by_name[name])
    return resolved


def _choose_ground(nodes: Iterable[str], ports: Sequence[PortDef]) -> str:
    candidates = [port.neg for port in ports if port.neg in nodes]
    for preferred in ("0", "gnd", "GND"):
        if preferred in nodes:
            return preferred
    if candidates:
        return candidates[0]
    return sorted(nodes)[0]


def _circuit_ir_to_graph(circuit: CircuitIR, value_mode: str, ports: Sequence[PortDef]):
    nodes = set()
    for comp in circuit.components:
        nodes.update([comp.node_a, comp.node_b])
    for port in ports:
        nodes.update([port.pos, port.neg])
    ground = _choose_ground(nodes, ports)

    components = []
    for comp in circuit.components:
        value = comp.value.resolved("snapped" if value_mode == "snapped" else "continuous")
        if comp.kind == "R":
            components.append(Resistor(comp.node_a, comp.node_b, value))
        elif comp.kind == "C":
            components.append(Capacitor(comp.node_a, comp.node_b, value))
        elif comp.kind == "L":
            components.append(Inductor(comp.node_a, comp.node_b, value))
        else:
            raise CircuitIRValidationError(f"Unsupported component kind: {comp.kind}")

    from fidp.circuits import CircuitGraph

    return CircuitGraph(ground=ground, components=components)


def _portdef_to_port(port: PortDef) -> Port:
    return Port(pos=port.pos, neg=port.neg)


def _solve_descriptor_system(
    system: DescriptorSystem,
    freqs_hz: np.ndarray,
    options: MnaOptions,
) -> tuple[np.ndarray, dict[str, Any], list[EvalError]]:
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    n_ports = system.B.shape[1]
    if n_ports == 1:
        Z = np.zeros_like(freqs_hz, dtype=complex)
    else:
        Z = np.zeros((freqs_hz.size, n_ports, n_ports), dtype=complex)

    errors: list[EvalError] = []
    cond_values: list[float] = []

    for idx, freq in enumerate(freqs_hz):
        s = 1j * 2.0 * np.pi * freq
        A = system.G + s * system.C
        try:
            lu = spla.splu(A.tocsc(), permc_spec=options.permc_spec)
            x = lu.solve(system.B)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(
                EvalError(
                    code="singular_matrix",
                    message="Sparse solve failed.",
                    details={"freq_hz": float(freq), "error": str(exc)},
                )
            )
            break

        if not np.isfinite(x).all():
            errors.append(
                EvalError(
                    code="invalid_solution",
                    message="Descriptor solve produced non-finite values.",
                    details={"freq_hz": float(freq)},
                )
            )
            break

        Z_slice = system.L.T @ x
        if n_ports == 1:
            Z[idx] = Z_slice[0, 0]
        else:
            Z[idx] = Z_slice

        if options.enable_diagnostics and A.shape[0] <= 200:
            dense = A.toarray()
            cond_values.append(float(np.linalg.cond(dense)))

    diag: dict[str, Any] = {}
    if cond_values:
        diag["cond_max"] = float(max(cond_values))
        diag["cond_min"] = float(min(cond_values))
        if diag["cond_max"] > options.cond_threshold and not errors:
            errors.append(
                EvalError(
                    code="ill_conditioned",
                    message="Condition number exceeds threshold.",
                    details={"cond_max": diag["cond_max"], "threshold": options.cond_threshold},
                )
            )

    return Z, diag, errors


def default_grid() -> FrequencyGridSpec:
    return FrequencyGridSpec(f_start_hz=1.0, f_stop_hz=1e6, points=50)


def _error_result(freqs: np.ndarray, code: str, message: str, details: dict[str, Any]) -> EvalResult:
    Z = np.full_like(freqs, np.nan, dtype=complex)
    return EvalResult(
        freqs_hz=freqs,
        Z=Z,
        status="error",
        errors=[EvalError(code=code, message=message, details=details)],
    )
