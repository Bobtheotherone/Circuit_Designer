"""PRIMA evaluator with optional hierarchical reduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import time

import numpy as np
import scipy.sparse as sp

from fidp.circuits import CircuitIR, Port, Resistor, Capacitor, Inductor
from fidp.circuits.ir import PortDef, SubCircuit
from fidp.circuits.ops import flatten_circuit
from fidp.data import DescriptorSystem, ReducedDescriptorSystem
from fidp.errors import CircuitIRValidationError, ReductionError
from fidp.evaluators.mna.descriptor import assemble_descriptor_system
from fidp.evaluators.mor.prima import (
    PrimaConfig,
    PrimaDiagnostics,
    prima_reduce_adaptive,
    evaluate_impedance_reduced,
    reduced_to_state_space,
)
from fidp.evaluators.passivity import check_state_space_passivity, check_impedance_passivity
from fidp.evaluators.types import EvalError, EvalRequest, EvalResult


@dataclass(frozen=True)
class MorOptions:
    """Configuration for MOR evaluation."""

    prima_config: PrimaConfig = field(default_factory=PrimaConfig)
    validation_points: int = 16
    hierarchical: bool = False
    subcircuit_max_order: int = 6


class PrimaEvaluator:
    """Evaluator using PRIMA model order reduction."""

    def __init__(self, options: Optional[MorOptions] = None) -> None:
        self.options = options or MorOptions()

    def evaluate(self, circuit: CircuitIR, request: EvalRequest) -> EvalResult:
        freqs = request.grid.make_grid()
        start = time.perf_counter()
        try:
            circuit.validate()
        except CircuitIRValidationError as exc:
            return _error_result(freqs, "circuit_invalid", str(exc), {})
        try:
            ports = _resolve_ports(circuit, request.ports)
        except CircuitIRValidationError as exc:
            return _error_result(freqs, "port_resolution_failed", str(exc), {})

        try:
            if self.options.hierarchical and circuit.subcircuits:
                system = _assemble_hierarchical_descriptor(circuit, ports, request.value_mode, self.options)
            else:
                flat = flatten_circuit(circuit, max_depth=request.max_depth)
                if flat.subcircuits:
                    raise CircuitIRValidationError("MOR requires full flattening without subcircuits.")
                system = _assemble_descriptor(flat, ports, request.value_mode)
        except (CircuitIRValidationError, ReductionError) as exc:
            return _error_result(freqs, "mor_assembly_failed", str(exc), {})

        validation_freqs = _validation_freqs(request.grid, self.options.validation_points)
        try:
            reduced, diagnostics = prima_reduce_adaptive(system, self.options.prima_config, validation_freqs)
        except ReductionError as exc:
            return _error_result(freqs, "mor_reduction_failed", str(exc), {})

        try:
            sweep = evaluate_impedance_reduced(reduced, freqs)
        except ReductionError as exc:
            return _error_result(freqs, "mor_eval_failed", str(exc), {})

        passivity_report = check_impedance_passivity(freqs, sweep.Z)
        state_space_report = check_state_space_passivity(reduced_to_state_space(reduced))

        meta = {
            "prima": {
                "orders": diagnostics.orders,
                "max_rel_errors": diagnostics.max_rel_errors,
                "selected_order": diagnostics.selected_order,
                "converged": diagnostics.converged,
            },
            "passivity": {
                "grid": {
                    "min_eig": passivity_report.min_eig,
                    "worst_freq_hz": passivity_report.worst_freq_hz,
                    "n_violations": passivity_report.n_violations,
                },
                "state_space": state_space_report.to_json_dict(),
            },
        }

        if not passivity_report.is_passive or not state_space_report.is_passive:
            return EvalResult(
                freqs_hz=freqs,
                Z=sweep.Z,
                status="error",
                errors=[
                    EvalError(
                        code="passivity_violation",
                        message="Reduced model failed passivity checks.",
                        details={
                            "grid_min_eig": passivity_report.min_eig,
                            "state_space_max_eig": state_space_report.max_eig,
                        },
                    )
                ],
                meta=meta,
                timing_s={"total": time.perf_counter() - start},
            )

        return EvalResult(
            freqs_hz=freqs,
            Z=sweep.Z,
            status="ok",
            meta=meta,
            timing_s={"total": time.perf_counter() - start},
        )


def _resolve_ports(circuit: CircuitIR, ports: Optional[list[str]]) -> list[PortDef]:
    by_name = {port.name: port for port in circuit.ports}
    if ports is None:
        return list(circuit.ports)
    resolved = []
    for name in ports:
        if name not in by_name:
            raise CircuitIRValidationError(f"Unknown port: {name}")
        resolved.append(by_name[name])
    return resolved


def _assemble_descriptor(circuit: CircuitIR, ports: list[PortDef], value_mode: str) -> DescriptorSystem:
    graph = _circuit_ir_to_graph(circuit, value_mode, ports)
    return assemble_descriptor_system(graph, [_portdef_to_port(port) for port in ports])


def _assemble_hierarchical_descriptor(
    circuit: CircuitIR,
    ports: list[PortDef],
    value_mode: str,
    options: MorOptions,
) -> DescriptorSystem:
    base = _assemble_descriptor(circuit, ports, value_mode)
    if not circuit.subcircuits:
        return base

    macros = []
    for sub in circuit.subcircuits:
        reduced = _reduce_subcircuit(sub, value_mode, options)
        macros.append((sub, reduced))

    return _inject_macromodels(base, macros)


def _reduce_subcircuit(sub: SubCircuit, value_mode: str, options: MorOptions) -> ReducedDescriptorSystem:
    if len(sub.circuit.ports) != 1:
        raise ReductionError(f"Subcircuit {sub.name} must be one-port for MOR.")
    flat = flatten_circuit(sub.circuit)
    port = flat.ports[0]
    system = _assemble_descriptor(flat, [port], value_mode)
    config = PrimaConfig(
        min_order=min(options.subcircuit_max_order, options.prima_config.min_order),
        max_order=options.subcircuit_max_order,
        order_step=options.prima_config.order_step,
        expansion_points_s0=options.prima_config.expansion_points_s0,
        target_rel_error=options.prima_config.target_rel_error,
        tol=options.prima_config.tol,
    )
    reduced, _ = prima_reduce_adaptive(system, config, np.array([1e3]))
    return reduced


def _inject_macromodels(
    base: DescriptorSystem,
    macros: list[tuple[SubCircuit, ReducedDescriptorSystem]],
) -> DescriptorSystem:
    n_base = base.G.shape[0]
    n_macro = len(macros)
    n_states = sum(reduced.G_r.shape[0] for _, reduced in macros)
    n_total = n_base + n_macro + n_states

    G = sp.lil_matrix((n_total, n_total), dtype=float)
    C = sp.lil_matrix((n_total, n_total), dtype=float)
    G[:n_base, :n_base] = base.G
    C[:n_base, :n_base] = base.C

    B = np.zeros((n_total, base.B.shape[1]), dtype=float)
    L = np.zeros((n_total, base.L.shape[1]), dtype=float)
    B[:n_base, :] = base.B
    L[:n_base, :] = base.L

    node_index = base.meta.get("node_index", {})
    state_offset = 0

    for macro_idx, (sub, reduced) in enumerate(macros):
        current_idx = n_base + macro_idx
        state_start = n_base + n_macro + state_offset
        state_offset += reduced.G_r.shape[0]

        port_name = sub.circuit.ports[0].name
        conn = sub.port_map[port_name]
        pos_idx = node_index.get(conn.pos)
        neg_idx = node_index.get(conn.neg)
        if pos_idx is None and neg_idx is None:
            raise ReductionError(f"Subcircuit {sub.name} port nodes missing in base circuit.")

        # KCL contributions at port nodes.
        if pos_idx is not None:
            G[pos_idx, current_idx] += 1.0
        if neg_idx is not None:
            G[neg_idx, current_idx] -= 1.0

        # Port voltage constraint: v(pos) - v(neg) - L_r^T x = 0.
        if pos_idx is not None:
            G[current_idx, pos_idx] += 1.0
        if neg_idx is not None:
            G[current_idx, neg_idx] -= 1.0
        for k in range(reduced.L_r.shape[0]):
            G[current_idx, state_start + k] -= reduced.L_r[k, 0]

        # Internal macromodel equations.
        G[state_start:state_start + reduced.G_r.shape[0], state_start:state_start + reduced.G_r.shape[1]] = reduced.G_r
        C[state_start:state_start + reduced.C_r.shape[0], state_start:state_start + reduced.C_r.shape[1]] = reduced.C_r
        for k in range(reduced.B_r.shape[0]):
            G[state_start + k, current_idx] -= reduced.B_r[k, 0]

    meta = dict(base.meta)
    meta["macromodels"] = len(macros)
    return DescriptorSystem(G=G.tocsc(), C=C.tocsc(), B=B, L=L, meta=meta)


def _validation_freqs(grid, count: int) -> np.ndarray:
    freqs = grid.make_grid()
    if freqs.size <= count:
        return freqs
    idx = np.linspace(0, freqs.size - 1, count, dtype=int)
    return freqs[idx]


def _circuit_ir_to_graph(circuit: CircuitIR, value_mode: str, ports: list[PortDef]):
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


def _choose_ground(nodes: set[str], ports: list[PortDef]) -> str:
    for preferred in ("0", "gnd", "GND"):
        if preferred in nodes:
            return preferred
    return ports[0].neg


def _portdef_to_port(port: PortDef) -> Port:
    return Port(pos=port.pos, neg=port.neg)


def _error_result(freqs: np.ndarray, code: str, message: str, details: dict[str, Any]) -> EvalResult:
    Z = np.full_like(freqs, np.nan, dtype=complex)
    return EvalResult(
        freqs_hz=freqs,
        Z=Z,
        status="error",
        errors=[EvalError(code=code, message=message, details=details)],
    )
