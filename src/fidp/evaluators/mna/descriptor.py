"""Sparse MNA descriptor assembly and impedance evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fidp.circuits import CircuitGraph, Port, Resistor, Capacitor, Inductor
from fidp.data import DescriptorSystem, ImpedanceSweep
from fidp.errors import CircuitValidationError, SingularCircuitError


@dataclass
class _NodeIndex:
    node_to_index: Dict[str, int]
    index_to_node: List[str]


def _build_node_index(circuit: CircuitGraph) -> _NodeIndex:
    if circuit.ground not in circuit.nodes:
        raise CircuitValidationError("Ground node is not registered in circuit.")
    nodes = sorted(node for node in circuit.nodes if node != circuit.ground)
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    return _NodeIndex(node_to_index=node_to_index, index_to_node=nodes)


def assemble_descriptor_system(
    circuit: CircuitGraph,
    ports: Port | Sequence[Port],
) -> DescriptorSystem:
    """
    Assemble descriptor-form system (G + sC) x = B for one or more ports.

    Sign convention:
    - Port current is +1 A entering at port.pos and leaving at port.neg.
    - Port voltage is V(port.pos) - V(port.neg).
    - Impedance Z(s) = V(pos) - V(neg) = L^T x where x solves (G + sC) x = B.
    """
    if isinstance(ports, Port):
        ports = [ports]
    ports = list(ports)
    if not ports:
        raise CircuitValidationError("At least one port is required.")
    for port in ports:
        if port.pos not in circuit.nodes or port.neg not in circuit.nodes:
            raise CircuitValidationError("Port nodes must exist in circuit.")

    node_index = _build_node_index(circuit)
    n_nodes = len(node_index.index_to_node)
    inductors = [comp for comp in circuit.iter_components() if isinstance(comp, Inductor)]
    n_inductors = len(inductors)
    n_unknowns = n_nodes + n_inductors

    G = sp.lil_matrix((n_unknowns, n_unknowns), dtype=float)
    C = sp.lil_matrix((n_unknowns, n_unknowns), dtype=float)

    def node_idx(node: str) -> int | None:
        return node_index.node_to_index.get(node)

    def stamp_admittance(mat: sp.lil_matrix, value: float, node_a: str, node_b: str) -> None:
        idx_a = node_idx(node_a)
        idx_b = node_idx(node_b)
        if idx_a is None and idx_b is None:
            return
        if idx_a is not None:
            mat[idx_a, idx_a] += value
        if idx_b is not None:
            mat[idx_b, idx_b] += value
        if idx_a is not None and idx_b is not None:
            mat[idx_a, idx_b] -= value
            mat[idx_b, idx_a] -= value

    # Stamp passive elements.
    for comp in circuit.iter_components():
        if isinstance(comp, Resistor):
            conductance = 1.0 / comp.resistance_ohms
            stamp_admittance(G, conductance, comp.node_a, comp.node_b)
        elif isinstance(comp, Capacitor):
            stamp_admittance(C, comp.capacitance_f, comp.node_a, comp.node_b)

    # Inductors add current variables and KVL rows.
    for idx, ind in enumerate(inductors):
        current_idx = n_nodes + idx
        idx_a = node_idx(ind.node_a)
        idx_b = node_idx(ind.node_b)

        # KCL contributions with current defined from node_a -> node_b.
        if idx_a is not None:
            G[idx_a, current_idx] += 1.0
        if idx_b is not None:
            G[idx_b, current_idx] -= 1.0

        # Constitutive equation: v(a) - v(b) - s*L*i = 0.
        if idx_a is not None:
            G[current_idx, idx_a] += 1.0
        if idx_b is not None:
            G[current_idx, idx_b] -= 1.0
        C[current_idx, current_idx] -= ind.inductance_h

    # Build excitation and observation matrices.
    n_ports = len(ports)
    B = np.zeros((n_unknowns, n_ports), dtype=float)
    L = np.zeros((n_unknowns, n_ports), dtype=float)

    for col, port in enumerate(ports):
        pos_idx = node_idx(port.pos)
        neg_idx = node_idx(port.neg)

        # Inject +1 A into port.pos and extract 1 A from port.neg.
        if pos_idx is not None:
            B[pos_idx, col] += 1.0
            L[pos_idx, col] += 1.0
        if neg_idx is not None:
            B[neg_idx, col] -= 1.0
            L[neg_idx, col] -= 1.0

    meta = {
        "ground": circuit.ground,
        "node_index": node_index.node_to_index,
        "inductor_count": n_inductors,
        "ports": ports,
        "n_ports": n_ports,
        "sign_convention": "Z = V(pos) - V(neg) for +1A entering pos and leaving neg",
    }

    return DescriptorSystem(G=G.tocsc(), C=C.tocsc(), B=B, L=L, meta=meta)


def evaluate_impedance_descriptor(system: DescriptorSystem, freqs_hz: np.ndarray) -> ImpedanceSweep:
    """
    Evaluate impedance sweep from descriptor system.

    Z(s) = L^T (G + sC)^{-1} B where B encodes the +1A port current.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    n_ports = system.B.shape[1] if system.B.ndim == 2 else 1
    if n_ports == 1:
        Z = np.zeros_like(freqs_hz, dtype=complex)
    else:
        Z = np.zeros((freqs_hz.size, n_ports, n_ports), dtype=complex)
    G = system.G
    C = system.C
    B = system.B
    L = system.L

    for idx, freq in enumerate(freqs_hz):
        s = 1j * 2.0 * np.pi * freq
        A = G + s * C
        try:
            lu = spla.splu(A.tocsc())
            x = lu.solve(B)
        except Exception as exc:
            raise SingularCircuitError("Descriptor system is singular.") from exc
        if np.any(~np.isfinite(x)):
            raise SingularCircuitError("Descriptor system solution is invalid.")
        Z_slice = L.T @ x
        if n_ports == 1:
            Z[idx] = Z_slice[0, 0]
        else:
            Z[idx] = Z_slice

    meta = {"descriptor_meta": system.meta}
    return ImpedanceSweep(freqs_hz=freqs_hz, Z=Z, meta=meta)


def evaluate_impedance_mna(
    circuit: CircuitGraph,
    port: Port,
    freqs_hz: np.ndarray,
) -> ImpedanceSweep:
    """Convenience wrapper: assemble descriptor system and evaluate impedance."""
    system = assemble_descriptor_system(circuit, port)
    return evaluate_impedance_descriptor(system, freqs_hz)
