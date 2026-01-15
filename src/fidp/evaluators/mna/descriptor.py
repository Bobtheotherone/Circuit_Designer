"""Sparse MNA descriptor assembly and impedance evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings

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


def assemble_descriptor_system(circuit: CircuitGraph, port: Port) -> DescriptorSystem:
    """
    Assemble descriptor-form system (G + sC) x = B.

    Sign convention:
    - Port current injection is +1 A flowing from port.pos -> port.neg.
    - Port voltage is V(pos) - V(neg).
    - Impedance Z(s) = L^T x where x solves (G + sC) x = B.
    """
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
    B = np.zeros((n_unknowns, 1), dtype=float)
    L = np.zeros((n_unknowns, 1), dtype=float)

    pos_idx = node_idx(port.pos)
    neg_idx = node_idx(port.neg)

    if pos_idx is not None:
        B[pos_idx, 0] += 1.0
        L[pos_idx, 0] += 1.0
    if neg_idx is not None:
        B[neg_idx, 0] -= 1.0
        L[neg_idx, 0] -= 1.0

    meta = {
        "ground": circuit.ground,
        "node_index": node_index.node_to_index,
        "inductor_count": n_inductors,
        "port": port,
        "sign_convention": "Z = V(pos) - V(neg) for +1A from pos to neg",
    }

    return DescriptorSystem(G=G.tocsc(), C=C.tocsc(), B=B, L=L, meta=meta)


def evaluate_impedance_descriptor(system: DescriptorSystem, freqs_hz: np.ndarray) -> ImpedanceSweep:
    """
    Evaluate impedance sweep from descriptor system.

    Z(s) = L^T (G + sC)^{-1} B where B encodes a +1A injection.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    Z = np.zeros_like(freqs_hz, dtype=complex)
    G = system.G
    C = system.C
    B = system.B
    L = system.L

    for idx, freq in enumerate(freqs_hz):
        s = 1j * 2.0 * np.pi * freq
        A = G + s * C
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", spla.MatrixRankWarning)
                x = spla.spsolve(A, B[:, 0])
        except (spla.MatrixRankWarning, RuntimeError, ValueError) as exc:
            raise SingularCircuitError("Descriptor system is singular.") from exc
        if np.any(~np.isfinite(x)):
            raise SingularCircuitError("Descriptor system solution is invalid.")
        Z[idx] = L[:, 0].T @ x

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
