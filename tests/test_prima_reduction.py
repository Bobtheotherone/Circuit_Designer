import numpy as np

from fidp.circuits import CircuitGraph, Port
from fidp.evaluators.mna import assemble_descriptor_system, evaluate_impedance_descriptor
from fidp.evaluators.mor import prima_reduce, evaluate_impedance_reduced


def build_rc_ladder(sections: int, r: float, c: float) -> CircuitGraph:
    circuit = CircuitGraph(ground="0")
    for idx in range(sections):
        node_a = str(idx + 1)
        node_b = str(idx + 2)
        circuit.add_resistor(node_a, node_b, r)
        circuit.add_capacitor(node_b, "0", c)
    return circuit


def test_prima_matches_near_expansion_point():
    circuit = build_rc_ladder(sections=12, r=10.0, c=1e-6)
    port = Port(pos="1", neg="0")
    system = assemble_descriptor_system(circuit, port)

    f0 = 1000.0
    freqs = np.array([0.8 * f0, f0, 1.2 * f0])
    full = evaluate_impedance_descriptor(system, freqs)

    reduced = prima_reduce(system, order_r=6, expansion_point_s0=1j * 2.0 * np.pi * f0)
    reduced_sweep = evaluate_impedance_reduced(reduced, freqs)

    rel_err = np.abs(reduced_sweep.Z - full.Z) / np.maximum(np.abs(full.Z), 1e-12)
    # Bound chosen for stable agreement near expansion point for RC ladder.
    assert np.max(rel_err) < 0.1
