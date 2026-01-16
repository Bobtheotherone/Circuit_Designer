import numpy as np

from fidp.circuits import CircuitGraph, Port
from fidp.evaluators.mna import assemble_descriptor_system, evaluate_impedance_descriptor
from fidp.evaluators.mor import (
    PrimaConfig,
    prima_reduce_adaptive,
    evaluate_impedance_reduced,
    reduced_to_state_space,
)
from fidp.evaluators.passivity import check_impedance_passivity, check_state_space_passivity


def _build_rc_ladder(stages: int, r: float, c: float) -> CircuitGraph:
    circuit = CircuitGraph(ground="0")
    for idx in range(stages):
        node_a = str(idx + 1)
        node_b = str(idx + 2)
        circuit.add_resistor(node_a, node_b, r)
        circuit.add_capacitor(node_b, "0", c)
    return circuit


def test_prima_passivity_and_error() -> None:
    circuit = _build_rc_ladder(stages=10, r=25.0, c=1e-6)
    port = Port(pos="1", neg="0")
    system = assemble_descriptor_system(circuit, port)

    freqs = np.logspace(1, 4, 20)
    config = PrimaConfig(min_order=4, max_order=10, order_step=2, target_rel_error=0.2)

    reduced, diagnostics = prima_reduce_adaptive(system, config, freqs)
    reduced_sweep = evaluate_impedance_reduced(reduced, freqs)
    full = evaluate_impedance_descriptor(system, freqs)

    rel_err = np.abs(reduced_sweep.Z - full.Z) / np.maximum(np.abs(full.Z), 1e-12)
    assert np.max(rel_err) <= diagnostics.max_rel_errors[-1] + 1e-9

    passivity = check_impedance_passivity(freqs, reduced_sweep.Z)
    state_space = check_state_space_passivity(reduced_to_state_space(reduced))

    assert passivity.is_passive
    assert state_space.is_passive
