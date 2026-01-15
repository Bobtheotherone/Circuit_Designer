import numpy as np
import pytest

from fidp.circuits import CircuitGraph, Port
from fidp.evaluators.mna import evaluate_impedance_mna
from fidp.errors import SingularCircuitError


def test_resistor_to_ground_impedance_sign():
    circuit = CircuitGraph(ground="0")
    circuit.add_resistor("1", "0", 100.0)
    port = Port(pos="1", neg="0")
    freqs = np.array([1.0, 10.0, 100.0])

    sweep = evaluate_impedance_mna(circuit, port, freqs)

    assert np.allclose(sweep.Z, 100.0, rtol=1e-6, atol=1e-9)
    assert np.all(sweep.Z.real > 0.0)


def test_parallel_rc_to_ground():
    circuit = CircuitGraph(ground="0")
    circuit.add_resistor("1", "0", 50.0)
    circuit.add_capacitor("1", "0", 1e-6)
    port = Port(pos="1", neg="0")
    freqs = np.array([10.0, 100.0, 1000.0])
    s = 1j * 2.0 * np.pi * freqs
    expected = 1.0 / (1.0 / 50.0 + s * 1e-6)

    sweep = evaluate_impedance_mna(circuit, port, freqs)

    assert np.allclose(sweep.Z, expected, rtol=1e-5, atol=1e-8)


def test_series_rc_to_ground():
    circuit = CircuitGraph(ground="0")
    circuit.add_resistor("1", "2", 75.0)
    circuit.add_capacitor("2", "0", 2e-6)
    port = Port(pos="1", neg="0")
    freqs = np.array([5.0, 50.0, 500.0])
    s = 1j * 2.0 * np.pi * freqs
    expected = 75.0 + 1.0 / (s * 2e-6)

    sweep = evaluate_impedance_mna(circuit, port, freqs)

    assert np.allclose(sweep.Z, expected, rtol=1e-5, atol=1e-8)


def test_series_rlc_to_ground():
    circuit = CircuitGraph(ground="0")
    circuit.add_resistor("1", "2", 10.0)
    circuit.add_inductor("2", "3", 1e-3)
    circuit.add_capacitor("3", "0", 5e-6)
    port = Port(pos="1", neg="0")
    freqs = np.array([100.0, 1000.0, 5000.0])
    s = 1j * 2.0 * np.pi * freqs
    expected = 10.0 + s * 1e-3 + 1.0 / (s * 5e-6)

    sweep = evaluate_impedance_mna(circuit, port, freqs)

    assert np.allclose(sweep.Z, expected, rtol=1e-4, atol=1e-6)


def test_disconnected_circuit_raises():
    circuit = CircuitGraph(ground="0")
    circuit.add_resistor("1", "2", 10.0)
    port = Port(pos="1", neg="0")
    freqs = np.array([100.0])

    with pytest.raises(SingularCircuitError):
        evaluate_impedance_mna(circuit, port, freqs)
