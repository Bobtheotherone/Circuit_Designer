import json
from pathlib import Path

import numpy as np

from fidp.circuits import CircuitGraph, Port
from fidp.evaluators.mna import evaluate_impedance_mna


def test_mna_golden_fixture() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "rc_series_gold.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    freqs = np.array(payload["freqs_hz"], dtype=float)
    expected = np.array(payload["z_real"], dtype=float) + 1j * np.array(payload["z_imag"], dtype=float)

    circuit = CircuitGraph(ground="0")
    circuit.add_resistor("1", "2", 75.0)
    circuit.add_capacitor("2", "0", 2e-6)
    port = Port(pos="1", neg="0")

    sweep = evaluate_impedance_mna(circuit, port, freqs)
    assert np.allclose(sweep.Z, expected, rtol=1e-6, atol=1e-9)
