from pathlib import Path

import numpy as np
import pytest

from fidp.circuits import CircuitGraph, CircuitIR, Component, ParamValue, Port, PortDef
from fidp.evaluators.spice import (
    AcAnalysisSpec,
    export_spice_netlist,
    parse_spice_csv,
    NgSpiceRunner,
    XyceRunner,
)
from fidp.evaluators.spice.spice import _build_spice_node_map, _build_spice_node_map_graph
from fidp.errors import SpiceNotAvailableError


def test_export_spice_netlist_lines():
    circuit = CircuitGraph(ground="0")
    circuit.add_resistor("1", "0", 100.0)
    circuit.add_capacitor("1", "0", 1e-6)
    circuit.add_inductor("1", "0", 1e-3)
    port = Port(pos="1", neg="0")
    spec = AcAnalysisSpec(sweep_type="dec", points=10, f_start_hz=1.0, f_stop_hz=1e3)

    netlist = export_spice_netlist(circuit, port, spec, output_csv="out.csv", simulator="ngspice")

    assert "R1 1 0 100.0" in netlist
    assert "C2 1 0 1e-06" in netlist
    assert "L3 1 0 0.001" in netlist
    assert "IIMP 0 1 AC 1" in netlist
    assert ".ac dec 10 1.0 1000.0" in netlist
    assert "wrdata out.csv frequency v(1)" in netlist
    assert "v(0)" not in netlist


def test_spice_node_map_collision_graph():
    circuit = CircuitGraph(ground="0")
    circuit.add_resistor("a-b", "0", 10.0)
    circuit.add_resistor("a_b", "0", 20.0)
    port = Port(pos="a-b", neg="0")

    node_map = _build_spice_node_map_graph(circuit, port)

    mapped = {node: mapped for node, mapped in node_map.items() if mapped != "0"}
    assert len(set(mapped.values())) == len(mapped)
    assert node_map["a-b"] == "a_b"
    assert node_map["a_b"] == "a_b__1"


def test_spice_node_map_collision_ir_reserved_base():
    components = [
        Component("R1", "R", "a-b", "gnd", ParamValue(10.0)),
        Component("R2", "R", "a_b", "gnd", ParamValue(10.0)),
        Component("R3", "R", "a_b__1", "gnd", ParamValue(10.0)),
    ]
    ports = [PortDef("P", "a-b", "gnd")]
    circuit = CircuitIR(name="collision", ports=ports, components=components)

    node_map = _build_spice_node_map(circuit)

    mapped = {node: mapped for node, mapped in node_map.items() if mapped != "0"}
    assert len(set(mapped.values())) == len(mapped)
    assert node_map["a-b"] == "a_b"
    assert node_map["a_b"] == "a_b__2"
    assert node_map["a_b__1"] == "a_b__1"


def test_parse_spice_csv_data():
    data_path = Path(__file__).parent / "data" / "spice_output.csv"
    sweep = parse_spice_csv(data_path, Port(pos="1", neg="0"))

    assert np.allclose(sweep.freqs_hz, np.array([1.0, 10.0]))
    assert np.allclose(sweep.Z, np.array([2.0 + 3.0j, 1.0 + 0.0j]))


def test_ngspice_runner_command_and_parse(tmp_path: Path, monkeypatch):
    circuit = CircuitGraph(ground="0")
    circuit.add_resistor("1", "0", 50.0)
    port = Port(pos="1", neg="0")
    spec = AcAnalysisSpec(sweep_type="dec", points=1, f_start_hz=1.0, f_stop_hz=1.0)
    netlist = export_spice_netlist(circuit, port, spec, output_csv="spice_output.csv")

    output_path = tmp_path / "spice_output.csv"
    output_path.write_text(
        "frequency,v(1)_real,v(1)_imag\n1.0,2.0,0.0\n",
        encoding="utf-8",
    )

    called = {}

    def fake_run(cmd, cwd=None, capture_output=None, text=None, timeout=None):
        called["cmd"] = cmd
        class Result:
            returncode = 0
            stderr = ""
            stdout = ""
        return Result()

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ngspice")
    monkeypatch.setattr("subprocess.run", fake_run)

    runner = NgSpiceRunner()
    sweep = runner.run(netlist, port, tmp_path)

    assert "-b" in called["cmd"]
    assert any(str(tmp_path / "circuit.cir") in str(item) for item in called["cmd"])
    assert np.allclose(sweep.Z, np.array([2.0 + 0.0j]))


def test_xyce_command_construction():
    runner = XyceRunner(executable="Xyce")
    cmd = runner.build_command(Path("/tmp/netlist.cir"), "output.csv")
    assert cmd == ["/tmp/netlist.cir"]


def test_missing_binary_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shutil.which", lambda name: None)
    runner = NgSpiceRunner()
    with pytest.raises(SpiceNotAvailableError):
        runner.run("* netlist\n.end\n", Port(pos="1", neg="0"), tmp_path)
