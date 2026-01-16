import pytest

from fidp.circuits.ir import CircuitIR, Component, ParamValue, PortDef
from fidp.circuits.ir_export import export_spice_netlist, lint_spice_netlist
from fidp.errors import CircuitIRValidationError


def test_spice_export_lint_passes():
    components = [
        Component("R1", "R", "n1", "n0", ParamValue(100.0)),
        Component("C1", "C", "n1", "n0", ParamValue(1e-6)),
    ]
    ports = [PortDef("P", "n1", "n0")]
    circuit = CircuitIR(name="simple", ports=ports, components=components)
    netlist = export_spice_netlist(circuit, title="simple")
    assert lint_spice_netlist(netlist) == []


def test_spice_export_deterministic_ordering():
    components_a = [
        Component("R1", "R", "a", "b", ParamValue(100.0)),
        Component("C1", "C", "b", "a", ParamValue(1e-6)),
    ]
    ports_a = [PortDef("P", "a", "b")]
    circuit_a = CircuitIR(name="a", ports=ports_a, components=components_a)

    components_b = [
        Component("C1", "C", "x", "y", ParamValue(1e-6)),
        Component("R1", "R", "y", "x", ParamValue(100.0)),
    ]
    ports_b = [PortDef("P", "x", "y")]
    circuit_b = CircuitIR(name="b", ports=ports_b, components=components_b)

    netlist_a = export_spice_netlist(circuit_a)
    netlist_b = export_spice_netlist(circuit_b)
    assert netlist_a == netlist_b


def test_spice_export_invalid_ir_raises():
    components = [Component("R1", "R", "n1", "n0", ParamValue(100.0))]
    ports = [PortDef("P", "n2", "n3")]
    circuit = CircuitIR(name="bad", ports=ports, components=components)
    with pytest.raises(CircuitIRValidationError):
        export_spice_netlist(circuit)
