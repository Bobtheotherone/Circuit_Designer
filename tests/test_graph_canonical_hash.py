import random

import pytest

from fidp.circuits.canonical import DedupeIndex, are_isomorphic, canonicalize_circuit
from fidp.circuits.ir import CircuitIR, Component, ParamValue, PortConnection, PortDef, SubCircuit
from fidp.errors import CircuitIRValidationError


def _relabel_circuit(circuit: CircuitIR, mapping: dict[str, str]) -> CircuitIR:
    components = [
        Component(
            cid=comp.cid,
            kind=comp.kind,
            node_a=mapping.get(comp.node_a, comp.node_a),
            node_b=mapping.get(comp.node_b, comp.node_b),
            value=comp.value,
            metadata=dict(comp.metadata),
        )
        for comp in circuit.components
    ]
    ports = [
        PortDef(
            name=port.name,
            pos=mapping.get(port.pos, port.pos),
            neg=mapping.get(port.neg, port.neg),
        )
        for port in circuit.ports
    ]
    return CircuitIR(
        name=circuit.name,
        ports=ports,
        components=components,
        symbols=dict(circuit.symbols),
        metadata=dict(circuit.metadata),
    )


def _base_circuit() -> CircuitIR:
    components = [
        Component("R1", "R", "a", "b", ParamValue(100.0)),
        Component("C1", "C", "b", "c", ParamValue(1e-6)),
        Component("R2", "R", "c", "a", ParamValue(200.0)),
    ]
    ports = [PortDef("P1", "a", "b"), PortDef("P2", "c", "b")]
    return CircuitIR(name="base", ports=ports, components=components)


def test_canonical_hash_stable_under_relabeling():
    circuit = _base_circuit()
    mapping = {"a": "n1", "b": "n2", "c": "n3"}
    relabeled = _relabel_circuit(circuit, mapping)
    hash_a = canonicalize_circuit(circuit).canonical_hash
    hash_b = canonicalize_circuit(relabeled).canonical_hash
    assert hash_a == hash_b


def test_port_identity_changes_hash():
    circuit = _base_circuit()
    swapped_ports = CircuitIR(
        name="base",
        ports=[
            PortDef("P1", circuit.ports[1].pos, circuit.ports[1].neg),
            PortDef("P2", circuit.ports[0].pos, circuit.ports[0].neg),
        ],
        components=circuit.components,
    )
    hash_a = canonicalize_circuit(circuit).canonical_hash
    hash_b = canonicalize_circuit(swapped_ports).canonical_hash
    assert hash_a != hash_b
    assert not are_isomorphic(circuit, swapped_ports)


def test_isomorphism_and_dedupe_agree():
    circuit = _base_circuit()
    nodes = ["a", "b", "c"]
    shuffled = nodes[:]
    random.Random(0).shuffle(shuffled)
    mapping = dict(zip(nodes, shuffled))
    relabeled = _relabel_circuit(circuit, mapping)
    assert are_isomorphic(circuit, relabeled)

    dedupe = DedupeIndex()
    _, is_new = dedupe.add(circuit)
    assert is_new
    _, is_new = dedupe.add(relabeled)
    assert not is_new


def test_symmetric_topology_canonicalization():
    components = [
        Component("R1", "R", "n0", "n1", ParamValue(100.0)),
        Component("R2", "R", "n1", "n2", ParamValue(100.0)),
        Component("R3", "R", "n2", "n3", ParamValue(100.0)),
        Component("R4", "R", "n3", "n0", ParamValue(100.0)),
    ]
    ports = [PortDef("P1", "n0", "n2"), PortDef("P2", "n1", "n3")]
    circuit = CircuitIR(name="square", ports=ports, components=components)

    mapping = {"n0": "a", "n1": "b", "n2": "c", "n3": "d"}
    relabeled = _relabel_circuit(circuit, mapping)

    hash_a = canonicalize_circuit(circuit).canonical_hash
    hash_b = canonicalize_circuit(relabeled).canonical_hash
    assert hash_a == hash_b


def test_motif_metadata_ignored_for_isomorphism_and_hash():
    components_a = [
        Component("R1", "R", "a", "b", ParamValue(100.0), metadata={"motif_id": "alpha"}),
        Component("C1", "C", "b", "c", ParamValue(1e-6), metadata={"motif_id": "beta"}),
        Component("R2", "R", "c", "a", ParamValue(200.0), metadata={"motif_id": "gamma"}),
    ]
    components_b = [
        Component("R1", "R", "a", "b", ParamValue(100.0), metadata={"motif_id": "delta"}),
        Component("C1", "C", "b", "c", ParamValue(1e-6), metadata={"motif_id": "epsilon"}),
        Component("R2", "R", "c", "a", ParamValue(200.0), metadata={"motif_id": "zeta"}),
    ]
    ports = [PortDef("P1", "a", "b"), PortDef("P2", "c", "b")]
    circuit_a = CircuitIR(name="motif_a", ports=ports, components=components_a)
    circuit_b = CircuitIR(name="motif_b", ports=ports, components=components_b)

    assert are_isomorphic(circuit_a, circuit_b)
    hash_a = canonicalize_circuit(circuit_a, mode="continuous").canonical_hash
    hash_b = canonicalize_circuit(circuit_b, mode="continuous").canonical_hash
    assert hash_a == hash_b

    dedupe = DedupeIndex()
    _, is_new = dedupe.add(circuit_a)
    assert is_new
    _, is_new = dedupe.add(circuit_b)
    assert not is_new


def test_canonicalize_budget_guard_raises():
    leaf = CircuitIR(
        name="leaf",
        ports=[PortDef("P", "p", "n")],
        components=[Component("R1", "R", "p", "n", ParamValue(1.0))],
    )
    subcircuits = [
        SubCircuit(
            name=f"leaf_{idx}",
            circuit=leaf,
            port_map={"P": PortConnection(pos="p", neg="n")},
        )
        for idx in range(3)
    ]
    circuit = CircuitIR(
        name="parent",
        ports=[PortDef("P", "p", "n")],
        subcircuits=subcircuits,
    )

    with pytest.raises(CircuitIRValidationError, match="max_components"):
        canonicalize_circuit(circuit, max_components=2)
