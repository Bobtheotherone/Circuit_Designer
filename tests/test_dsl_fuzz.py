import random

from hypothesis import given, settings
from hypothesis import strategies as st

from fidp.circuits.canonical import canonicalize_circuit
from fidp.circuits.ir import CircuitIR, Component, ParamValue, PortDef
from fidp.circuits.ir_export import circuit_from_json, circuit_to_json, export_spice_netlist, lint_spice_netlist
from fidp.dsl import compile_dsl, format_program, parse_dsl
from fidp.dsl.ast import ElementExpr, NumberValue, ParallelExpr, PortSpec, RepeatExpr, ScaleExpr, SeriesExpr, DSLProgram


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
        subcircuits=circuit.subcircuits,
        symbols=dict(circuit.symbols),
        metadata=dict(circuit.metadata),
    )


def _expr_strategy():
    values = st.sampled_from([1.0, 10.0, 100.0, 1e-6]).map(NumberValue)
    elements = st.builds(
        ElementExpr,
        kind=st.sampled_from(["R", "C", "L"]),
        value=values,
    )

    def extend(children):
        return st.one_of(
            st.builds(SeriesExpr, items=st.lists(children, min_size=2, max_size=3)),
            st.builds(ParallelExpr, items=st.lists(children, min_size=2, max_size=3)),
            st.builds(RepeatExpr, expr=children, depth=st.integers(min_value=1, max_value=3)),
            st.builds(ScaleExpr, expr=children, factor=values),
        )

    return st.recursive(elements, extend, max_leaves=5)


@settings(max_examples=20, deadline=None)
@given(expr=_expr_strategy())
def test_fuzz_dsl_roundtrip_and_export(expr):
    program = DSLProgram(name="Fuzz", ports=[PortSpec(name="P")], bindings=[], body=expr)
    dsl_text = format_program(program)
    parsed = parse_dsl(dsl_text)
    circuit = compile_dsl(parsed, seed=0)

    json_text = circuit_to_json(circuit)
    restored = circuit_from_json(json_text)
    assert circuit_to_json(restored) == json_text

    nodes = sorted({
        node
        for comp in circuit.components
        for node in (comp.node_a, comp.node_b)
    })
    rng = random.Random(0)
    shuffled = nodes[:]
    rng.shuffle(shuffled)
    mapping = dict(zip(nodes, shuffled))
    relabeled = _relabel_circuit(circuit, mapping)
    assert canonicalize_circuit(circuit).canonical_hash == canonicalize_circuit(relabeled).canonical_hash

    netlist = export_spice_netlist(circuit)
    assert lint_spice_netlist(netlist) == []
