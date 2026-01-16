"""Compiler from DSL AST to CircuitIR."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional

from fidp.circuits.ir import CircuitIR, Component, ParamSymbol, ParamValue, PortDef
from fidp.circuits.ops import apply_snap, compose_parallel, compose_series, scale_circuit
from fidp.dsl import generators
from fidp.dsl.ast import (
    ArgValue,
    DSLExpr,
    DSLProgram,
    ElementExpr,
    GenExpr,
    LetBinding,
    NetlistBlock,
    NumberValue,
    ParallelExpr,
    ParamSymbolSpec,
    PortSpec,
    RepeatExpr,
    ScaleExpr,
    SeriesExpr,
    SymbolValue,
    RefExpr,
    ValueExpr,
)
from fidp.errors import CircuitIRValidationError, DSLValidationError


class _NodeGenerator:
    def __init__(self) -> None:
        self._counter = 0

    def next(self) -> str:
        node = f"n{self._counter}"
        self._counter += 1
        return node


class _ComponentIdGenerator:
    def __init__(self) -> None:
        self._counts = {"R": 0, "C": 0, "L": 0}

    def next(self, kind: str) -> str:
        if kind not in self._counts:
            raise CircuitIRValidationError(f"Unsupported component kind: {kind}")
        self._counts[kind] += 1
        return f"{kind}{self._counts[kind]}"


class _CompilerContext:
    def __init__(self, seed: Optional[int]) -> None:
        self.seed = seed
        self.node_gen = _NodeGenerator()
        self.comp_gen = _ComponentIdGenerator()
        self.symbols: Dict[str, ParamSymbol] = {}


def compile_dsl(program: DSLProgram, seed: Optional[int] = None) -> CircuitIR:
    """Compile a DSL program into CircuitIR."""
    if program.body is None and program.netlist is None:
        raise DSLValidationError("DSL program must define body or netlist.")
    if program.body is not None and program.netlist is not None:
        raise DSLValidationError("DSL program cannot define both body and netlist.")
    if not program.ports:
        raise DSLValidationError("DSL program must define ports.")

    context = _CompilerContext(seed)
    bindings = {binding.name: binding.expr for binding in program.bindings}

    if program.body is not None:
        circuit = _compile_expr(program.body, bindings, context)
    else:
        circuit = _compile_netlist(program.netlist, context)

    circuit = _apply_port_specs(circuit, program.ports)
    symbols = _merge_symbols(context.symbols, circuit.collect_symbols())
    metadata = dict(circuit.metadata)
    metadata.setdefault("generator", None)
    metadata.setdefault("recursion_depth", 0)
    metadata.setdefault("motif_ids", [])
    metadata.update({"source": "dsl", "seed": seed})
    circuit = replace(
        circuit,
        name=program.name,
        symbols=symbols,
        metadata=metadata,
    )
    circuit.validate()
    return circuit


def _compile_expr(expr: DSLExpr, bindings: Dict[str, DSLExpr], context: _CompilerContext) -> CircuitIR:
    if isinstance(expr, ElementExpr):
        value = _value_expr_to_param(expr.value, context)
        value = apply_snap(value)
        node_pos = context.node_gen.next()
        node_neg = context.node_gen.next()
        component = Component(
            cid=context.comp_gen.next(expr.kind),
            kind=expr.kind,
            node_a=node_pos,
            node_b=node_neg,
            value=value,
        )
        symbols = _symbols_for_value(value)
        return CircuitIR(
            name=f"{expr.kind}_element",
            ports=[PortDef(name="P", pos=node_pos, neg=node_neg)],
            components=[component],
            symbols=symbols,
        )
    if isinstance(expr, SeriesExpr):
        circuits = [_compile_expr(item, bindings, context) for item in expr.items]
        return compose_series(circuits)
    if isinstance(expr, ParallelExpr):
        circuits = [_compile_expr(item, bindings, context) for item in expr.items]
        return compose_parallel(circuits)
    if isinstance(expr, RepeatExpr):
        if expr.depth <= 0:
            raise DSLValidationError("Repeat depth must be positive.")
        circuits = [_compile_expr(expr.expr, bindings, context) for _ in range(expr.depth)]
        return compose_series(circuits, name="repeat")
    if isinstance(expr, ScaleExpr):
        factor = _numeric_value(expr.factor)
        return scale_circuit(_compile_expr(expr.expr, bindings, context), factor)
    if isinstance(expr, GenExpr):
        args = _evaluate_args(expr.args, context)
        circuit = generators.generate(expr.name, seed=context.seed, **args)
        if circuit is None:
            raise DSLValidationError(f"Unknown generator: {expr.name}")
        return circuit
    if isinstance(expr, RefExpr):
        if expr.name not in bindings:
            raise DSLValidationError(f"Unknown binding: {expr.name}")
        return _compile_expr(bindings[expr.name], bindings, context)
    raise DSLValidationError(f"Unsupported DSL expression: {expr}")


def _compile_netlist(netlist: Optional[NetlistBlock], context: _CompilerContext) -> CircuitIR:
    if netlist is None:
        raise DSLValidationError("Netlist block is missing.")
    components: List[Component] = []
    for stmt in netlist.statements:
        value = _value_expr_to_param(stmt.value, context)
        value = apply_snap(value)
        components.append(
            Component(
                cid=context.comp_gen.next(stmt.kind),
                kind=stmt.kind,
                node_a=stmt.node_a,
                node_b=stmt.node_b,
                value=value,
            )
        )
    return CircuitIR(name="netlist", ports=[], components=components, symbols=context.symbols)


def _apply_port_specs(circuit: CircuitIR, ports: List[PortSpec]) -> CircuitIR:
    if not circuit.ports:
        mapped_ports = []
        for spec in ports:
            if not (spec.pos_node and spec.neg_node):
                raise DSLValidationError("Netlist ports must define node bindings.")
            mapped_ports.append(PortDef(name=spec.name, pos=spec.pos_node, neg=spec.neg_node))
        return replace(circuit, ports=mapped_ports)

    if len(ports) != len(circuit.ports):
        raise DSLValidationError(
            f"Port count mismatch: expected {len(circuit.ports)}, got {len(ports)}"
        )

    mapped_ports: List[PortDef] = []
    mapping: Dict[str, str] = {}
    for spec, port in zip(ports, circuit.ports):
        pos = port.pos
        neg = port.neg
        if spec.pos_node and spec.neg_node:
            mapping[pos] = spec.pos_node
            mapping[neg] = spec.neg_node
            pos = spec.pos_node
            neg = spec.neg_node
        mapped_ports.append(PortDef(name=spec.name, pos=pos, neg=neg))

    if mapping:
        circuit = _remap_nodes(circuit, mapping)
    return replace(circuit, ports=mapped_ports)


def _remap_nodes(circuit: CircuitIR, mapping: Dict[str, str]) -> CircuitIR:
    components = [
        replace(comp, node_a=mapping.get(comp.node_a, comp.node_a), node_b=mapping.get(comp.node_b, comp.node_b))
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
    subcircuits = []
    for sub in circuit.subcircuits:
        port_map = {
            key: replace(
                conn,
                pos=mapping.get(conn.pos, conn.pos),
                neg=mapping.get(conn.neg, conn.neg),
            )
            for key, conn in sub.port_map.items()
        }
        subcircuits.append(replace(sub, port_map=port_map))
    return replace(circuit, components=components, ports=ports, subcircuits=subcircuits)


def _value_expr_to_param(expr: ValueExpr, context: _CompilerContext) -> ParamValue:
    if isinstance(expr, NumberValue):
        return ParamValue(nominal=expr.value)
    if isinstance(expr, SymbolValue):
        spec = expr.symbol
        value = ParamValue(
            nominal=spec.nominal,
            symbol=spec.name,
            min_value=spec.min_value,
            max_value=spec.max_value,
            snap=spec.snap,
            snapped=spec.snapped,
        )
        _register_symbol(value, context)
        return value
    raise DSLValidationError(f"Unsupported value expression: {expr}")


def _register_symbol(value: ParamValue, context: _CompilerContext) -> None:
    if value.symbol is None:
        return
    symbol = ParamSymbol(
        name=value.symbol,
        nominal=value.nominal,
        min_value=value.min_value,
        max_value=value.max_value,
        snap=value.snap,
        snapped=value.snapped,
    )
    if value.symbol in context.symbols and context.symbols[value.symbol] != symbol:
        raise DSLValidationError(f"Conflicting symbol definition for {value.symbol}.")
    context.symbols[value.symbol] = symbol


def _symbols_for_value(value: ParamValue) -> Dict[str, ParamSymbol]:
    if value.symbol is None:
        return {}
    return {
        value.symbol: ParamSymbol(
            name=value.symbol,
            nominal=value.nominal,
            min_value=value.min_value,
            max_value=value.max_value,
            snap=value.snap,
            snapped=value.snapped,
        )
    }


def _numeric_value(value: ValueExpr) -> float:
    if isinstance(value, NumberValue):
        return value.value
    raise DSLValidationError("Scaling requires a numeric value.")


def _evaluate_args(args: Dict[str, ArgValue], context: _CompilerContext) -> Dict[str, Any]:
    evaluated: Dict[str, Any] = {}
    int_keys = {"stages", "depth", "order", "rows", "cols"}
    for key, value in args.items():
        if isinstance(value, NumberValue) and key in int_keys and float(value.value).is_integer():
            evaluated[key] = int(value.value)
        elif isinstance(value, ValueExpr):
            evaluated[key] = _value_expr_to_param(value, context)
        else:
            evaluated[key] = value
    return evaluated


def _merge_symbols(
    left: Dict[str, ParamSymbol],
    right: Dict[str, ParamSymbol],
) -> Dict[str, ParamSymbol]:
    merged = dict(left)
    for key, symbol in right.items():
        if key in merged and merged[key] != symbol:
            raise DSLValidationError(f"Conflicting symbol definition for {key}.")
        merged[key] = symbol
    return merged
