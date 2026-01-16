"""Exporters for CircuitIR."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Dict, Iterable, List, Optional

from fidp.circuits.canonical import canonicalize_circuit
from fidp.circuits.ir import CircuitIR, Component, ParamSymbol, ParamValue, PortConnection, PortDef, SubCircuit
from fidp.circuits.ops import flatten_circuit
from fidp.errors import CircuitIRValidationError, SpiceNetlistError


def circuit_to_json(circuit: CircuitIR) -> str:
    """Serialize CircuitIR to deterministic JSON."""
    payload = _circuit_to_dict(circuit)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def circuit_from_json(text: str) -> CircuitIR:
    """Deserialize CircuitIR from JSON."""
    data = json.loads(text)
    return _circuit_from_dict(data)


def export_spice_netlist(
    circuit: CircuitIR,
    title: Optional[str] = None,
    flatten: bool = True,
    max_depth: Optional[int] = None,
    canonicalize: bool = True,
    value_mode: str = "snapped",
) -> str:
    """Export a CircuitIR to a SPICE-compatible netlist."""
    if value_mode not in {"snapped", "continuous"}:
        raise CircuitIRValidationError(f"Unknown value mode: {value_mode}")
    circuit.validate()
    if flatten:
        flat = flatten_circuit(circuit, max_depth=max_depth)
        node_map: Optional[Dict[str, str]] = None
        if canonicalize:
            canonical = canonicalize_circuit(flat, mode=value_mode)
            node_map = canonical.node_mapping
            if title is None:
                title = canonical.canonical_hash
        if title is None:
            title = circuit.name
        netlist = _export_flat_netlist(flat, title=title, node_map=node_map, value_mode=value_mode)
    else:
        netlist = _export_hierarchical_netlist(circuit, title=title or circuit.name, value_mode=value_mode)
    return netlist


def lint_spice_netlist(netlist: str) -> List[str]:
    """Lint a SPICE netlist without invoking external simulators."""
    errors: List[str] = []
    subckts_defined: set[str] = set()
    subckt_stack: List[str] = []
    instances: List[str] = []
    has_end = False

    for raw_line in netlist.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        tokens = line.split()
        if not tokens:
            continue
        head = tokens[0].lower()
        if head == ".subckt":
            if len(tokens) < 2:
                errors.append("Invalid .subckt line.")
                continue
            name = tokens[1]
            subckts_defined.add(name)
            subckt_stack.append(name)
            continue
        if head == ".ends":
            if not subckt_stack:
                errors.append(".ends without .subckt")
            else:
                subckt_stack.pop()
            continue
        if head == ".end":
            has_end = True
            continue
        prefix = tokens[0][0].upper()
        if prefix in {"R", "C", "L"}:
            if len(tokens) < 4:
                errors.append(f"Invalid component line: {line}")
                continue
            try:
                float(tokens[3])
            except ValueError:
                errors.append(f"Invalid component value: {tokens[3]}")
            continue
        if prefix == "X":
            if len(tokens) < 3:
                errors.append(f"Invalid subcircuit instance: {line}")
                continue
            instances.append(tokens[-1])
            continue
        if head.startswith("."):
            continue
        errors.append(f"Unknown line: {line}")

    if subckt_stack:
        errors.append("Unclosed .subckt block.")
    if not has_end:
        errors.append("Missing .end statement.")
    for name in instances:
        if name not in subckts_defined:
            errors.append(f"Unknown subckt referenced: {name}")
    return errors


def validate_spice_netlist(netlist: str) -> None:
    """Raise SpiceNetlistError if netlist fails lint checks."""
    errors = lint_spice_netlist(netlist)
    if errors:
        raise SpiceNetlistError("\n".join(errors))


def _export_flat_netlist(
    circuit: CircuitIR,
    title: Optional[str],
    node_map: Optional[Dict[str, str]],
    value_mode: str,
) -> str:
    lines = [f"* {title}"]
    components = _sorted_components(circuit.components, node_map, value_mode)
    type_counts: Dict[str, int] = {"R": 0, "C": 0, "L": 0}
    for kind, node_a, node_b, value in components:
        type_counts[kind] += 1
        name = f"{kind}{type_counts[kind]}"
        lines.append(f"{name} {node_a} {node_b} {value}")
    lines.append(".end")
    return "\n".join(lines)


def _export_hierarchical_netlist(
    circuit: CircuitIR,
    title: Optional[str],
    value_mode: str,
) -> str:
    lines: List[str] = [f"* {title or circuit.name}"]
    subckt_defs: List[str] = []
    _emit_subckts(circuit, prefix="", defs=subckt_defs, value_mode=value_mode)
    lines.extend(subckt_defs)
    lines.extend(_emit_circuit_body(circuit, prefix="", value_mode=value_mode))
    lines.append(".end")
    return "\n".join(lines)


def _emit_subckts(
    circuit: CircuitIR,
    prefix: str,
    defs: List[str],
    value_mode: str,
) -> None:
    for idx, sub in enumerate(sorted(circuit.subcircuits, key=lambda item: item.name)):
        sub_name = f"{prefix}{sub.name}_{idx}"
        _emit_subckts(sub.circuit, prefix=f"{sub_name}_", defs=defs, value_mode=value_mode)
        port_nodes: List[str] = []
        for port in sub.circuit.ports:
            port_nodes.extend([_sanitize_node(port.pos), _sanitize_node(port.neg)])
        ports = " ".join(port_nodes)
        defs.append(f".subckt {sub_name} {ports}")
        defs.extend(_emit_circuit_body(sub.circuit, prefix=f"{sub_name}_", value_mode=value_mode))
        defs.append(f".ends {sub_name}")


def _emit_circuit_body(circuit: CircuitIR, prefix: str, value_mode: str) -> List[str]:
    lines: List[str] = []
    components = _sorted_components(circuit.components, None, value_mode)
    type_counts: Dict[str, int] = {"R": 0, "C": 0, "L": 0}
    for kind, node_a, node_b, value in components:
        type_counts[kind] += 1
        name = f"{kind}{type_counts[kind]}"
        lines.append(f"{name} {_sanitize_node(node_a)} {_sanitize_node(node_b)} {value}")
    for idx, sub in enumerate(sorted(circuit.subcircuits, key=lambda item: item.name)):
        sub_name = f"{prefix}{sub.name}_{idx}"
        port_nodes: List[str] = []
        for port in sub.circuit.ports:
            conn = sub.port_map[port.name]
            port_nodes.extend([_sanitize_node(conn.pos), _sanitize_node(conn.neg)])
        name = f"X{sub_name}"
        lines.append(f"{name} {' '.join(port_nodes)} {sub_name}")
    return lines


def _sorted_components(
    components: Iterable[Component],
    node_map: Optional[Dict[str, str]],
    value_mode: str,
) -> List[tuple[str, str, str, str]]:
    mapped: List[tuple[str, str, str, str]] = []
    for comp in components:
        node_a = node_map.get(comp.node_a, comp.node_a) if node_map else comp.node_a
        node_b = node_map.get(comp.node_b, comp.node_b) if node_map else comp.node_b
        if node_a > node_b:
            node_a, node_b = node_b, node_a
        value = comp.value.resolved("snapped" if value_mode == "snapped" else "continuous")
        mapped.append((comp.kind, _sanitize_node(node_a), _sanitize_node(node_b), _format_float(value)))
    mapped.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return mapped


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _sanitize_node(node: str) -> str:
    if not node:
        raise CircuitIRValidationError("Node name cannot be empty.")
    sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in node)
    if not sanitized[0].isalnum():
        sanitized = f"n{sanitized}"
    return sanitized


def _circuit_to_dict(circuit: CircuitIR) -> Dict[str, object]:
    return {
        "name": circuit.name,
        "ports": [_port_to_dict(port) for port in circuit.ports],
        "components": [_component_to_dict(comp) for comp in circuit.components],
        "subcircuits": [_subcircuit_to_dict(sub) for sub in circuit.subcircuits],
        "symbols": {key: _symbol_to_dict(symbol) for key, symbol in circuit.symbols.items()},
        "metadata": dict(circuit.metadata),
    }


def _circuit_from_dict(data: Dict[str, object]) -> CircuitIR:
    ports = [_port_from_dict(item) for item in data.get("ports", [])]
    components = [_component_from_dict(item) for item in data.get("components", [])]
    subcircuits = [_subcircuit_from_dict(item) for item in data.get("subcircuits", [])]
    symbols = {key: _symbol_from_dict(value) for key, value in data.get("symbols", {}).items()}
    metadata = dict(data.get("metadata", {}))
    circuit = CircuitIR(
        name=str(data.get("name", "circuit")),
        ports=ports,
        components=components,
        subcircuits=subcircuits,
        symbols=symbols,
        metadata=metadata,
    )
    circuit.validate()
    return circuit


def _port_to_dict(port: PortDef) -> Dict[str, object]:
    return {"name": port.name, "pos": port.pos, "neg": port.neg}


def _port_from_dict(data: Dict[str, object]) -> PortDef:
    return PortDef(name=str(data["name"]), pos=str(data["pos"]), neg=str(data["neg"]))


def _component_to_dict(comp: Component) -> Dict[str, object]:
    return {
        "cid": comp.cid,
        "kind": comp.kind,
        "node_a": comp.node_a,
        "node_b": comp.node_b,
        "value": _value_to_dict(comp.value),
        "metadata": dict(comp.metadata),
    }


def _component_from_dict(data: Dict[str, object]) -> Component:
    return Component(
        cid=str(data["cid"]),
        kind=str(data["kind"]),
        node_a=str(data["node_a"]),
        node_b=str(data["node_b"]),
        value=_value_from_dict(data["value"]),
        metadata=dict(data.get("metadata", {})),
    )


def _subcircuit_to_dict(sub: SubCircuit) -> Dict[str, object]:
    return {
        "name": sub.name,
        "circuit": _circuit_to_dict(sub.circuit),
        "port_map": {
            key: {"pos": conn.pos, "neg": conn.neg} for key, conn in sub.port_map.items()
        },
        "metadata": dict(sub.metadata),
    }


def _subcircuit_from_dict(data: Dict[str, object]) -> SubCircuit:
    circuit = _circuit_from_dict(data["circuit"])
    port_map = {
        key: PortConnection(pos=str(val["pos"]), neg=str(val["neg"]))
        for key, val in data.get("port_map", {}).items()
    }
    return SubCircuit(
        name=str(data["name"]),
        circuit=circuit,
        port_map=port_map,
        metadata=dict(data.get("metadata", {})),
    )


def _value_to_dict(value: ParamValue) -> Dict[str, object]:
    return {
        "nominal": value.nominal,
        "symbol": value.symbol,
        "min_value": value.min_value,
        "max_value": value.max_value,
        "snap": value.snap,
        "snapped": value.snapped,
    }


def _value_from_dict(data: Dict[str, object]) -> ParamValue:
    return ParamValue(
        nominal=float(data["nominal"]),
        symbol=data.get("symbol"),
        min_value=float(data["min_value"]) if data.get("min_value") is not None else None,
        max_value=float(data["max_value"]) if data.get("max_value") is not None else None,
        snap=data.get("snap"),
        snapped=float(data["snapped"]) if data.get("snapped") is not None else None,
    )


def _symbol_to_dict(symbol: ParamSymbol) -> Dict[str, object]:
    return {
        "name": symbol.name,
        "nominal": symbol.nominal,
        "min_value": symbol.min_value,
        "max_value": symbol.max_value,
        "snap": symbol.snap,
        "snapped": symbol.snapped,
    }


def _symbol_from_dict(data: Dict[str, object]) -> ParamSymbol:
    return ParamSymbol(
        name=str(data["name"]),
        nominal=float(data["nominal"]),
        min_value=float(data["min_value"]) if data.get("min_value") is not None else None,
        max_value=float(data["max_value"]) if data.get("max_value") is not None else None,
        snap=data.get("snap"),
        snapped=float(data["snapped"]) if data.get("snapped") is not None else None,
    )
