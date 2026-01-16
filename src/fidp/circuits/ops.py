"""Operations on CircuitIR structures."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from fidp.circuits.ir import (
    CircuitIR,
    Component,
    ParamSymbol,
    ParamValue,
    PortDef,
    PortConnection,
    SubCircuit,
)
from fidp.errors import CircuitIRValidationError


E_SERIES: Dict[str, Tuple[float, ...]] = {
    "E6": (1.0, 1.5, 2.2, 3.3, 4.7, 6.8),
    "E12": (1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2),
    "E24": (
        1.0,
        1.1,
        1.2,
        1.3,
        1.5,
        1.6,
        1.8,
        2.0,
        2.2,
        2.4,
        2.7,
        3.0,
        3.3,
        3.6,
        3.9,
        4.3,
        4.7,
        5.1,
        5.6,
        6.2,
        6.8,
        7.5,
        8.2,
        9.1,
    ),
}


def snap_to_series(value: float, series: str) -> float:
    """Snap a value to the nearest E-series mantissa."""
    if value <= 0:
        raise CircuitIRValidationError("Snap value must be positive.")
    if series not in E_SERIES:
        raise CircuitIRValidationError(f"Unknown E-series: {series}")
    import math

    decade = 10 ** int(math.floor(math.log10(value)))
    normalized = value / decade
    best = min(E_SERIES[series], key=lambda m: abs(m - normalized))
    return best * decade


def apply_snap(value: ParamValue) -> ParamValue:
    """Apply snapping/bounds rules to a parameter value."""
    snapped = value.snapped
    if value.snap:
        snapped = snap_to_series(value.nominal, value.snap)
        if value.min_value is not None:
            snapped = max(snapped, value.min_value)
        if value.max_value is not None:
            snapped = min(snapped, value.max_value)
    return replace(value, snapped=snapped)


def namespace_circuit(circuit: CircuitIR, prefix: str) -> CircuitIR:
    """Return a copy of the circuit with node/component IDs prefixed."""
    node_map: Dict[str, str] = {}

    def map_node(node: str) -> str:
        if node not in node_map:
            node_map[node] = f"{prefix}{node}"
        return node_map[node]

    components = [
        replace(
            comp,
            cid=f"{prefix}{comp.cid}",
            node_a=map_node(comp.node_a),
            node_b=map_node(comp.node_b),
        )
        for comp in circuit.components
    ]
    ports = [
        PortDef(name=port.name, pos=map_node(port.pos), neg=map_node(port.neg))
        for port in circuit.ports
    ]
    subcircuits = [
        SubCircuit(
            name=f"{prefix}{sub.name}",
            circuit=sub.circuit,
            port_map={
                key: PortConnection(pos=map_node(conn.pos), neg=map_node(conn.neg))
                for key, conn in sub.port_map.items()
            },
            metadata=dict(sub.metadata),
        )
        for sub in circuit.subcircuits
    ]
    return CircuitIR(
        name=circuit.name,
        ports=ports,
        components=components,
        subcircuits=subcircuits,
        symbols=dict(circuit.symbols),
        metadata=dict(circuit.metadata),
    )


def remap_nodes(circuit: CircuitIR, mapping: Mapping[str, str]) -> CircuitIR:
    """Return a copy of the circuit with selected nodes remapped."""
    def remap(node: str) -> str:
        return mapping.get(node, node)

    components = [
        replace(comp, node_a=remap(comp.node_a), node_b=remap(comp.node_b))
        for comp in circuit.components
    ]
    ports = [PortDef(name=port.name, pos=remap(port.pos), neg=remap(port.neg)) for port in circuit.ports]
    subcircuits = [
        SubCircuit(
            name=sub.name,
            circuit=sub.circuit,
            port_map={
                key: PortConnection(pos=remap(conn.pos), neg=remap(conn.neg))
                for key, conn in sub.port_map.items()
            },
            metadata=dict(sub.metadata),
        )
        for sub in circuit.subcircuits
    ]
    return CircuitIR(
        name=circuit.name,
        ports=ports,
        components=components,
        subcircuits=subcircuits,
        symbols=dict(circuit.symbols),
        metadata=dict(circuit.metadata),
    )


def compose_series(circuits: Iterable[CircuitIR], name: str = "series") -> CircuitIR:
    """Compose one-port circuits in series (port.neg chained to next port.pos)."""
    circuits = list(circuits)
    if not circuits:
        raise CircuitIRValidationError("Series composition requires at least one circuit.")
    namespaced = [namespace_circuit(circuit, f"s{i}_") for i, circuit in enumerate(circuits)]
    base = namespaced[0]
    _ensure_one_port(base)
    port = base.ports[0]
    components = list(base.components)
    subcircuits = list(base.subcircuits)
    symbols = dict(base.symbols)

    for next_circuit in namespaced[1:]:
        _ensure_one_port(next_circuit)
        next_port = next_circuit.ports[0]
        remapped = remap_nodes(next_circuit, {next_port.pos: port.neg})
        components.extend(remapped.components)
        subcircuits.extend(remapped.subcircuits)
        port = PortDef(name=port.name, pos=port.pos, neg=remapped.ports[0].neg)
        symbols = _merge_symbols(symbols, remapped.symbols)

    return CircuitIR(
        name=name,
        ports=[port],
        components=components,
        subcircuits=subcircuits,
        symbols=symbols,
        metadata={"composition": "series"},
    )


def compose_parallel(circuits: Iterable[CircuitIR], name: str = "parallel") -> CircuitIR:
    """Compose one-port circuits in parallel (ports share nodes)."""
    circuits = list(circuits)
    if not circuits:
        raise CircuitIRValidationError("Parallel composition requires at least one circuit.")
    namespaced = [namespace_circuit(circuit, f"p{i}_") for i, circuit in enumerate(circuits)]
    base = namespaced[0]
    _ensure_one_port(base)
    port = base.ports[0]
    components = list(base.components)
    subcircuits = list(base.subcircuits)
    symbols = dict(base.symbols)

    for next_circuit in namespaced[1:]:
        _ensure_one_port(next_circuit)
        next_port = next_circuit.ports[0]
        remapped = remap_nodes(
            next_circuit,
            {next_port.pos: port.pos, next_port.neg: port.neg},
        )
        components.extend(remapped.components)
        subcircuits.extend(remapped.subcircuits)
        symbols = _merge_symbols(symbols, remapped.symbols)

    return CircuitIR(
        name=name,
        ports=[port],
        components=components,
        subcircuits=subcircuits,
        symbols=symbols,
        metadata={"composition": "parallel"},
    )


def scale_circuit(circuit: CircuitIR, factor: float) -> CircuitIR:
    """Scale component values by a positive factor."""
    if factor <= 0:
        raise CircuitIRValidationError("Scale factor must be positive.")

    def scale_value(value: ParamValue) -> ParamValue:
        return replace(
            value,
            nominal=value.nominal * factor,
            min_value=value.min_value * factor if value.min_value is not None else None,
            max_value=value.max_value * factor if value.max_value is not None else None,
            snapped=value.snapped * factor if value.snapped is not None else None,
        )

    components = [replace(comp, value=scale_value(comp.value)) for comp in circuit.components]
    subcircuits = [
        SubCircuit(
            name=sub.name,
            circuit=scale_circuit(sub.circuit, factor),
            port_map=dict(sub.port_map),
            metadata=dict(sub.metadata),
        )
        for sub in circuit.subcircuits
    ]
    symbols = {
        key: ParamSymbol(
            name=symbol.name,
            nominal=symbol.nominal * factor,
            min_value=symbol.min_value * factor if symbol.min_value is not None else None,
            max_value=symbol.max_value * factor if symbol.max_value is not None else None,
            snap=symbol.snap,
            snapped=symbol.snapped * factor if symbol.snapped is not None else None,
        )
        for key, symbol in circuit.symbols.items()
    }
    return CircuitIR(
        name=circuit.name,
        ports=list(circuit.ports),
        components=components,
        subcircuits=subcircuits,
        symbols=symbols,
        metadata=dict(circuit.metadata),
    )


def flatten_circuit(
    circuit: CircuitIR,
    max_depth: Optional[int] = None,
    max_nodes: Optional[int] = None,
    max_components: Optional[int] = None,
) -> CircuitIR:
    """Flatten hierarchical subcircuits into a single circuit."""
    if max_depth is not None and max_depth < 0:
        raise CircuitIRValidationError("max_depth must be non-negative.")
    if max_nodes is not None and max_nodes < 0:
        raise CircuitIRValidationError("max_nodes must be non-negative.")
    if max_components is not None and max_components < 0:
        raise CircuitIRValidationError("max_components must be non-negative.")
    components: List[Component] = []
    subcircuits: List[SubCircuit] = []
    def check_budget() -> None:
        if max_components is not None and len(components) > max_components:
            raise CircuitIRValidationError(f"Flattening exceeds max_components={max_components}.")
        if max_nodes is not None and len(nodes_seen) > max_nodes:
            raise CircuitIRValidationError(f"Flattening exceeds max_nodes={max_nodes}.")

    nodes_seen: set[str] = set()
    for port in circuit.ports:
        nodes_seen.update((port.pos, port.neg))
    check_budget()

    def resolve_node(node: str, mapping: Dict[str, str], prefix: str) -> str:
        if node in mapping:
            return mapping[node]
        return f"{prefix}{node}" if prefix else node

    def flatten(current: CircuitIR, mapping: Dict[str, str], prefix: str, depth: int) -> None:
        for comp in current.components:
            remapped = replace(
                comp,
                cid=f"{prefix}{comp.cid}",
                node_a=resolve_node(comp.node_a, mapping, prefix),
                node_b=resolve_node(comp.node_b, mapping, prefix),
            )
            components.append(remapped)
            nodes_seen.update((remapped.node_a, remapped.node_b))
            check_budget()
        for idx, sub in enumerate(current.subcircuits):
            sub_prefix = f"{prefix}{sub.name}_{idx}_"
            if max_depth is not None and depth >= max_depth:
                port_map = {
                    key: PortConnection(
                        pos=resolve_node(conn.pos, mapping, prefix),
                        neg=resolve_node(conn.neg, mapping, prefix),
                    )
                    for key, conn in sub.port_map.items()
                }
                for conn in port_map.values():
                    nodes_seen.update((conn.pos, conn.neg))
                check_budget()
                subcircuits.append(
                    SubCircuit(
                        name=sub.name,
                        circuit=sub.circuit,
                        port_map=port_map,
                        metadata=dict(sub.metadata),
                    )
                )
            else:
                sub_mapping: Dict[str, str] = {}
                ports_by_name = {port.name: port for port in sub.circuit.ports}
                for key, conn in sub.port_map.items():
                    port = ports_by_name[key]
                    sub_mapping[port.pos] = resolve_node(conn.pos, mapping, prefix)
                    sub_mapping[port.neg] = resolve_node(conn.neg, mapping, prefix)
                flatten(sub.circuit, sub_mapping, sub_prefix, depth + 1)

    flatten(circuit, {}, "", 0)

    return CircuitIR(
        name=circuit.name,
        ports=list(circuit.ports),
        components=components,
        subcircuits=subcircuits,
        symbols=dict(circuit.collect_symbols()),
        metadata=dict(circuit.metadata),
    )


def _ensure_one_port(circuit: CircuitIR) -> None:
    if len(circuit.ports) != 1:
        raise CircuitIRValidationError("Composition requires one-port circuits.")


def _merge_symbols(existing: Dict[str, ParamSymbol], incoming: Dict[str, ParamSymbol]) -> Dict[str, ParamSymbol]:
    merged = dict(existing)
    for key, symbol in incoming.items():
        if key in merged:
            if merged[key] != symbol:
                raise CircuitIRValidationError(f"Symbol mismatch for {key}.")
        else:
            merged[key] = symbol
    return merged
