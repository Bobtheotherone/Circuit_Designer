"""Intermediate representation for circuit descriptions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

from fidp.errors import CircuitIRValidationError


NodeId = str
ComponentKind = str

VALID_COMPONENT_KINDS = {"R", "C", "L"}


@dataclass(frozen=True)
class ParamValue:
    """Represents a component value with optional symbol metadata."""

    nominal: float
    symbol: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    snap: Optional[str] = None
    snapped: Optional[float] = None

    def resolved(self, mode: str = "continuous") -> float:
        """Return the resolved numeric value for the given mode."""
        if mode == "snapped" and self.snapped is not None:
            return self.snapped
        return self.nominal


@dataclass(frozen=True)
class ParamSymbol:
    """Parameter symbol definition with bounds and snapping policy."""

    name: str
    nominal: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    snap: Optional[str] = None
    snapped: Optional[float] = None


@dataclass(frozen=True)
class Component:
    """Two-terminal passive component."""

    cid: str
    kind: ComponentKind
    node_a: NodeId
    node_b: NodeId
    value: ParamValue
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in VALID_COMPONENT_KINDS:
            raise CircuitIRValidationError(f"Unsupported component kind: {self.kind}")
        if not self.node_a or not self.node_b:
            raise CircuitIRValidationError("Component nodes must be non-empty.")
        if self.node_a == self.node_b:
            raise CircuitIRValidationError("Component nodes must be distinct.")
        _validate_param_value(self.value)


@dataclass(frozen=True)
class PortDef:
    """Port definition with explicit positive/negative nodes."""

    name: str
    pos: NodeId
    neg: NodeId


@dataclass(frozen=True)
class PortConnection:
    """Port mapping for subcircuit instances."""

    pos: NodeId
    neg: NodeId


@dataclass(frozen=True)
class SubCircuit:
    """Hierarchical subcircuit instance with its own circuit definition."""

    name: str
    circuit: "CircuitIR"
    port_map: Dict[str, PortConnection]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitIR:
    """Hierarchical circuit representation with explicit ports and metadata."""

    name: str
    ports: List[PortDef]
    components: List[Component] = field(default_factory=list)
    subcircuits: List[SubCircuit] = field(default_factory=list)
    symbols: Dict[str, ParamSymbol] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate circuit structure and component definitions."""
        if not self.name:
            raise CircuitIRValidationError("Circuit name must be provided.")
        if not self.ports:
            raise CircuitIRValidationError("At least one port is required.")
        port_names = [port.name for port in self.ports]
        if len(set(port_names)) != len(port_names):
            raise CircuitIRValidationError("Port names must be unique.")
        for port in self.ports:
            if not port.pos or not port.neg:
                raise CircuitIRValidationError("Port nodes must be non-empty.")
            if port.pos == port.neg:
                raise CircuitIRValidationError("Port nodes must be distinct.")

        for comp in self.components:
            _validate_param_value(comp.value)

        self._validate_symbols()

        connected_nodes = set()
        for comp in self.components:
            connected_nodes.add(comp.node_a)
            connected_nodes.add(comp.node_b)
        for sub in self.subcircuits:
            sub.circuit.validate()
            _validate_subcircuit_port_map(sub)
            for conn in sub.port_map.values():
                connected_nodes.add(conn.pos)
                connected_nodes.add(conn.neg)

        for port in self.ports:
            if port.pos not in connected_nodes or port.neg not in connected_nodes:
                raise CircuitIRValidationError(
                    f"Port {port.name} is floating (pos={port.pos}, neg={port.neg})."
                )

    def _validate_symbols(self) -> None:
        """Ensure symbol definitions are consistent with component values."""
        for comp in self.components:
            symbol = comp.value.symbol
            if symbol is None:
                continue
            if symbol not in self.symbols:
                raise CircuitIRValidationError(
                    f"Component {comp.cid} references undefined symbol {symbol}."
                )
            expected = self.symbols[symbol]
            if not _symbols_compatible(comp.value, expected):
                raise CircuitIRValidationError(
                    f"Symbol metadata mismatch for {symbol} in component {comp.cid}."
                )
        for sub in self.subcircuits:
            for key, sym in sub.circuit.symbols.items():
                if key not in self.symbols:
                    continue
                if not _symbol_defs_match(sym, self.symbols[key]):
                    raise CircuitIRValidationError(
                        f"Symbol {key} differs between circuit and subcircuit {sub.name}."
                    )

    def collect_symbols(self) -> Dict[str, ParamSymbol]:
        """Collect symbol definitions from components and subcircuits."""
        symbols: Dict[str, ParamSymbol] = dict(self.symbols)
        for comp in self.components:
            symbol = comp.value.symbol
            if symbol is None:
                continue
            if symbol not in symbols:
                symbols[symbol] = ParamSymbol(
                    name=symbol,
                    nominal=comp.value.nominal,
                    min_value=comp.value.min_value,
                    max_value=comp.value.max_value,
                    snap=comp.value.snap,
                    snapped=comp.value.snapped,
                )
        for sub in self.subcircuits:
            for key, sym in sub.circuit.collect_symbols().items():
                if key not in symbols:
                    symbols[key] = sym
        return symbols


def _validate_param_value(value: ParamValue) -> None:
    if not math.isfinite(value.nominal) or value.nominal <= 0:
        raise CircuitIRValidationError("Component value must be positive and finite.")
    if value.snapped is not None:
        if not math.isfinite(value.snapped) or value.snapped <= 0:
            raise CircuitIRValidationError("Snapped value must be positive and finite.")
    if value.min_value is not None:
        if not math.isfinite(value.min_value):
            raise CircuitIRValidationError("Min value must be finite.")
        if value.min_value <= 0:
            raise CircuitIRValidationError("Min value must be positive.")
    if value.max_value is not None:
        if not math.isfinite(value.max_value):
            raise CircuitIRValidationError("Max value must be finite.")
        if value.max_value <= 0:
            raise CircuitIRValidationError("Max value must be positive.")
    if value.min_value is not None and value.max_value is not None:
        if value.min_value > value.max_value:
            raise CircuitIRValidationError("Min value cannot exceed max value.")


def _symbols_compatible(value: ParamValue, symbol: ParamSymbol) -> bool:
    return _symbol_defs_match(
        ParamSymbol(
            name=value.symbol or "",
            nominal=value.nominal,
            min_value=value.min_value,
            max_value=value.max_value,
            snap=value.snap,
            snapped=value.snapped,
        ),
        symbol,
    )


def _symbol_defs_match(left: ParamSymbol, right: ParamSymbol) -> bool:
    return (
        left.name == right.name
        and _float_equal(left.nominal, right.nominal)
        and _float_equal(left.min_value, right.min_value)
        and _float_equal(left.max_value, right.max_value)
        and left.snap == right.snap
        and _float_equal(left.snapped, right.snapped)
    )


def _float_equal(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return left is right
    return left == right


def _validate_subcircuit_port_map(sub: SubCircuit) -> None:
    port_names = {port.name for port in sub.circuit.ports}
    if set(sub.port_map.keys()) != port_names:
        missing = port_names - set(sub.port_map.keys())
        extra = set(sub.port_map.keys()) - port_names
        raise CircuitIRValidationError(
            f"Subcircuit {sub.name} port map mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for port_name, conn in sub.port_map.items():
        if not conn.pos or not conn.neg:
            raise CircuitIRValidationError(
                f"Subcircuit {sub.name} port {port_name} has empty node mapping."
            )
        if conn.pos == conn.neg:
            raise CircuitIRValidationError(
                f"Subcircuit {sub.name} port {port_name} maps to identical nodes."
            )
