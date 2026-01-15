"""Core circuit data structures for linear passive networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Set

from fidp.errors import CircuitValidationError


NodeId = str


@dataclass(frozen=True)
class Resistor:
    """Passive resistor between two nodes."""

    node_a: NodeId
    node_b: NodeId
    resistance_ohms: float

    def __post_init__(self) -> None:
        if self.resistance_ohms <= 0:
            raise CircuitValidationError("Resistor value must be positive.")


@dataclass(frozen=True)
class Capacitor:
    """Passive capacitor between two nodes."""

    node_a: NodeId
    node_b: NodeId
    capacitance_f: float

    def __post_init__(self) -> None:
        if self.capacitance_f <= 0:
            raise CircuitValidationError("Capacitor value must be positive.")


@dataclass(frozen=True)
class Inductor:
    """Passive inductor between two nodes."""

    node_a: NodeId
    node_b: NodeId
    inductance_h: float

    def __post_init__(self) -> None:
        if self.inductance_h <= 0:
            raise CircuitValidationError("Inductor value must be positive.")


@dataclass(frozen=True)
class Port:
    """Port definition with explicit positive/negative nodes."""

    pos: NodeId
    neg: NodeId


@dataclass
class CircuitGraph:
    """Container for circuit elements with explicit ground node."""

    ground: NodeId = "0"
    components: List[object] = field(default_factory=list)
    nodes: Set[NodeId] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.ground:
            raise CircuitValidationError("Ground node must be provided.")
        self.nodes.add(self.ground)
        for comp in self.components:
            self._register_component_nodes(comp)

    def _register_component_nodes(self, comp: object) -> None:
        if hasattr(comp, "node_a") and hasattr(comp, "node_b"):
            self.nodes.add(comp.node_a)
            self.nodes.add(comp.node_b)
        else:
            raise CircuitValidationError("Component lacks node_a/node_b fields.")

    def add_resistor(self, node_a: NodeId, node_b: NodeId, resistance_ohms: float) -> Resistor:
        resistor = Resistor(node_a=node_a, node_b=node_b, resistance_ohms=resistance_ohms)
        self.components.append(resistor)
        self._register_component_nodes(resistor)
        return resistor

    def add_capacitor(self, node_a: NodeId, node_b: NodeId, capacitance_f: float) -> Capacitor:
        capacitor = Capacitor(node_a=node_a, node_b=node_b, capacitance_f=capacitance_f)
        self.components.append(capacitor)
        self._register_component_nodes(capacitor)
        return capacitor

    def add_inductor(self, node_a: NodeId, node_b: NodeId, inductance_h: float) -> Inductor:
        inductor = Inductor(node_a=node_a, node_b=node_b, inductance_h=inductance_h)
        self.components.append(inductor)
        self._register_component_nodes(inductor)
        return inductor

    def iter_components(self) -> Iterable[object]:
        return iter(self.components)
