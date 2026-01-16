"""AST definitions for the circuit DSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass(frozen=True)
class PortSpec:
    """Port specification with optional node bindings."""

    name: str
    pos_node: Optional[str] = None
    neg_node: Optional[str] = None


@dataclass(frozen=True)
class ParamSymbolSpec:
    """Symbolic parameter with bounds and snapping policy."""

    name: str
    nominal: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    snap: Optional[str] = None
    snapped: Optional[float] = None


class ValueExpr:
    """Base class for value expressions."""


@dataclass(frozen=True)
class NumberValue(ValueExpr):
    value: float


@dataclass(frozen=True)
class SymbolValue(ValueExpr):
    symbol: ParamSymbolSpec


@dataclass(frozen=True)
class NetlistStmt:
    kind: str
    value: ValueExpr
    node_a: str
    node_b: str


@dataclass(frozen=True)
class NetlistBlock:
    statements: List[NetlistStmt]


class DSLExpr:
    """Base class for DSL expressions."""


@dataclass(frozen=True)
class ElementExpr(DSLExpr):
    kind: str
    value: ValueExpr


@dataclass(frozen=True)
class SeriesExpr(DSLExpr):
    items: List[DSLExpr]


@dataclass(frozen=True)
class ParallelExpr(DSLExpr):
    items: List[DSLExpr]


@dataclass(frozen=True)
class RepeatExpr(DSLExpr):
    expr: DSLExpr
    depth: int


@dataclass(frozen=True)
class ScaleExpr(DSLExpr):
    expr: DSLExpr
    factor: ValueExpr


@dataclass(frozen=True)
class GenExpr(DSLExpr):
    name: str
    args: Dict[str, "ArgValue"]


@dataclass(frozen=True)
class RefExpr(DSLExpr):
    name: str


@dataclass(frozen=True)
class LetBinding:
    name: str
    expr: DSLExpr


@dataclass(frozen=True)
class DSLProgram:
    name: str
    ports: List[PortSpec]
    bindings: List[LetBinding]
    body: Optional[DSLExpr] = None
    netlist: Optional[NetlistBlock] = None


ArgValue = Union[ValueExpr, int, float, str]
