"""Parser and formatter for the circuit DSL."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

from lark import Lark, Transformer, UnexpectedInput

from fidp.dsl.ast import (
    ArgValue,
    DSLExpr,
    DSLProgram,
    ElementExpr,
    GenExpr,
    LetBinding,
    NetlistBlock,
    NetlistStmt,
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
from fidp.errors import DSLParseError


_GRAMMAR_PATH = Path(__file__).with_name("grammar.lark")


def _load_parser() -> Lark:
    return Lark(_GRAMMAR_PATH.read_text(encoding="utf-8"), parser="lalr", propagate_positions=True)


_PARSER = _load_parser()


class _DSLTransformer(Transformer):
    def statement(self, items):
        return items[0]

    def program(self, items):
        name = str(items[0])
        ports: List[PortSpec] = []
        bindings: List[LetBinding] = []
        body: DSLExpr | None = None
        netlist: NetlistBlock | None = None
        for item in items[1:]:
            if isinstance(item, list) and item and isinstance(item[0], PortSpec):
                ports = item
            elif isinstance(item, LetBinding):
                bindings.append(item)
            elif isinstance(item, NetlistBlock):
                netlist = item
            elif isinstance(item, DSLExpr):
                body = item
        return DSLProgram(name=name, ports=ports, bindings=bindings, body=body, netlist=netlist)

    def port_decl(self, items):
        return items[0]

    def port_list(self, items):
        return items

    def port_spec(self, items):
        name = str(items[0])
        if len(items) == 1:
            return PortSpec(name=name)
        pos = str(items[1])
        neg = str(items[2])
        return PortSpec(name=name, pos_node=pos, neg_node=neg)

    def let_decl(self, items):
        return LetBinding(name=str(items[0]), expr=items[1])

    def body_decl(self, items):
        return items[0]

    def netlist_decl(self, items):
        return NetlistBlock(statements=items)

    def net_stmt(self, items):
        return NetlistStmt(kind=str(items[0]), value=items[1], node_a=str(items[2]), node_b=str(items[3]))

    def series_expr(self, items):
        return SeriesExpr(items=items[0])

    def parallel_expr(self, items):
        return ParallelExpr(items=items[0])

    def expr_list(self, items):
        return items

    def repeat_expr(self, items):
        return RepeatExpr(expr=items[0], depth=int(items[1]))

    def scale_expr(self, items):
        return ScaleExpr(expr=items[0], factor=items[1])

    def element_expr(self, items):
        return ElementExpr(kind=str(items[0]), value=items[1])

    def gen_expr(self, items):
        name = str(items[0])
        args: Dict[str, ArgValue] = {}
        if len(items) > 1:
            for key, value in items[1]:
                args[key] = value
        return GenExpr(name=name, args=args)

    def arg_list(self, items):
        return items

    def arg(self, items):
        return (str(items[0]), items[1])

    def arg_value(self, items):
        return items[0]

    def ref_expr(self, items):
        return RefExpr(name=str(items[0]))

    def value_expr(self, items):
        value = items[0]
        if isinstance(value, float):
            return NumberValue(value=value)
        return value

    def symbol_expr(self, items):
        name = str(items[0])
        nominal = _coerce_number(items[1])
        spec = ParamSymbolSpec(name=name, nominal=nominal)
        for key, val in items[2:]:
            if key == "min":
                spec = replace(spec, min_value=_coerce_number(val))
            elif key == "max":
                spec = replace(spec, max_value=_coerce_number(val))
            elif key == "snap":
                spec = replace(spec, snap=val)
            elif key == "snapped":
                spec = replace(spec, snapped=_coerce_number(val))
        return SymbolValue(symbol=spec)

    def min_kv(self, items):
        return ("min", items[0])

    def max_kv(self, items):
        return ("max", items[0])

    def snap_kv(self, items):
        return ("snap", str(items[0]))

    def snapped_kv(self, items):
        return ("snapped", items[0])

    def symbol_kv(self, items):
        return items[0]

    def NUMBER(self, token):
        return _parse_number(str(token))

    def INT(self, token):
        return int(token)

    def NAME(self, token):
        return str(token)

    def STRING(self, token):
        return str(token)[1:-1]


def parse_dsl(text: str) -> DSLProgram:
    """Parse DSL text into an AST program."""
    try:
        tree = _PARSER.parse(text)
    except UnexpectedInput as exc:  # pragma: no cover - covered in tests
        context = exc.get_context(text) if hasattr(exc, "get_context") else ""
        raise DSLParseError(
            f"DSL parse error at line {exc.line}, column {exc.column}: {exc}".strip(),
            line=exc.line,
            column=exc.column,
            context=context,
        ) from exc
    program = _DSLTransformer().transform(tree)
    return program


def format_program(program: DSLProgram) -> str:
    """Format a DSL program into canonical text."""
    lines = [f"circuit {program.name} {{"]
    if program.ports:
        ports = ", ".join(_format_port(port) for port in program.ports)
        lines.append(f"  ports: ({ports});")
    for binding in program.bindings:
        lines.append(f"  let {binding.name} = {_format_expr(binding.expr)};")
    if program.body is not None:
        lines.append(f"  body: {_format_expr(program.body)};")
    if program.netlist is not None:
        lines.append("  netlist: {")
        for stmt in program.netlist.statements:
            lines.append(
                f"    {stmt.kind}({_format_value(stmt.value)}) {stmt.node_a} {stmt.node_b};"
            )
        lines.append("  };")
    lines.append("}")
    return "\n".join(lines)


def _format_port(port: PortSpec) -> str:
    if port.pos_node and port.neg_node:
        return f"{port.name}:({port.pos_node},{port.neg_node})"
    return port.name


def _format_expr(expr: DSLExpr) -> str:
    if isinstance(expr, SeriesExpr):
        return f"series({_format_expr_list(expr.items)})"
    if isinstance(expr, ParallelExpr):
        return f"parallel({_format_expr_list(expr.items)})"
    if isinstance(expr, RepeatExpr):
        return f"repeat({_format_expr(expr.expr)},{expr.depth})"
    if isinstance(expr, ScaleExpr):
        return f"scale({_format_expr(expr.expr)},{_format_value(expr.factor)})"
    if isinstance(expr, ElementExpr):
        return f"{expr.kind}({_format_value(expr.value)})"
    if isinstance(expr, GenExpr):
        return _format_gen(expr)
    if isinstance(expr, RefExpr):
        return expr.name
    raise TypeError(f"Unsupported expression: {expr}")


def _format_gen(expr: GenExpr) -> str:
    if not expr.args:
        return f"gen {expr.name}()"
    items = ",".join(f"{key}={_format_arg(expr.args[key])}" for key in sorted(expr.args))
    return f"gen {expr.name}({items})"


def _format_expr_list(items: List[DSLExpr]) -> str:
    return ",".join(_format_expr(item) for item in items)


def _format_arg(value: ArgValue) -> str:
    if isinstance(value, ValueExpr):
        return _format_value(value)
    if isinstance(value, str):
        return _format_string(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_number(value)
    raise TypeError(f"Unsupported arg value: {value}")


def _format_value(value: ValueExpr) -> str:
    if isinstance(value, NumberValue):
        return _format_number(value.value)
    if isinstance(value, SymbolValue):
        return _format_symbol(value.symbol)
    raise TypeError(f"Unsupported value expression: {value}")


def _format_symbol(symbol: ParamSymbolSpec) -> str:
    parts = [f"sym({symbol.name},{_format_number(symbol.nominal)}"]
    if symbol.min_value is not None:
        parts.append(f",min={_format_number(symbol.min_value)}")
    if symbol.max_value is not None:
        parts.append(f",max={_format_number(symbol.max_value)}")
    if symbol.snap is not None:
        parts.append(f",snap={symbol.snap}")
    if symbol.snapped is not None:
        parts.append(f",snapped={_format_number(symbol.snapped)}")
    parts.append(")")
    return "".join(parts)


def _format_number(value: float) -> str:
    return f"{value:.12g}"


def _format_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', "\\\"")
    return f"\"{escaped}\""


def _parse_number(token: str) -> float:
    if not token:
        raise DSLParseError("Empty number token.")
    suffix = token[-1]
    multipliers = {
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "K": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
    }
    if suffix.isalpha() and suffix in multipliers:
        return float(token[:-1]) * multipliers[suffix]
    return float(token)


def _coerce_number(value: object) -> float:
    if isinstance(value, NumberValue):
        return value.value
    if isinstance(value, float):
        return value
    return _parse_number(str(value))
