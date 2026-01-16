"""Fractal and tree generator motifs."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

from fidp.circuits.ir import CircuitIR, Component, ParamSymbol, ParamValue, PortConnection, PortDef, SubCircuit
from fidp.dsl.generators.utils import (
    ComponentIdGenerator,
    ensure_param,
    merge_symbol_maps,
    scale_param,
    snap_param,
    symbols_for,
)
from fidp.errors import CircuitIRValidationError


def fractal_ladder(
    order: int,
    r_value: ParamValue | float,
    c_value: ParamValue | float,
    scale: float = 0.5,
    seed: Optional[int] = None,
) -> CircuitIR:
    """Generate a self-similar fractal ladder (one-port)."""
    if order < 0:
        raise CircuitIRValidationError("Fractal order must be non-negative.")
    r_param = ensure_param(r_value)
    c_param = ensure_param(c_value)
    return _fractal_ladder_recursive(order, r_param, c_param, scale, seed)


def _fractal_ladder_recursive(
    order: int,
    r_param: ParamValue,
    c_param: ParamValue,
    scale: float,
    seed: Optional[int],
) -> CircuitIR:
    comp_gen = ComponentIdGenerator()
    port = PortDef(name="P", pos="p", neg="n")
    if order == 0:
        r_snap = snap_param(r_param)
        c_snap = snap_param(c_param)
        components = [
            Component(
                cid=comp_gen.next("R"),
                kind="R",
                node_a="p",
                node_b="x",
                value=r_snap,
                metadata={"motif_id": "fractal_base"},
            ),
            Component(
                cid=comp_gen.next("C"),
                kind="C",
                node_a="x",
                node_b="n",
                value=c_snap,
                metadata={"motif_id": "fractal_base"},
            ),
        ]
        symbols = symbols_for([r_snap, c_snap])
        return CircuitIR(
            name="fractal_ladder",
            ports=[port],
            components=components,
            symbols=symbols,
            metadata={
                "generator": "fractal_ladder",
                "recursion_depth": 0,
                "seed": seed,
                "motif_ids": ["fractal"],
            },
        )

    r_scaled = _scaled_symbol(scale_param(r_param, scale), suffix=f"d{order}")
    c_scaled = _scaled_symbol(scale_param(c_param, scale), suffix=f"d{order}")
    child = _fractal_ladder_recursive(order - 1, r_scaled, c_scaled, scale, seed)

    r_snap = snap_param(r_param)
    c_snap = snap_param(c_param)
    components = [
        Component(
            cid=comp_gen.next("R"),
            kind="R",
            node_a="p",
            node_b="x",
            value=r_snap,
            metadata={"motif_id": "fractal_resistor", "depth": order},
        ),
        Component(
            cid=comp_gen.next("C"),
            kind="C",
            node_a="x",
            node_b="n",
            value=c_snap,
            metadata={"motif_id": "fractal_shunt", "depth": order},
        ),
    ]
    subcircuits = [
        SubCircuit(
            name=f"fractal_child_{order}",
            circuit=child,
            port_map={"P": PortConnection(pos="x", neg="n")},
            metadata={"motif_id": "fractal_child", "depth": order},
        )
    ]
    symbols = merge_symbol_maps(symbols_for([r_snap, c_snap]), child.collect_symbols())
    return CircuitIR(
        name="fractal_ladder",
        ports=[port],
        components=components,
        subcircuits=subcircuits,
        symbols=symbols,
        metadata={
            "generator": "fractal_ladder",
            "recursion_depth": order,
            "seed": seed,
            "motif_ids": ["fractal"],
        },
    )


def binary_tree(
    depth: int,
    r_value: ParamValue | float,
    c_value: ParamValue | float,
    scale: float = 0.5,
    seed: Optional[int] = None,
) -> CircuitIR:
    """Generate a binary RC tree (one-port)."""
    if depth < 0:
        raise CircuitIRValidationError("Binary tree depth must be non-negative.")
    r_param = ensure_param(r_value)
    c_param = ensure_param(c_value)
    return _binary_tree_recursive(depth, r_param, c_param, scale, seed)


def _binary_tree_recursive(
    depth: int,
    r_param: ParamValue,
    c_param: ParamValue,
    scale: float,
    seed: Optional[int],
) -> CircuitIR:
    comp_gen = ComponentIdGenerator()
    port = PortDef(name="P", pos="p", neg="n")
    if depth == 0:
        r_snap = snap_param(r_param)
        components = [
            Component(
                cid=comp_gen.next("R"),
                kind="R",
                node_a="p",
                node_b="n",
                value=r_snap,
                metadata={"motif_id": "tree_leaf"},
            )
        ]
        symbols = symbols_for([r_snap])
        return CircuitIR(
            name="binary_tree",
            ports=[port],
            components=components,
            symbols=symbols,
            metadata={
                "generator": "binary_tree",
                "recursion_depth": 0,
                "seed": seed,
                "motif_ids": ["binary_tree"],
            },
        )

    r_snap = snap_param(r_param)
    c_snap = snap_param(c_param)
    components = [
        Component(
            cid=comp_gen.next("R"),
            kind="R",
            node_a="p",
            node_b="l",
            value=r_snap,
            metadata={"motif_id": "tree_branch", "depth": depth},
        ),
        Component(
            cid=comp_gen.next("R"),
            kind="R",
            node_a="p",
            node_b="r",
            value=r_snap,
            metadata={"motif_id": "tree_branch", "depth": depth},
        ),
        Component(
            cid=comp_gen.next("C"),
            kind="C",
            node_a="l",
            node_b="n",
            value=c_snap,
            metadata={"motif_id": "tree_shunt", "depth": depth},
        ),
        Component(
            cid=comp_gen.next("C"),
            kind="C",
            node_a="r",
            node_b="n",
            value=c_snap,
            metadata={"motif_id": "tree_shunt", "depth": depth},
        ),
    ]

    r_scaled = _scaled_symbol(scale_param(r_param, scale), suffix=f"d{depth}")
    c_scaled = _scaled_symbol(scale_param(c_param, scale), suffix=f"d{depth}")
    left = _binary_tree_recursive(depth - 1, r_scaled, c_scaled, scale, seed)
    right = _binary_tree_recursive(depth - 1, r_scaled, c_scaled, scale, seed)
    subcircuits = [
        SubCircuit(
            name=f"tree_left_{depth}",
            circuit=left,
            port_map={"P": PortConnection(pos="l", neg="n")},
            metadata={"motif_id": "tree_left", "depth": depth},
        ),
        SubCircuit(
            name=f"tree_right_{depth}",
            circuit=right,
            port_map={"P": PortConnection(pos="r", neg="n")},
            metadata={"motif_id": "tree_right", "depth": depth},
        ),
    ]
    symbols = merge_symbol_maps(
        symbols_for([r_snap, c_snap]), left.collect_symbols(), right.collect_symbols()
    )
    return CircuitIR(
        name="binary_tree",
        ports=[port],
        components=components,
        subcircuits=subcircuits,
        symbols=symbols,
        metadata={
            "generator": "binary_tree",
            "recursion_depth": depth,
            "seed": seed,
            "motif_ids": ["binary_tree"],
        },
    )


def sierpinski(
    depth: int,
    r_value: ParamValue | float,
    scale: float = 0.5,
    seed: Optional[int] = None,
) -> CircuitIR:
    """Generate a Sierpinski-like triangular resistor motif (one-port)."""
    if depth < 0:
        raise CircuitIRValidationError("Sierpinski depth must be non-negative.")
    r_param = ensure_param(r_value)
    return _sierpinski_recursive(depth, r_param, scale, seed)


def _sierpinski_recursive(
    depth: int,
    r_param: ParamValue,
    scale: float,
    seed: Optional[int],
) -> CircuitIR:
    comp_gen = ComponentIdGenerator()
    port = PortDef(name="P", pos="a", neg="b")
    if depth == 0:
        r_snap = snap_param(r_param)
        components = [
            Component(
                cid=comp_gen.next("R"),
                kind="R",
                node_a="a",
                node_b="b",
                value=r_snap,
                metadata={"motif_id": "sierpinski_edge"},
            ),
            Component(
                cid=comp_gen.next("R"),
                kind="R",
                node_a="b",
                node_b="c",
                value=r_snap,
                metadata={"motif_id": "sierpinski_edge"},
            ),
            Component(
                cid=comp_gen.next("R"),
                kind="R",
                node_a="c",
                node_b="a",
                value=r_snap,
                metadata={"motif_id": "sierpinski_edge"},
            ),
        ]
        symbols = symbols_for([r_snap])
        return CircuitIR(
            name="sierpinski",
            ports=[port],
            components=components,
            symbols=symbols,
            metadata={
                "generator": "sierpinski",
                "recursion_depth": 0,
                "seed": seed,
                "motif_ids": ["sierpinski"],
            },
        )

    r_scaled = _scaled_symbol(scale_param(r_param, scale), suffix=f"d{depth}")
    child = _sierpinski_recursive(depth - 1, r_scaled, scale, seed)
    subcircuits = [
        SubCircuit(
            name=f"sier_ab_{depth}",
            circuit=child,
            port_map={"P": PortConnection(pos="a", neg="b")},
            metadata={"motif_id": "sierpinski_edge", "depth": depth},
        ),
        SubCircuit(
            name=f"sier_bc_{depth}",
            circuit=child,
            port_map={"P": PortConnection(pos="b", neg="c")},
            metadata={"motif_id": "sierpinski_edge", "depth": depth},
        ),
        SubCircuit(
            name=f"sier_ca_{depth}",
            circuit=child,
            port_map={"P": PortConnection(pos="c", neg="a")},
            metadata={"motif_id": "sierpinski_edge", "depth": depth},
        ),
    ]
    symbols = merge_symbol_maps(child.collect_symbols())
    return CircuitIR(
        name="sierpinski",
        ports=[port],
        subcircuits=subcircuits,
        symbols=symbols,
        metadata={
            "generator": "sierpinski",
            "recursion_depth": depth,
            "seed": seed,
            "motif_ids": ["sierpinski"],
        },
    )


def _scaled_symbol(value: ParamValue, suffix: str) -> ParamValue:
    if value.symbol is None:
        return value
    return replace(value, symbol=f"{value.symbol}_{suffix}")
