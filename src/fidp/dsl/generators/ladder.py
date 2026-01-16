"""Ladder generator motifs."""

from __future__ import annotations

from typing import Dict, List, Optional

from fidp.circuits.ir import CircuitIR, Component, ParamValue, PortDef
from fidp.dsl.generators.utils import ComponentIdGenerator, ensure_param, scale_param, snap_param, symbols_for
from fidp.errors import CircuitIRValidationError


def domino_ladder(
    stages: int,
    r_value: ParamValue | float,
    c_value: ParamValue | float,
    scale: float = 1.0,
    seed: Optional[int] = None,
) -> CircuitIR:
    """Generate a classic domino RC ladder (one-port)."""
    if stages <= 0:
        raise CircuitIRValidationError("Domino ladder stages must be positive.")
    r_param = ensure_param(r_value)
    c_param = ensure_param(c_value)
    if scale != 1.0:
        r_param = scale_param(r_param, scale)
        c_param = scale_param(c_param, scale)
    r_param = snap_param(r_param)
    c_param = snap_param(c_param)

    nodes = [f"n{i}" for i in range(stages + 1)]
    ground = "gnd"
    port = PortDef(name="P", pos=nodes[0], neg=ground)

    comp_gen = ComponentIdGenerator()
    components: List[Component] = []
    for idx in range(stages):
        components.append(
            Component(
                cid=comp_gen.next("R"),
                kind="R",
                node_a=nodes[idx],
                node_b=nodes[idx + 1],
                value=r_param,
                metadata={"motif_id": "domino_resistor", "stage": idx},
            )
        )
        components.append(
            Component(
                cid=comp_gen.next("C"),
                kind="C",
                node_a=nodes[idx + 1],
                node_b=ground,
                value=c_param,
                metadata={"motif_id": "domino_shunt", "stage": idx},
            )
        )

    symbols = symbols_for([r_param, c_param])
    return CircuitIR(
        name="domino_ladder",
        ports=[port],
        components=components,
        symbols=symbols,
        metadata={
            "generator": "domino_ladder",
            "recursion_depth": 0,
            "seed": seed,
            "motif_ids": ["domino"],
        },
    )


def cross_ladder(
    stages: int,
    r_value: ParamValue | float,
    c_value: ParamValue | float,
    scale: float = 1.0,
    seed: Optional[int] = None,
) -> CircuitIR:
    """Generate a cross-ladder RC network (two-port)."""
    if stages <= 0:
        raise CircuitIRValidationError("Cross ladder stages must be positive.")
    r_param = ensure_param(r_value)
    c_param = ensure_param(c_value)
    if scale != 1.0:
        r_param = scale_param(r_param, scale)
        c_param = scale_param(c_param, scale)
    r_param = snap_param(r_param)
    c_param = snap_param(c_param)

    top = [f"t{i}" for i in range(stages + 1)]
    bottom = [f"b{i}" for i in range(stages + 1)]
    ports = [
        PortDef(name="P1", pos=top[0], neg=bottom[0]),
        PortDef(name="P2", pos=top[-1], neg=bottom[-1]),
    ]

    comp_gen = ComponentIdGenerator()
    components: List[Component] = []
    for idx in range(stages):
        components.append(
            Component(
                cid=comp_gen.next("R"),
                kind="R",
                node_a=top[idx],
                node_b=top[idx + 1],
                value=r_param,
                metadata={"motif_id": "cross_top", "stage": idx},
            )
        )
        components.append(
            Component(
                cid=comp_gen.next("R"),
                kind="R",
                node_a=bottom[idx],
                node_b=bottom[idx + 1],
                value=r_param,
                metadata={"motif_id": "cross_bottom", "stage": idx},
            )
        )
        components.append(
            Component(
                cid=comp_gen.next("C"),
                kind="C",
                node_a=top[idx + 1],
                node_b=bottom[idx + 1],
                value=c_param,
                metadata={"motif_id": "cross_shunt", "stage": idx},
            )
        )

    symbols = symbols_for([r_param, c_param])
    return CircuitIR(
        name="cross_ladder",
        ports=ports,
        components=components,
        symbols=symbols,
        metadata={
            "generator": "cross_ladder",
            "recursion_depth": 0,
            "seed": seed,
            "motif_ids": ["cross"],
        },
    )
