"""Lattice generator motifs."""

from __future__ import annotations

from typing import List, Optional

from fidp.circuits.ir import CircuitIR, Component, ParamValue, PortDef
from fidp.dsl.generators.utils import ComponentIdGenerator, ensure_param, snap_param, symbols_for
from fidp.errors import CircuitIRValidationError


def lattice_grid(
    rows: int,
    cols: int,
    r_value: ParamValue | float,
    c_value: ParamValue | float | None = None,
    seed: Optional[int] = None,
) -> CircuitIR:
    """Generate a resistor lattice with optional capacitive cross-links."""
    if rows < 2 or cols < 2:
        raise CircuitIRValidationError("Lattice grid requires rows>=2 and cols>=2.")
    r_param = snap_param(ensure_param(r_value))
    c_param = snap_param(ensure_param(c_value)) if c_value is not None else None

    nodes = [[f"n{r}_{c}" for c in range(cols)] for r in range(rows)]

    comp_gen = ComponentIdGenerator()
    components: List[Component] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                components.append(
                    Component(
                        cid=comp_gen.next("R"),
                        kind="R",
                        node_a=nodes[r][c],
                        node_b=nodes[r][c + 1],
                        value=r_param,
                        metadata={"motif_id": "lattice_horizontal"},
                    )
                )
                if c_param is not None:
                    components.append(
                        Component(
                            cid=comp_gen.next("C"),
                            kind="C",
                            node_a=nodes[r][c],
                            node_b=nodes[r][c + 1],
                            value=c_param,
                            metadata={"motif_id": "lattice_cap_horizontal"},
                        )
                    )
            if r + 1 < rows:
                components.append(
                    Component(
                        cid=comp_gen.next("R"),
                        kind="R",
                        node_a=nodes[r][c],
                        node_b=nodes[r + 1][c],
                        value=r_param,
                        metadata={"motif_id": "lattice_vertical"},
                    )
                )

    ports = [
        PortDef(name="P1", pos=nodes[0][0], neg=nodes[-1][0]),
        PortDef(name="P2", pos=nodes[0][-1], neg=nodes[-1][-1]),
    ]

    values = [r_param] + ([c_param] if c_param is not None else [])
    symbols = symbols_for(values)
    return CircuitIR(
        name="lattice_grid",
        ports=ports,
        components=components,
        symbols=symbols,
        metadata={
            "generator": "lattice_grid",
            "recursion_depth": 0,
            "seed": seed,
            "motif_ids": ["lattice"],
        },
    )
