"""Circuit generator registry."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from fidp.circuits.ir import CircuitIR
from fidp.dsl.generators.fractal import binary_tree, fractal_ladder, sierpinski
from fidp.dsl.generators.ladder import cross_ladder, domino_ladder
from fidp.dsl.generators.lattice import lattice_grid


GeneratorFn = Callable[..., CircuitIR]


_REGISTRY: Dict[str, GeneratorFn] = {
    "domino_ladder": domino_ladder,
    "cross_ladder": cross_ladder,
    "fractal_ladder": fractal_ladder,
    "binary_tree": binary_tree,
    "sierpinski": sierpinski,
    "lattice_grid": lattice_grid,
}


def generate(name: str, **kwargs) -> Optional[CircuitIR]:
    """Lookup and invoke a generator by name."""
    fn = _REGISTRY.get(name)
    if fn is None:
        return None
    return fn(**kwargs)
