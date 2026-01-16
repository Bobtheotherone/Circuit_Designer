"""Generator utility helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, Optional

from fidp.circuits.ir import ParamSymbol, ParamValue
from fidp.circuits.ops import apply_snap
from fidp.errors import CircuitIRValidationError


class ComponentIdGenerator:
    def __init__(self) -> None:
        self._counts = {"R": 0, "C": 0, "L": 0}

    def next(self, kind: str) -> str:
        if kind not in self._counts:
            raise CircuitIRValidationError(f"Unsupported component kind: {kind}")
        self._counts[kind] += 1
        return f"{kind}{self._counts[kind]}"


def ensure_param(value: ParamValue | float) -> ParamValue:
    if isinstance(value, ParamValue):
        return value
    return ParamValue(nominal=float(value))


def scale_param(value: ParamValue, factor: float) -> ParamValue:
    return replace(
        value,
        nominal=value.nominal * factor,
        min_value=value.min_value * factor if value.min_value is not None else None,
        max_value=value.max_value * factor if value.max_value is not None else None,
        snapped=value.snapped * factor if value.snapped is not None else None,
    )


def snap_param(value: ParamValue) -> ParamValue:
    return apply_snap(value)


def symbols_for(values: Iterable[ParamValue]) -> Dict[str, ParamSymbol]:
    symbols: Dict[str, ParamSymbol] = {}
    for value in values:
        if value.symbol is None:
            continue
        symbol = ParamSymbol(
            name=value.symbol,
            nominal=value.nominal,
            min_value=value.min_value,
            max_value=value.max_value,
            snap=value.snap,
            snapped=value.snapped,
        )
        if value.symbol in symbols and symbols[value.symbol] != symbol:
            raise CircuitIRValidationError(f"Conflicting symbol definition for {value.symbol}.")
        symbols[value.symbol] = symbol
    return symbols


def merge_symbol_maps(*maps: Dict[str, ParamSymbol]) -> Dict[str, ParamSymbol]:
    merged: Dict[str, ParamSymbol] = {}
    for mapping in maps:
        for key, symbol in mapping.items():
            if key in merged and merged[key] != symbol:
                raise CircuitIRValidationError(f"Conflicting symbol definition for {key}.")
            merged[key] = symbol
    return merged
