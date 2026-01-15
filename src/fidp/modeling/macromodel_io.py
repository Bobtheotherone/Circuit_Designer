"""Serialization helpers for rational macromodels."""

from __future__ import annotations

import json
from pathlib import Path

from fidp.modeling.vector_fit import RationalModel


def save_rational_model_json(model: RationalModel, path: str) -> None:
    """Save a rational model to JSON."""
    data = model.to_json_dict()
    target = Path(path)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_rational_model_json(path: str) -> RationalModel:
    """Load a rational model from JSON."""
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    return RationalModel.from_json_dict(data)


def format_pole_residue_text(model: RationalModel) -> str:
    """Return a human-readable pole-residue listing."""
    lines = [
        f"Kind: {model.kind}",
        f"d: {model.d.real:.6e} {model.d.imag:+.6e}j",
        f"h: {model.h.real:.6e} {model.h.imag:+.6e}j",
        "Poles/Residues:",
    ]
    for idx, (pole, residue) in enumerate(zip(model.poles, model.residues)):
        pole_str = f"{pole.real:.6e} {pole.imag:+.6e}j"
        res_str = f"{residue.real:.6e} {residue.imag:+.6e}j"
        lines.append(f"  [{idx}] pole={pole_str} residue={res_str}")
    return "\n".join(lines)
