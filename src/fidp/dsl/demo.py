"""Demo entry point for the circuit DSL pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from fidp.circuits.canonical import canonicalize_circuit
from fidp.circuits.ir_export import circuit_to_json, export_spice_netlist, validate_spice_netlist
from fidp.dsl import compile_dsl, parse_dsl


_SAMPLE_DSL = """circuit Demo {
  ports: (P);
  body: series(R(100),C(1e-6),R(200));
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="FIDP DSL demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dsl", type=str, help="DSL string to parse")
    group.add_argument("--file", type=Path, help="Path to DSL file")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/dsl_demo"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.dsl:
        text = args.dsl
    else:
        text = _SAMPLE_DSL

    program = parse_dsl(text)
    circuit = compile_dsl(program, seed=args.seed)
    canonical = canonicalize_circuit(circuit)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{program.name}_{canonical.canonical_hash}"
    json_path = out_dir / f"{stem}.json"
    spice_path = out_dir / f"{stem}.cir"

    json_path.write_text(circuit_to_json(circuit), encoding="utf-8")
    netlist = export_spice_netlist(circuit, title=program.name)
    validate_spice_netlist(netlist)
    spice_path.write_text(netlist, encoding="utf-8")

    print(f"Canonical hash: {canonical.canonical_hash}")
    print(f"JSON: {json_path}")
    print(f"SPICE: {spice_path}")


if __name__ == "__main__":
    main()
