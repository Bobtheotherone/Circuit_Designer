"""Minimal pipeline stage stubs for DVC wiring."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class StageResult:
    """Result summary for a pipeline stage."""

    stage: str
    output: Path
    payload: Dict[str, Any]


def load_params(path: Path) -> Dict[str, Any]:
    """Load pipeline parameters from YAML."""
    if not path.exists():
        raise FileNotFoundError(f"params file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def run_stage(stage: str, params_path: Path, output_path: Path | None = None) -> StageResult:
    """Run a stub pipeline stage and write a JSON summary."""
    params = load_params(params_path)
    stage_params = params.get(stage)
    if stage_params is None:
        raise KeyError(f"Stage '{stage}' not found in {params_path}.")

    resolved_output = Path(output_path or stage_params.get("output", ""))
    if not resolved_output:
        raise ValueError("Output path not provided and missing from params.")

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": stage_params,
    }
    resolved_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return StageResult(stage=stage, output=resolved_output, payload=payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FIDP pipeline stage stub.")
    parser.add_argument("--stage", required=True, help="Stage name to run.")
    parser.add_argument(
        "--params",
        dest="params_path",
        default="params.yaml",
        help="Path to params.yaml.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=None,
        help="Override output path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_stage(args.stage, Path(args.params_path), Path(args.output_path) if args.output_path else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
