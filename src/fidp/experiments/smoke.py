"""Reproducibility smoke test for MLflow tracking."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fidp.experiments import tracking


@dataclass(frozen=True)
class SmokeResult:
    """Result summary for the reproducibility smoke test."""

    output_path: Path
    payload: Dict[str, Any]


def build_manifest(stage: str) -> Dict[str, Any]:
    """Build a minimal manifest payload with environment info."""
    return {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def run_smoke(output_path: Path, stage: str) -> SmokeResult:
    """Run a smoke experiment and log it to MLflow."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_manifest(stage)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with tracking.start_run(run_name="repro_smoke", stage_name=stage):
        tracking.log_params({"output": str(output_path), "stage": stage})
        tracking.log_metrics({"artifact_written": 1.0})
        tracking.log_tags({"purpose": "repro-smoke"})
        tracking.log_artifact(output_path)

    return SmokeResult(output_path=output_path, payload=payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FIDP reproducibility smoke test.")
    parser.add_argument(
        "--output",
        default="data/repro_smoke/manifest.json",
        help="Path to the manifest JSON output.",
    )
    parser.add_argument(
        "--stage",
        default="repro_smoke",
        help="Stage name to tag in MLflow.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_smoke(Path(args.output), args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
