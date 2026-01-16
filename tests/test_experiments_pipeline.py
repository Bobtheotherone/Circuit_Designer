import json
from pathlib import Path

import pytest
import yaml

from fidp.experiments import pipeline


def _write_params(tmp_path: Path, stage_params: dict) -> Path:
    params_path = tmp_path / "params.yaml"
    payload = {"stage_a": stage_params}
    params_path.write_text(
        yaml.safe_dump(payload, sort_keys=True),
        encoding="utf-8",
    )
    return params_path


def test_run_stage_missing_output(tmp_path: Path) -> None:
    params_path = _write_params(tmp_path, {"seed": 1})

    with pytest.raises(ValueError, match="No output path provided"):
        pipeline.run_stage("stage_a", params_path)


def test_run_stage_rejects_dot_or_empty_output(tmp_path: Path) -> None:
    params_path = _write_params(tmp_path, {"output": str(tmp_path / "ok.json")})

    for output_value in [".", "", Path("."), Path("")]:
        with pytest.raises(ValueError, match="No output path provided"):
            pipeline.run_stage("stage_a", params_path, output_path=output_value)


def test_run_stage_writes_output(tmp_path: Path) -> None:
    output_path = tmp_path / "results" / "stage.json"
    params_path = _write_params(tmp_path, {"output": str(output_path)})

    result = pipeline.run_stage("stage_a", params_path)

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "stage_a"
    assert result.output == output_path
