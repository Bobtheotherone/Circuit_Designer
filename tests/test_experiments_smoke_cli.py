import json
from pathlib import Path

import mlflow

from fidp.experiments import smoke


def test_smoke_cli_creates_manifest(tmp_path, monkeypatch) -> None:
    tracking_uri = tmp_path / "mlruns"
    monkeypatch.setenv("FIDP_MLFLOW_TRACKING_URI", str(tracking_uri))
    monkeypatch.setenv("FIDP_MLFLOW_EXPERIMENT", "fidp-smoke")

    output_path = tmp_path / "manifest.json"
    result = smoke.main(["--output", str(output_path), "--stage", "smoke-test"])
    assert result == 0

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "smoke-test"

    client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_uri))
    experiment = client.get_experiment_by_name("fidp-smoke")
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.tags["fidp.stage"] == "smoke-test"
    assert run.data.metrics["artifact_written"] == 1.0

    artifacts = client.list_artifacts(run.info.run_id)
    assert any(Path(item.path).name == "manifest.json" for item in artifacts)
