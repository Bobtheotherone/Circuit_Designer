import json
from pathlib import Path

import mlflow

from fidp.experiments import tracking


def test_mlflow_logger_end_to_end(tmp_path, monkeypatch) -> None:
    tracking_uri = tmp_path / "mlruns"
    monkeypatch.setenv("FIDP_MLFLOW_TRACKING_URI", str(tracking_uri))
    monkeypatch.setenv("FIDP_MLFLOW_EXPERIMENT", "fidp-test")

    artifact_path = tmp_path / "artifact.txt"
    artifact_path.write_text("hello", encoding="utf-8")

    with tracking.start_run(run_name="unit-test", stage_name="stage-a"):
        tracking.log_params({"alpha": 0.5, "payload": {"x": 1}})
        tracking.log_metrics({"score": 0.99})
        tracking.log_tags({"purpose": "unit-test"})
        tracking.log_artifact(artifact_path)
        tracking.log_design_record({"spec": "demo", "metadata": {"x": 1}})

    client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_uri))
    experiment = client.get_experiment_by_name("fidp-test")
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    assert run.data.params["alpha"] == "0.5"
    assert json.loads(run.data.params["payload"]) == {"x": 1}
    assert run.data.metrics["score"] == 0.99
    assert run.data.tags["fidp.stage"] == "stage-a"
    assert run.data.tags["purpose"] == "unit-test"
    assert "fidp.python_version" in run.data.tags
    assert run.data.tags["fidp.package"] == "fidp"

    artifacts = client.list_artifacts(run.info.run_id)
    assert any(item.path == "artifact.txt" for item in artifacts)

    design_artifacts = client.list_artifacts(run.info.run_id, "design_record")
    assert any(Path(item.path).name == "design_record.json" for item in design_artifacts)
