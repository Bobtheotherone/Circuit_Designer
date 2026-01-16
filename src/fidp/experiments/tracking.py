"""MLflow tracking helpers for FIDP experiments."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

TRACKING_URI_ENV = "FIDP_MLFLOW_TRACKING_URI"
EXPERIMENT_ENV = "FIDP_MLFLOW_EXPERIMENT"
DEFAULT_TRACKING_URI = "mlruns"
DEFAULT_EXPERIMENT = "fidp"


class MLflowUnavailableError(RuntimeError):
    """Raised when MLflow is required but unavailable."""


def _load_mlflow():
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - exercised via env
        raise MLflowUnavailableError(
            "MLflow is required for experiment tracking. Install mlflow and retry."
        ) from exc
    return mlflow


def get_tracking_uri() -> str:
    """Return the configured MLflow tracking URI."""
    return os.environ.get(TRACKING_URI_ENV, DEFAULT_TRACKING_URI)


def set_tracking_uri(uri: str) -> None:
    """Set the MLflow tracking URI for subsequent runs."""
    os.environ[TRACKING_URI_ENV] = uri


def get_experiment_name() -> str:
    """Return the configured MLflow experiment name."""
    return os.environ.get(EXPERIMENT_ENV, DEFAULT_EXPERIMENT)


def set_experiment_name(name: str) -> None:
    """Set the MLflow experiment name for subsequent runs."""
    os.environ[EXPERIMENT_ENV] = name


def _get_git_sha() -> Optional[str]:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        sha = output.decode().strip()
        return sha or None
    except Exception:
        return None


def _get_fidp_version() -> Optional[str]:
    try:
        return importlib.metadata.version("fidp")
    except importlib.metadata.PackageNotFoundError:
        return None


def _auto_tags(stage_name: Optional[str]) -> Dict[str, str]:
    tags: Dict[str, str] = {
        "fidp.platform": platform.platform(),
        "fidp.python_version": platform.python_version(),
        "fidp.package": "fidp",
    }
    version = _get_fidp_version()
    if version:
        tags["fidp.version"] = version
    sha = _get_git_sha()
    if sha:
        tags["fidp.git_sha"] = sha
    if stage_name:
        tags["fidp.stage"] = stage_name
    return tags


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def _stringify_param(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, default=_json_default, ensure_ascii=True, sort_keys=True)


def _summarize_graph(graph: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"repr": repr(graph)}
    if hasattr(graph, "components"):
        summary["component_count"] = len(getattr(graph, "components"))
    if hasattr(graph, "ports"):
        summary["port_count"] = len(getattr(graph, "ports"))
    if hasattr(graph, "nodes"):
        summary["node_count"] = len(getattr(graph, "nodes"))
    return summary


def _coerce_design_record(record: Any) -> Dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)

    payload: Dict[str, Any] = {}
    for field_name in ("spec", "metadata", "parameters", "global_features", "novelty"):
        if hasattr(record, field_name):
            payload[field_name] = getattr(record, field_name)
    if hasattr(record, "graph"):
        payload["graph_summary"] = _summarize_graph(getattr(record, "graph"))
    if not payload:
        payload["repr"] = repr(record)
    return payload


@contextmanager
def start_run(
    run_name: Optional[str] = None,
    stage_name: Optional[str] = None,
    tags: Optional[Mapping[str, str]] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
):
    """Start an MLflow run with FIDP-standard tags."""
    mlflow = _load_mlflow()
    uri = tracking_uri or get_tracking_uri()
    exp_name = experiment_name or get_experiment_name()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp_name)

    merged_tags = _auto_tags(stage_name)
    if tags:
        merged_tags.update({key: str(value) for key, value in tags.items()})

    with mlflow.start_run(run_name=run_name) as run:
        if merged_tags:
            mlflow.set_tags(merged_tags)
        yield run


def log_params(params: Mapping[str, Any]) -> None:
    """Log parameters to the active MLflow run."""
    mlflow = _load_mlflow()
    payload = {key: _stringify_param(value) for key, value in params.items()}
    if payload:
        mlflow.log_params(payload)


def log_metrics(metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to the active MLflow run."""
    mlflow = _load_mlflow()
    payload = {key: float(value) for key, value in metrics.items()}
    if payload:
        mlflow.log_metrics(payload, step=step)


def log_tags(tags: Mapping[str, Any]) -> None:
    """Log tags to the active MLflow run."""
    mlflow = _load_mlflow()
    payload = {key: str(value) for key, value in tags.items()}
    if payload:
        mlflow.set_tags(payload)


def log_artifact(path: Path | str) -> None:
    """Log a file or directory as an MLflow artifact."""
    mlflow = _load_mlflow()
    artifact_path = Path(path)
    if artifact_path.is_dir():
        mlflow.log_artifacts(str(artifact_path))
    else:
        mlflow.log_artifact(str(artifact_path))


def log_design_record(record: Any, artifact_path: str = "design_record") -> None:
    """Log a lightweight design record snapshot as JSON."""
    mlflow = _load_mlflow()
    payload = _coerce_design_record(record)
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "design_record.json"
        output_path.write_text(
            json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(output_path), artifact_path=artifact_path)
