import importlib
import shutil

from fidp import env_check


def test_env_check_reports_missing_mlflow(monkeypatch) -> None:
    original_import = importlib.import_module

    def _fake_import(name: str, package: str | None = None):
        if name == "mlflow":
            raise ImportError("mlflow not installed")
        return original_import(name, package=package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/ngspice")

    result = env_check.check_environment()
    assert not result.ok
    assert any("mlflow" in error.lower() for error in result.errors)
