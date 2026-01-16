import shutil

from fidp.env_check import check_environment


def test_env_check_success() -> None:
    result = check_environment()
    assert result.ok
    assert result.errors == []


def test_env_check_missing_executable(monkeypatch) -> None:
    def _missing(_: str) -> None:
        return None

    monkeypatch.setattr(shutil, "which", _missing)
    result = check_environment()
    assert not result.ok
    assert any("Missing required spice executable" in error for error in result.errors)
