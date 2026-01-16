import shutil

from fidp.env_check import check_environment


def test_env_check_success(monkeypatch) -> None:
    def _fake_which(name: str) -> str | None:
        if name in ("ngspice", "Xyce", "xyce"):
            return f"/usr/bin/{name}"
        return None

    monkeypatch.setattr(shutil, "which", _fake_which)
    result = check_environment()
    assert result.ok
    assert result.errors == []
    assert result.warnings == []


def test_env_check_missing_executable(monkeypatch) -> None:
    def _missing(_: str) -> None:
        return None

    monkeypatch.setattr(shutil, "which", _missing)
    result = check_environment()
    assert not result.ok
    assert any("Missing required spice executable" in error for error in result.errors)
    assert any("apt-get install -y ngspice" in error for error in result.errors)
    assert any("brew install ngspice" in error for error in result.errors)
    assert any("Windows" in error for error in result.errors)
    assert any("Missing required Xyce executable" in error for error in result.errors)
