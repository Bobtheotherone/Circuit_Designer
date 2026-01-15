from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py310 fallback
    import tomli as tomllib


def test_pyproject_setuptools_src_layout():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    tool = data.get("tool", {})
    setuptools = tool.get("setuptools", {})
    assert setuptools.get("package-dir") == {"": "src"}

    packages = setuptools.get("packages", {})
    find = packages.get("find", {})
    assert find.get("where") == ["src"]
