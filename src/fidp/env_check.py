"""Environment verification for FIDP dependencies."""

from __future__ import annotations

import importlib
import importlib.metadata
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


REQUIRED_PYTHON: Dict[str, str] = {
    "numpy": ">=1.24,<3",
    "scipy": ">=1.10,<2",
    "torch": ">=2.3,<3",
    "torchvision": ">=0.18,<1",
    "torchaudio": ">=2.3,<3",
    "botorch": ">=0.10,<0.13",
    "gpytorch": ">=1.11,<1.14",
    "pytest": ">=7.4,<9",
    "packaging": ">=23,<25",
    "lark": ">=1.1,<2",
    "networkx": ">=3.2,<4",
    "hypothesis": ">=6.100,<7",
    "pyarrow": ">=12,<17",
    "mlflow": ">=2.12,<3",
    "dvc": ">=3.50,<4",
    "ray": ">=2.9,<3",
    "yaml": ">=6,<7",
}

REQUIRED_EXECUTABLE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "spice": ("ngspice",),
    "xyce": ("Xyce", "xyce"),
}
OPTIONAL_EXECUTABLES: Dict[str, Tuple[str, ...]] = {}

DISTRIBUTION_OVERRIDES: Dict[str, str] = {
    "yaml": "PyYAML",
}


@dataclass(frozen=True)
class EnvCheckResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


def check_environment() -> EnvCheckResult:
    """Check required Python packages and external executables."""
    errors: List[str] = []
    warnings: List[str] = []

    for module, spec in REQUIRED_PYTHON.items():
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - exercised via env
            errors.append(f"Missing Python package: {module} ({exc})")
            continue
        dist_name = DISTRIBUTION_OVERRIDES.get(module, module)
        try:
            version = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Package metadata missing for {module}.")
            continue
        try:
            if spec and not SpecifierSet(spec).contains(Version(version)):
                errors.append(f"{module} version {version} does not satisfy {spec}.")
        except InvalidVersion:
            errors.append(f"Unable to parse version for {module}: {version}.")

    for group, executables in REQUIRED_EXECUTABLE_GROUPS.items():
        if not _any_executable(executables):
            if group == "spice":
                errors.append(_format_spice_missing(executables))
            elif group == "xyce":
                errors.append(_format_xyce_missing(executables))
            else:
                candidates = ", ".join(executables)
                errors.append(f"Missing required {group} executable (one of: {candidates}).")

    for name, executables in OPTIONAL_EXECUTABLES.items():
        if not _any_executable(executables):
            candidates = ", ".join(executables)
            warnings.append(f"Optional executable missing for {name} (one of: {candidates}).")

    return EnvCheckResult(ok=not errors, errors=errors, warnings=warnings)


def _any_executable(executables: Iterable[str]) -> bool:
    return any(shutil.which(executable) for executable in executables)


def _format_spice_missing(executables: Iterable[str]) -> str:
    candidates = ", ".join(executables)
    guidance = [
        f"Missing required spice executable (one of: {candidates}).",
        "Install ngspice and ensure it is on PATH:",
        "  Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y ngspice",
        "  macOS: brew install ngspice",
        "  Windows: install ngspice and add it to PATH",
    ]
    return "\n".join(guidance)


def _format_xyce_missing(executables: Iterable[str]) -> str:
    candidates = ", ".join(executables)
    guidance = [
        f"Missing required Xyce executable (one of: {candidates}).",
        "Install Xyce manually and ensure it is on PATH:",
        "  https://xyce.sandia.gov/",
    ]
    return "\n".join(guidance)


def main() -> None:
    result = check_environment()
    if result.ok:
        print("Environment check passed.")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
        return
    for error in result.errors:
        print(f"ERROR: {error}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
