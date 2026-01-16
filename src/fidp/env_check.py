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
}

REQUIRED_EXECUTABLE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "spice": ("ngspice", "Xyce", "xyce"),
}


@dataclass(frozen=True)
class EnvCheckResult:
    ok: bool
    errors: List[str]


def check_environment() -> EnvCheckResult:
    """Check required Python packages and external executables."""
    errors: List[str] = []

    for module, spec in REQUIRED_PYTHON.items():
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - exercised via env
            errors.append(f"Missing Python package: {module} ({exc})")
            continue
        try:
            version = importlib.metadata.version(module)
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
            else:
                candidates = ", ".join(executables)
                errors.append(f"Missing required {group} executable (one of: {candidates}).")

    return EnvCheckResult(ok=not errors, errors=errors)


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


def main() -> None:
    result = check_environment()
    if result.ok:
        print("Environment check passed.")
        return
    for error in result.errors:
        print(f"ERROR: {error}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
