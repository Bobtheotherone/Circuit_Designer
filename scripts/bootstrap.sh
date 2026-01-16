#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found in PATH." >&2
  exit 1
fi

PY_VERSION=$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
REQUIRED="3.10"
if [ "$(printf '%s\n' "$REQUIRED" "$PY_VERSION" | sort -V | head -n1)" != "$REQUIRED" ]; then
  echo "ERROR: Python >= $REQUIRED is required (found $PY_VERSION)." >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
VENV_PYTHON=".venv/bin/python"
VENV_PIP=".venv/bin/pip"
$VENV_PYTHON -m pip install --upgrade pip

has_nvidia=false
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L >/dev/null 2>&1; then
    has_nvidia=true
  fi
elif command -v lspci >/dev/null 2>&1; then
  if lspci | grep -i nvidia >/dev/null 2>&1; then
    has_nvidia=true
  fi
fi

if [ "$has_nvidia" = true ]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
  TORCH_MODE="cu128"
else
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
  TORCH_MODE="cpu"
fi

echo "Installing PyTorch ($TORCH_MODE)..."
LOCK_FILE="requirements/requirements-${TORCH_MODE}.lock"
if [ ! -f "$LOCK_FILE" ]; then
  echo "ERROR: Missing lock file: $LOCK_FILE" >&2
  exit 1
fi

$VENV_PYTHON -m pip install --upgrade --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
echo "Installing locked dependencies from $LOCK_FILE..."
$VENV_PYTHON -m pip install --upgrade --index-url "$TORCH_INDEX_URL" \
  --extra-index-url https://pypi.org/simple -r "$LOCK_FILE"

$VENV_PYTHON -m pip install -e .

$VENV_PYTHON - <<'PY'
import torch
print(f"torch {torch.__version__}")
PY

if [ "$has_nvidia" = true ]; then
  $VENV_PYTHON - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: NVIDIA GPU detected but torch.cuda.is_available() is False")
print("torch.cuda.is_available() True")
PY
fi

if ! command -v ngspice >/dev/null 2>&1 && ! command -v Xyce >/dev/null 2>&1 && ! command -v xyce >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    echo "ngspice not found; attempting to install via apt."
    if ! sudo apt-get update; then
      echo "ERROR: apt-get update failed. Install ngspice manually:" >&2
      echo "  sudo apt-get update && sudo apt-get install -y ngspice" >&2
      exit 1
    fi
    if ! sudo apt-get install -y ngspice; then
      echo "ERROR: apt-get install ngspice failed. Install ngspice manually:" >&2
      echo "  sudo apt-get install -y ngspice" >&2
      exit 1
    fi
  else
    echo "ERROR: ngspice not found and automatic install unavailable." >&2
    echo "Install ngspice manually (Debian/Ubuntu):" >&2
    echo "  sudo apt-get update && sudo apt-get install -y ngspice" >&2
    exit 1
  fi
fi

$VENV_PYTHON -m fidp.env_check
