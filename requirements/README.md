# Requirements Locks

This repo uses pip-tools to produce deterministic lock files for CPU-only and CUDA 12.8 (cu128)
installs. The bootstrap script selects the correct lock based on NVIDIA GPU detection.

## Regenerate lock files

Install pip-tools in your active environment:

```bash
python -m pip install --upgrade pip-tools
```

Then regenerate the locks:

```bash
pip-compile requirements/requirements.in \
  --output-file requirements/requirements-cpu.lock \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

pip-compile requirements/requirements.in \
  --output-file requirements/requirements-cu128.lock \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

## Bootstrap behavior

- Detects NVIDIA GPU (via `nvidia-smi` or `lspci`) to choose `requirements-cpu.lock` or
  `requirements-cu128.lock`.
- Installs PyTorch from the matching PyTorch index, then installs the lock file and the repo.
- Ensures a real SPICE executable (`ngspice` or `Xyce/xyce`) is available. On Debian/Ubuntu it
  will attempt `sudo apt-get update && sudo apt-get install -y ngspice` and fail with instructions
  if that is not possible.
