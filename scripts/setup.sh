#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
if [[ ! -x "$UV_BIN" ]]; then
  echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ or set UV_BIN" >&2
  exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/supermimic-repro}"
# Managed Python includes Python.h required by Triton; no sudo/system install.
"$UV_BIN" python install 3.12
"$UV_BIN" sync --frozen --group dev --python 3.12 --managed-python
