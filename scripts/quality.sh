#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  python_bin="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${project_root}/.venv/bin/python" ]]; then
  python_bin="${project_root}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  echo "Python 3 was not found. Create a virtual environment and install .[dev]." >&2
  exit 1
fi

"$python_bin" -m ruff check pricing_mapper tests comp_car_active_mapper_advanced.py
"$python_bin" -m black --check pricing_mapper tests comp_car_active_mapper_advanced.py
"$python_bin" -m mypy pricing_mapper
"$python_bin" -m pytest -q
