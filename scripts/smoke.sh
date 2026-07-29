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
  echo "Python 3 was not found. Create a virtual environment and install the project." >&2
  exit 1
fi

"$python_bin" -m pricing_mapper \
  --budget 20 \
  --init-n 10 \
  --batch-size 5 \
  --pool-size 500 \
  --distance-backend knn \
  --rf-n-models 4 \
  --rf-n-estimators 80 \
  --checkpoint-every-batches 0 \
  --output-dir local_smoke_outputs
