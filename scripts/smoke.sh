#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

smoke_root="$(mktemp -d)"
cleanup() {
  rm -rf "$smoke_root"
}
trap cleanup EXIT

python -m build --wheel --outdir "$smoke_root/dist"
python -m venv "$smoke_root/venv"
"$smoke_root/venv/bin/python" -m pip install "$smoke_root"/dist/*.whl

sed "s|output_dir = \"outputs\"|output_dir = \"$smoke_root/outputs\"|" \
  config.smoke.toml > "$smoke_root/config.toml"

"$smoke_root/venv/bin/pricing-mapper" config validate \
  --config "$smoke_root/config.toml"
"$smoke_root/venv/bin/pricing-mapper" map run \
  --config "$smoke_root/config.toml" \
  --run-id wheel-smoke
"$smoke_root/venv/bin/pricing-mapper" artifact inspect \
  --artifact "$smoke_root/outputs/wheel-smoke"
"$smoke_root/venv/bin/pricing-mapper" predict row \
  --artifact "$smoke_root/outputs/wheel-smoke" \
  --input examples/quote-row.json
"$smoke_root/venv/bin/pricing-mapper" model evaluate \
  --artifact "$smoke_root/outputs/wheel-smoke"
