#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python -m ruff check src tests scripts
python -m black --check src tests scripts
python -m mypy src/pricing_mapper
PYTHONPATH=src python scripts/generate_schemas.py
git diff --exit-code -- schema
python -m pytest --cov=pricing_mapper --cov-branch --cov-report=term-missing
python -m build
python -m twine check dist/*
