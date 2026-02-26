#!/usr/bin/env bash
set -euo pipefail

ruff check pricing_mapper tests comp_car_active_mapper_advanced.py
black --check pricing_mapper tests comp_car_active_mapper_advanced.py
mypy pricing_mapper
python -m pytest -q
