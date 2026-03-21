#!/usr/bin/env bash
set -euo pipefail

python -m pricing_mapper \
  --budget 20 \
  --init-n 10 \
  --batch-size 5 \
  --pool-size 500 \
  --distance-backend knn \
  --rf-n-models 4 \
  --rf-n-estimators 80 \
  --checkpoint-every-batches 0 \
  --output-dir local_smoke_outputs
