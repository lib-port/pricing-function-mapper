# Pricing Function Mapper

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=111111)](https://github.com/astral-sh/ruff)
[![Type check: mypy](https://img.shields.io/badge/type%20check-mypy-2A6DB2)](https://github.com/python/mypy)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://github.com/pytest-dev/pytest)

Active-learning mapper for a pricing function (default: mock comprehensive car insurance quote model).

This project incrementally samples the input space, trains surrogate models, and selects high-value next queries to approximate the behavior of a target quote/pricing function.

## Highlights

- Active-learning query strategy with uncertainty, boundary, local-error, and breakpoint components.
- Optional staged mapping and early stopping to reduce query volume.
- Segment-focused acquisition for targeted analysis (e.g., low-risk/high-premium cohorts).
- Reproducible artifacts: dataset, metadata, checkpoint state, and serialized pricing engine.

## Purpose

The mapper is designed to:

- Learn pricing behavior with fewer queries than naive random sampling.
- Concentrate samples near uncertain regions and likely decision boundaries.
- Produce reproducible datasets and run metadata for analysis.
- Support resumable runs and benchmark comparisons across strategy settings.

## High-Level Approach

For each run:

1. Generate an initial sample set from the domain.
2. Query the pricing function and cache results.
3. Train surrogate models:
   - Bootstrapped Random Forest ensemble (mean + uncertainty)
   - Optional monotonic HistGradientBoosting model (mean prediction)
4. Propose next batch using an acquisition mix:
   - uncertainty sampling
   - boundary refinement
   - error-driven local exploration
   - breakpoint probing
5. Repeat until budget is reached or early-stop criteria are met.
6. Write artifacts (dataset, metadata, state checkpoint, pricing engine).

## Repository Structure

- `pricing_mapper/domain.py`: variable definitions, domain construction, canonicalization.
- `pricing_mapper/encoding.py`: cached feature encoding pipeline.
- `pricing_mapper/models.py`: surrogate model wrappers.
- `pricing_mapper/active_mapper.py`: active-learning loop, persistence, profiling.
- `pricing_mapper/quote.py`: default quote function + pluggable provider loader.
- `pricing_mapper/cli.py`: CLI entrypoint.
- `pricing_mapper/engine.py`: serialized pricing engine artifact + inference helpers.
- `pricing_mapper/api.py`: optional FastAPI serving interface for engine inference.
- `pricing_mapper/benchmark.py`: benchmark presets and result writer.
- `config.example.json`: baseline configuration template.
- `config.segment.example.json`: segment-focused preset (low-risk/high-premium).
- `scripts/quality.sh` and `scripts/smoke.sh`: local quality and smoke helpers.
- `tests/`: unit and integration tests.

## Requirements

- Python 3.11+
- `numpy`, `pandas`, `scikit-learn`

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development tools:

```bash
pip install -e .[dev]
pre-commit install
```

## Quick Start

Default run:

```bash
python -m pricing_mapper
```

This creates a run folder under `outputs/<run_id>/` with:

- `comp_car_quotes_advanced.csv`
- `run_metadata.json`
- `run_state.json`
- `pricing_engine.pkl`

## CLI Usage

### Validate config only (no training)

```bash
python -m pricing_mapper --config config.example.json --dry-run
```

### Run with explicit overrides

```bash
python -m pricing_mapper \
  --budget 260 \
  --init-n 95 \
  --batch-size 20 \
  --pool-size 14000 \
  --distance-backend knn \
  --output-dir outputs \
  --run-id my_run
```

### Segment-focused mapping (low-risk, high-premium)

```bash
python -m pricing_mapper \
  --config config.example.json \
  --segment-focus-enabled \
  --segment-constraints '{"claims_5y":{"max":1},"convictions_5y":{"max":0},"postcode_risk":{"max":0.35},"parking":{"in":["garage","driveway"]}}' \
  --segment-target-weight 0.45 \
  --segment-sigma-weight 0.25
```

Or run the ready-made preset:

```bash
python -m pricing_mapper --config config.segment.example.json
```

### Early-stop + staged mapping example

```bash
python -m pricing_mapper \
  --config config.example.json \
  --staged-mapping-enabled \
  --staged-stage1-fraction 0.4 \
  --early-stop-patience-batches 3 \
  --early-stop-min-batches 4 \
  --early-stop-min-rel-improvement 0.005
```

### Resume a run

```bash
python -m pricing_mapper --config config.example.json --resume
```

Resume safety checks:

- Resume now validates that checkpointed state is compatible with the current run config.
- At minimum, `quote_provider` and `domain_overrides` must match the saved state.

### Benchmark presets

```bash
python -m pricing_mapper \
  --config config.example.json \
  --benchmark \
  --benchmark-output benchmark_results.json
```

This writes:

- `benchmark_results.json`
- `benchmark_results.csv`

### Use your own quote provider

```bash
python -m pricing_mapper --quote-provider my_module.my_quotes:quote
```

Provider contract:

- Callable signature: `quote(row: dict[str, Any]) -> float`
- Must be deterministic for reproducibility unless intentionally stochastic.

### Price a single row from a JSON string

```bash
python -m pricing_mapper \
  --engine-path outputs/<run_id>/pricing_engine.pkl \
  --price-row '{"driver_age":40,"years_licensed":20,"vehicle_year":2022,"vehicle_value":35000,"annual_km":10000,"claims_5y":0,"convictions_5y":0,"postcode_risk":0.2,"theft_risk":0.2,"excess":700,"usage":"private","parking":"garage","hire_car":"none","windscreen":"no","rating":"market"}'
```

### Price a single row from a JSON file

```bash
python -m pricing_mapper \
  --engine-path outputs/<run_id>/pricing_engine.pkl \
  --price-row-json row.json
```

### Price a batch CSV

```bash
python -m pricing_mapper \
  --engine-path outputs/<run_id>/pricing_engine.pkl \
  --price-input-csv input_rows.csv \
  --price-output-csv priced_rows.csv
```

### Serve the pricing engine as an API (optional)

Install optional dependencies:

```bash
pip install -e .[api]
```

Run API server:

```bash
python -m pricing_mapper \
  --engine-path outputs/<run_id>/pricing_engine.pkl \
  --serve-api \
  --host 127.0.0.1 \
  --port 8000
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /price` (JSON object row)
- `POST /price-batch` (`{"rows":[...]}`)

## Configuration

Use `config.example.json` as a template. Key options include:

- Sampling controls: `budget`, `init_n`, `batch_size`, `pool_size`
- Model controls: `rf_n_models`, `rf_n_estimators`, `refit_every_batches`
- Search behavior: `distance_backend`, `acquisition_mix`, `breakpoint_vars`
- Early stop controls: `early_stop_patience_batches`, `early_stop_min_batches`, `early_stop_min_rel_improvement`
- Staged mapping controls: `staged_mapping_enabled`, `staged_stage1_fraction`, `staged_focus_jitter_per_anchor`
- Segment targeting controls: `segment_focus_enabled`, `segment_constraints`, `segment_target_weight`, `segment_sigma_weight`, `segment_min_candidates`, `segment_pool_max_tries`
- Performance controls: `cv_subsample_max`
- Artifacts: `output_dir`, `run_id`, `output_csv`, `output_metadata_json`, `state_path`
- Engine artifact: `engine_path`
- Resume/checkpoint: `resume`, `checkpoint_every_batches`
- Provider: `quote_provider`
- Domain scoping: `domain_overrides`

Validation behavior:

- Unknown keys in `domain_overrides` are rejected (no silent ignore).
- `rf_n_models` and `rf_n_estimators` must be `> 0`; `rf_n_jobs` cannot be `0`.
- `segment_constraints` validate variable names and operator compatibility (`min`/`max`/`eq`/`in`).

## Output Artifacts

### Dataset CSV

Contains sampled rows and true `premium` values queried from the provider.

### Metadata JSON

Includes:

- run stats and elapsed time
- completion flags (`completed_budget`, `early_stopped`, `stop_reason`)
- per-phase profiling times
- MAE diagnostics
- optional segment-specific MAE diagnostics when `segment_constraints` are set
- resolved config
- feature list
- artifact paths

### State JSON

Checkpoint state used for resume support.

- versioned schema
- atomic writes
- migration support for older schema versions

### Pricing Engine (`pricing_engine.pkl`)

Serialized inference artifact containing:

- domain schema and canonicalization behavior
- fitted surrogate model(s)
- feature ordering metadata
- run config snapshot

## Quality and Testing

Run all local quality checks:

```bash
./scripts/quality.sh
```

Equivalent manual commands:

```bash
ruff check pricing_mapper tests comp_car_active_mapper_advanced.py
black --check pricing_mapper tests comp_car_active_mapper_advanced.py
mypy pricing_mapper
python -m pytest -q
```

Run a quick local smoke execution:

```bash
./scripts/smoke.sh
```

## Running in a Debian Vagrant VM

The following creates an isolated Debian VM suitable for running this project.

### 1. Install prerequisites on host

- Vagrant
- VirtualBox (or another Vagrant provider)

### 2. Create `Vagrantfile`

In the project root:

```ruby
Vagrant.configure("2") do |config|
  config.vm.box = "debian/bookworm64"
  config.vm.hostname = "pricing-mapper"

  config.vm.provider "virtualbox" do |vb|
    vb.memory = 4096
    vb.cpus = 2
  end

  config.vm.synced_folder ".", "/vagrant", type: "virtualbox"

  config.vm.provision "shell", inline: <<-SHELL
    set -e
    apt-get update
    apt-get install -y python3 python3-venv python3-pip git build-essential
  SHELL
end
```

### 3. Start VM and enter shell

```bash
vagrant up
vagrant ssh
```

### 4. Set up project inside VM

```bash
cd /vagrant
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 5. Run mapper

```bash
python -m pricing_mapper --config config.example.json
```

### 6. Optional: install dev tools and run checks

```bash
pip install -e .[dev]
ruff check pricing_mapper tests comp_car_active_mapper_advanced.py
mypy pricing_mapper
python -m pytest -q
```

### 7. Stop VM

```bash
exit
vagrant halt
```

## Disclaimers and Usage Boundaries

- This project is for research, testing, and internal analysis workflows.
- Default quote logic is synthetic and not an insurer-approved pricing engine.
- If you connect a real quote provider, you are responsible for authorization, legal compliance, and access controls.
- Do not use outputs as a sole basis for underwriting, premium setting, consumer disclosures, or regulatory filings.
- Ensure compliance with applicable insurance, privacy, consumer protection, and anti-discrimination laws in your jurisdiction.
- Respect provider terms of service, rate limits, and data handling requirements.
- Validate model behavior independently before operational use.
- No warranty is provided for fitness, correctness, or regulatory suitability.

Date handling:

- Vehicle-year limits and synthetic quote vehicle-age logic use the current UTC year at runtime (not a fixed year constant).

## License

See `LICENSE`.
