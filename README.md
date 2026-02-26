# Pricing Function Mapper

Active-learning mapper for a pricing function (default: mock comprehensive car insurance quote model).

This project incrementally samples the input space, trains surrogate models, and selects high-value next queries to approximate the behavior of a target quote/pricing function.

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
5. Repeat until budget is reached.
6. Write artifacts (dataset, metadata, state checkpoint).

## Repository Structure

- `pricing_mapper/domain.py`: variable definitions, domain construction, canonicalization.
- `pricing_mapper/encoding.py`: cached feature encoding pipeline.
- `pricing_mapper/models.py`: surrogate model wrappers.
- `pricing_mapper/active_mapper.py`: active-learning loop, persistence, profiling.
- `pricing_mapper/quote.py`: default quote function + pluggable provider loader.
- `pricing_mapper/cli.py`: CLI entrypoint.
- `pricing_mapper/benchmark.py`: benchmark presets and result writer.
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

### Resume a run

```bash
python -m pricing_mapper --config config.example.json --resume
```

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

## Configuration

Use `config.example.json` as a template. Key options include:

- Sampling controls: `budget`, `init_n`, `batch_size`, `pool_size`
- Model controls: `rf_n_models`, `rf_n_estimators`, `refit_every_batches`
- Search behavior: `distance_backend`, `acquisition_mix`, `breakpoint_vars`
- Performance controls: `cv_subsample_max`
- Artifacts: `output_dir`, `run_id`, `output_csv`, `output_metadata_json`, `state_path`
- Resume/checkpoint: `resume`, `checkpoint_every_batches`
- Provider: `quote_provider`

## Output Artifacts

### Dataset CSV

Contains sampled rows and true `premium` values queried from the provider.

### Metadata JSON

Includes:

- run stats and elapsed time
- per-phase profiling times
- MAE diagnostics
- resolved config
- feature list
- artifact paths

### State JSON

Checkpoint state used for resume support.

- versioned schema
- atomic writes
- migration support for older schema versions

## Quality and Testing

Lint, type-check, tests:

```bash
ruff check pricing_mapper tests comp_car_active_mapper_advanced.py
black --check pricing_mapper tests comp_car_active_mapper_advanced.py
mypy pricing_mapper
python -m unittest discover -s tests -v
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
python -m unittest discover -s tests -v
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

## License

See `LICENSE`.
