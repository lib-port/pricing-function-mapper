# Pricing Function Mapper

Production-grade offline Python library and CLI for learning a surrogate of a
trusted comprehensive-car-insurance quote function.

Version 1 separates adaptive mapping from independent validation, conformal
calibration, and final audit samples. It exports strict, hash-verified artifact
directories and never loads pickle.

## What v1 guarantees

- `CarQuoteInput` validates exactly 15 car-insurance fields. Public inference
  rejects missing, unknown, non-finite, infeasible, and out-of-domain values;
  it never clips them.
- Mapping and evaluation have separate quote budgets. Evaluation rows are
  generated before adaptive sampling, with default 40% validation, 30%
  calibration, and 30% audit allocation.
- When enabled, early stopping uses validation MAE and bootstrap confidence
  bounds; it is disabled by default. Neither calibration nor audit targets
  enter acquisition, tuning, early stopping, or model selection.
- Bounded randomized selection compares native-categorical monotonic and
  unconstrained histogram gradient boosting with an ExtraTrees pipeline under
  a warm single-row p95 latency ceiling.
- Predictions contain a split-conformal interval, model version, and warnings.
  The final audit reports coverage and interval width.
- Every completed quote is journaled transactionally in SQLite. Resume
  restores the mapping RNG and pending batch, reproducing an uninterrupted run.
- Artifacts use `skops`, JSON/TOML metadata, SHA-256 hashes, an exclusive run
  lock, and atomic staging-directory rename. v0 pickle/state files are rejected.

This project is an offline mapper, not an HTTP service. Hosted serving and
vendor-specific integrations are intentionally outside v1.

Model versions are deterministic hashes of model-affecting configuration,
fit observations, numerical/model dependency versions, and conformal state;
artifact paths and wall-clock timings are excluded.

## Install

Python 3.11–3.13 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For development:

```bash
python -m pip install -e '.[dev]'
pre-commit install
```

`pylock.toml` records the reproducible lock for the platform on which it was
generated. The project metadata retains compatible ranges for supported Python
versions and platforms.

## Quick start

Validate the strict TOML configuration and provider:

```bash
pricing-mapper config validate --config config.example.toml
```

Run mapping with the bundled deterministic reference provider:

```bash
pricing-mapper map run \
  --config config.example.toml \
  --run-id example
```

The v1 vehicle-year ceiling and illustrative reference provider use a fixed
2026 rating year so configuration fingerprints, benchmarks, and crash recovery
do not change at a calendar-year boundary.

The command prints the artifact path. Price one row:

```bash
pricing-mapper predict row \
  --artifact outputs/example \
  --input examples/quote-row.json
```

The result has this contract:

```json
{
  "premium": 789.12,
  "lower": 650.34,
  "upper": 927.90,
  "model_version": "v1-...",
  "warnings": []
}
```

Verify the complete artifact before moving it:

```bash
pricing-mapper artifact inspect --artifact outputs/example
pricing-mapper model evaluate --artifact outputs/example --require-gates
```

`model evaluate` recomputes independent audit metrics and interval coverage
from the exported audit rows, and remeasures warm single-row latency on the
current machine.

## CLI

Version 1 uses subcommands only:

```text
pricing-mapper config validate
pricing-mapper map run
pricing-mapper map resume
pricing-mapper model evaluate
pricing-mapper predict row
pricing-mapper predict batch
pricing-mapper artifact inspect
pricing-mapper benchmark
```

Resume after interruption with the same configuration and run ID:

```bash
pricing-mapper map resume \
  --config config.example.toml \
  --run-id example
```

Import validated CSV observations instead of a v0 checkpoint:

```bash
pricing-mapper map run \
  --config config.example.toml \
  --run-id seeded \
  --seed-data observations.csv
```

The seed CSV must have the exact 15 input columns in canonical order followed
by `premium`. It is validated against the configured domain. Pickle artifacts
and JSON state cannot be converted or loaded.

Batch prediction accepts the exact 15 input columns in canonical order:

```bash
pricing-mapper predict batch \
  --artifact outputs/example \
  --input rows.csv \
  --output predictions.csv
```

Documented process exit codes are:

- `0`: success
- `2`: invalid input or configuration
- `3`: provider, persistence, or run failure
- `4`: unsafe, corrupt, or incompatible artifact
- `5`: requested evaluation/benchmark gate failed

## Library API

```python
from pricing_mapper import CarQuoteInput, MapperConfig, MappingRun, PricingEngine

config = MapperConfig()
result = MappingRun(config, run_id="library-run").run()

engine = PricingEngine.load(result.artifact_dir)
quote = CarQuoteInput.model_validate(
    {
        "driver_age": 40.0,
        "years_licensed": 20,
        "vehicle_year": 2022,
        "vehicle_value": 35000.0,
        "annual_km": 10000,
        "claims_5y": 0,
        "convictions_5y": 0,
        "postcode_risk": 0.2,
        "theft_risk": 0.2,
        "excess": 700,
        "usage": "private",
        "parking": "garage",
        "hire_car": "none",
        "windscreen": "no",
        "rating": "market",
    },
    strict=True,
)
prediction = engine.predict(quote)
```

Stable public interfaces are `CarQuoteInput`, `QuoteProvider`, `MapperConfig`,
`Prediction`, `MappingRun`, and `PricingEngine`.

## Local quote providers

Configure a trusted callable:

```toml
[provider]
callable = "my_insurer.quotes:quote"
max_retries = 3
concurrency = 1
```

The callable receives `CarQuoteInput` and returns a finite, non-negative
number. Raise `ProviderUnavailable` for retryable failures and
`ProviderRejected` for permanent failures. Unexpected exceptions are treated
as permanent. Parallel execution is permitted only when the callable declares:

```python
quote.thread_safe = True
quote.max_concurrency = 4
quote.provider_id = "my-insurer-rating-v7"
```

Treat `provider_id` as a behavior version: change it whenever rating logic or
upstream data changes. Resume rejects an identity mismatch.

Attempt outcome, latency, error type, and provider identity are stored without
logging quote payloads. Sequential execution is the default.

## Artifact layout

An atomic v1 directory contains:

```text
manifest.json             SHA-256 hashes and artifact identity
schema.json               generated CarQuoteInput JSON Schema
config.toml               validated configuration snapshot
domain.json               resolved domain snapshot
dataset.csv               mapping and labeled evaluation observations
dataset.schema.json       explicit CSV types and split metadata
model.skops               fitted sklearn estimator only
model.json                estimator family, encoding, and selection report
conformal.json            calibration method and radius
evaluation.json           held-out metrics, confidence bounds, and gates
model-card.md             intended use, performance, and limitations
provenance.json           provider/run/build provenance without payload logs
dependencies.json         exact artifact dependency versions
```

Loading checks every hash, rejects unlisted files and symlinks, enforces the
saved scikit-learn version, checks a fixed `skops` type allow-list, and
reconstructs validation and encoding from versioned metadata.

## Development gates

```bash
scripts/quality.sh
scripts/smoke.sh
```

The CI matrix covers Python 3.11–3.13, Ruff, Black, strict mypy, branch
coverage, wheel/sdist builds, installed-wheel CLI smoke tests, dependency
audit, schema freshness, and artifact validation. A scheduled five-seed
benchmark compares active acquisition with equal-budget LHS and random
sampling and enforces accuracy, conformal coverage, per-run latency, and
latency-regression gates.

See:

- [Architecture](docs/architecture.md)
- [Operational playbooks](docs/operations.md)
- [Configuration reference](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Security policy](SECURITY.md)
- [Regulatory limitations](docs/regulatory-limitations.md)
- [Release procedure](docs/release.md)

## Regulatory notice

This software does not establish legal or regulatory compliance. The bundled
provider is synthetic. Real insurance use requires jurisdiction-specific
actuarial, anti-discrimination, privacy, explainability, governance,
rate-filing, monitoring, and human-oversight review.

## License

MIT. See [LICENSE](LICENSE).
