# Configuration reference

The primary format is TOML. Unknown keys are rejected at every level.
`schema/mapper-config.schema.json` is generated from the same Pydantic models
used at runtime.

Start with `config.example.toml`, then run:

```bash
pricing-mapper config validate --config your-config.toml
```

## `sampling`

- `mapping_budget`: maximum mapping observations, excluding evaluation.
- `initial_size`: initial mapping LHS size; no larger than the mapping budget.
- `batch_size`: observations in each later acquisition batch.
- `candidate_pool_size`: unique candidate rows scored per active batch.
- `seed`: unsigned 32-bit deterministic seed.
- `strategy`: `active`, `bayesian`, `lhs`, or `random`. `active` remains the
  default. `bayesian` fits a deterministic local Gaussian process and uses its
  posterior uncertainty plus validation-residual targeting.

`sampling.acquisition.weights` accepts only `uncertainty`, `residual`,
`breakpoint`, and `diversity`. Non-negative weights are normalized.
`greedy_diversity_weight` blends the aggregate score with max-min distance
while constructing the focused portion of a batch. `focused_fraction` sets
that portion; the remainder is a fresh LHS background so active learning does
not sacrifice broad population coverage.

## `optimizer.ollama` (optional and experimental)

This table is accepted only with `sampling.strategy = "bayesian"`. Omitting it
uses the fixed balanced Bayesian policy and makes no Ollama requests.

- `endpoint`: an HTTP origin using a literal IPv4 loopback address in
  `127.0.0.0/8` or the literal IPv6 loopback address `::1`; the deployment
  example uses `http://127.0.0.1:11434`. Hostnames, HTTPS, remote addresses,
  credentials, paths, queries, fragments, environment proxies, and redirects
  are rejected or disabled so aggregate diagnostics cannot leave the host.
- `model`: a fully tagged local model name; the deployment example uses
  `granite4.1:3b`.
- `required_digest`: mandatory full lowercase `sha256:` model digest.
- `prompt_version`: currently only `policy-advisor-v1`.
- `timeout_seconds`: positive and no greater than 60 seconds.
- `retry_count`: retries after the first attempt; fixed to at most two.
- `resource_mode`: currently only `cpu-only-2cpu-8gb`.

The supplied Compose deployment pins the Ollama image digest. At startup, the
mapper records `/api/version` and checks `/api/tags`, including model name, full
digest, size, and `Q4_K_M` quantization. Resume requires the same recorded
runtime and model metadata, so a mid-run change fails closed. The advisor
selects from five code-owned policies:

| Policy | Uncertainty | Residual | LHS exploration |
|---|---:|---:|---:|
| `balanced` | 70% | 30% | 20% |
| `uncertainty` | 85% | 15% | 20% |
| `residual` | 55% | 45% | 20% |
| `explore` | 60% | 20% | 40% |
| `exploit` | 80% | 20% | 10% |

It may also nominate at most three diagnostic bin IDs for a fixed 1.10 or 1.25
boost. The model cannot provide fields, breakpoints, arbitrary weights,
expressions, or quote inputs. Only aggregate, normalized diagnostics leave the
mapper; raw premiums, individual rows, calibration data, and audit data do not.
See `config.ollama.example.toml` and `deploy/ollama/README.md`.

## `model`

- `search_iterations`: bounded randomized trials per family.
- `hgb_max_iter`: boosting iterations.
- `extra_trees_estimators`: trees in each ExtraTrees candidate.
- `committee_size` / `committee_estimators`: deterministic RF acquisition
  committee.
- `n_jobs`: explicit worker count for RF/ExtraTrees operations. HGB fitting
  and warm single-row inference use one OpenMP thread for reproducibility and
  stable latency measurement.
- `max_p95_latency_ms`: promotion ceiling for a warm single-row prediction.
- `latency_repetitions`: measurements after warm-up (100 by default so the
  empirical p95 is not dominated by one scheduler outlier).

`model.monotonic_constraints` maps numeric fields to `-1`, `0`, or `1`.
Constrained and unconstrained HGB candidates are scored on validation, and the
accuracy delta is exported.

`model.early_stopping.patience_batches = 0` disables stopping. When enabled,
only validation MAE bootstrap bounds drive the decision.

## `evaluation`

`evaluation_budget` is separate from `mapping_budget`. The validation,
calibration, and audit fractions must be positive, sum to one, and allocate at
least one row each. Defaults are 40/30/30.

`conformal_coverage` defaults to 0.90. Independent audit coverage must lie
between `minimum_audit_coverage` and `maximum_audit_coverage` (85–95% by
default) for promotion eligibility.

`bootstrap_iterations = 0` is useful only for very small deterministic tests;
production runs should retain several hundred or more resamples.

## `provider`

- `callable`: optional trusted `module:function`; omission uses the reference
  synthetic provider.
- `max_retries`: retries after the initial call, only for
  `ProviderUnavailable`.
- `initial_backoff_seconds` / `maximum_backoff_seconds`: bounded exponential
  delay.
- `concurrency`: defaults to one; values above one require explicit provider
  thread-safety and concurrency declarations.

## `artifact`

- `output_dir`: parent for final artifacts, state, and lock files.
- `state_dir_name`: relative private journal directory (default `.runs`).
- `model_card_title`: title rendered into the exported model card.

## `domain`

Omitting `domain` resolves the supported v1 limits. To narrow a limit, provide
all bounds in a generated full configuration snapshot. Bounds may not expand
beyond the stable `CarQuoteInput` schema, and integer fields require integral
limits. The v1 vehicle-year ceiling is fixed at 2026 and stored in every
artifact; changing it requires an explicit schema-versioned release.
