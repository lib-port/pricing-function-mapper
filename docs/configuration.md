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
- `strategy`: `active`, `lhs`, or `random`.

`sampling.acquisition.weights` accepts only `uncertainty`, `residual`,
`breakpoint`, and `diversity`. Non-negative weights are normalized.
`greedy_diversity_weight` blends the aggregate score with max-min distance
while constructing the focused portion of a batch. `focused_fraction` sets
that portion; the remainder is a fresh LHS background so active learning does
not sacrifice broad population coverage.

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
