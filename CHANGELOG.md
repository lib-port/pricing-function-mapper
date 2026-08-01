# Changelog

All notable changes follow Keep a Changelog conventions.

## Unreleased

### Added

- Deterministic Gaussian-process Bayesian acquisition with an optional,
  digest-pinned Ollama policy advisor, aggregate-only diagnostics, strict
  finite policy catalogue, atomic decision replay, and separate advisor
  latency/resident-memory provenance.
- Pinned two-CPU/8-GiB offline Ollama Compose deployment, verified Granite 4.1
  provisioning, and a five-seed active/Bayesian/hybrid ablation gate.

### Security

- Advisor output is treated as untrusted. Duplicate keys, unknown fields,
  non-finite values, invalid policies/bins/boosts, oversized replies, runtime
  drift, and model-digest mismatches fail closed before quote consumption.
- Advisor endpoints are restricted to literal HTTP loopback addresses;
  environment proxies and redirects are disabled to prevent diagnostic egress.
- Development dependency floors and CI audits cover patched tooling as well as
  declared runtime dependencies.

### Changed

- Generated immutable model cards now include complete selection, conformal,
  audit, risk-slice, promotion, provider, and dependency evidence. Supplemental
  organizational governance records remain outside the artifact.

## 1.0.0 - 2026-07-29

### Added

- Strict 15-field `CarQuoteInput`, nested TOML `MapperConfig`, generated JSON
  Schemas, `Prediction`, `MappingRun`, and `PricingEngine`.
- Independent validation, conformal calibration, and final audit budgets.
- Validation-confidence early stopping, production metrics, bootstrap bounds,
  risk slices, model latency selection, and monotonic tradeoff reporting.
- Pluggable uncertainty, residual, breakpoint, and diversity acquisition.
- Provider retry/permanent failure types, payload-free telemetry, declared
  bounded parallelism, and per-quote durability.
- Transactional SQLite resume, RNG/batch recovery, run locks, atomic artifact
  publication, `skops` persistence, hashes, model cards, and provenance.
- Subcommand CLI, multi-version CI, installed-wheel smoke test, dependency
  audit, and scheduled five-seed benchmark.
- Mermaid system, data-flow, lifecycle, recovery, and trust-boundary diagrams,
  plus operational playbooks for runs, incidents, promotion, upgrades, and
  rollback.

### Removed

- Legacy flag-mode CLI, JSON configuration, pickle engines, JSON checkpoints,
  FastAPI extra, and Vagrant workflow.

### Security

- v0 pickle/state artifacts are rejected. Existing CSV observations may be
  reused only through strict `--seed-data` validation.
