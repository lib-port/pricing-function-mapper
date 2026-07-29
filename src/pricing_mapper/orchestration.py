"""Durable v1 mapping orchestration with explicit holdout boundaries."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from pricing_mapper.acquisition import AcquisitionStrategy, build_context
from pricing_mapper.artifact import export_artifact, validate_artifact
from pricing_mapper.config import MapperConfig, config_toml
from pricing_mapper.domain import (
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
    FIELD_ORDER,
    INTEGER_FIELDS,
    CarQuoteInput,
    DomainSpec,
    row_key,
)
from pricing_mapper.encoding import FeatureEncoder
from pricing_mapper.evaluation import (
    EarlyStopTracker,
    bootstrap_intervals,
    conformal_bounds,
    conformal_radius,
    interval_report,
    regression_report,
)
from pricing_mapper.exceptions import PersistenceError
from pricing_mapper.models import (
    AcquisitionCommittee,
    ModelKind,
    finite_nonnegative_targets,
    fit_monitor_model,
    measure_single_row_latency,
    predict_estimator,
    refit_selected,
    select_model,
)
from pricing_mapper.persistence import (
    RunLock,
    RunStore,
    SampleRecord,
    SplitName,
    validate_run_id,
)
from pricing_mapper.provider import ProviderExecutor, resolve_provider
from pricing_mapper.types import QuoteProvider


@dataclass(frozen=True)
class MappingResult:
    """Completed mapping run locations and unbiased evaluation summary."""

    run_id: str
    artifact_dir: Path
    state_database: Path
    mapping_samples: int
    evaluation_samples: int
    early_stopped: bool
    evaluation_report: dict[str, Any]


def default_run_id() -> str:
    """Generate a collision-resistant, path-safe run identifier."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"run-{timestamp}-{secrets.token_hex(4)}"


def _generator(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def _restore_generator(state: Mapping[str, Any]) -> np.random.Generator:
    rng = np.random.default_rng()
    try:
        rng.bit_generator.state = dict(state)
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"stored mapping RNG state is invalid: {exc}") from exc
    return rng


def _unique_samples(
    domain: DomainSpec,
    *,
    count: int,
    rng: np.random.Generator,
    used_hashes: set[str],
    method: str,
) -> list[CarQuoteInput]:
    if count < 0:
        raise ValueError("sample count cannot be negative")
    if count == 0:
        return []
    selected: list[CarQuoteInput] = []
    local_hashes = set(used_hashes)
    attempts = 0
    while len(selected) < count and attempts < 30:
        remaining = count - len(selected)
        request = max(remaining, min(remaining * 2, 10_000))
        raw_rows = (
            domain.sample_lhs(request, rng)
            if method == "lhs"
            else domain.sample_random(request, rng)
        )
        for raw in raw_rows:
            quote = CarQuoteInput.from_mapping(raw, domain=domain)
            hashed = row_key(quote)
            if hashed in local_hashes:
                continue
            local_hashes.add(hashed)
            selected.append(quote)
            if len(selected) == count:
                break
        attempts += 1
    if len(selected) != count:
        raise RuntimeError(
            f"unable to generate {count} unique {method} rows after {attempts} attempts"
        )
    return selected


def _records_data(
    records: Sequence[SampleRecord],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if any(record.premium is None for record in records):
        raise PersistenceError("model data includes incomplete quote records")
    rows = [record.row for record in records]
    premium_values: list[float] = []
    for record in records:
        if record.premium is None:
            raise AssertionError("premium was checked above")
        premium_values.append(record.premium)
    targets = finite_nonnegative_targets(premium_values)
    return rows, targets


def _parse_seed_number(raw: str, field: str, *, integer: bool) -> int | float:
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"seed-data field {field!r} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"seed-data field {field!r} must be finite")
    if integer:
        if not value.is_integer():
            raise ValueError(f"seed-data field {field!r} must be an integer")
        return int(value)
    return value


def load_seed_data(
    path: str | Path,
    domain: DomainSpec,
) -> list[tuple[CarQuoteInput, float]]:
    """Validate a legacy observation CSV without loading any legacy state/model."""

    target = Path(path)
    try:
        handle = target.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(f"cannot read --seed-data CSV {target}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        expected = [*FIELD_ORDER, "premium"]
        if reader.fieldnames != expected:
            raise ValueError(
                "--seed-data CSV columns must exactly match the v1 schema in order: " f"{expected}"
            )
        result: list[tuple[CarQuoteInput, float]] = []
        seen: set[str] = set()
        for line_number, raw in enumerate(reader, start=2):
            values: dict[str, Any] = {}
            try:
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError("row has missing or extra columns")
                for name in CONTINUOUS_FIELDS:
                    values[name] = _parse_seed_number(raw[name], name, integer=False)
                for name in INTEGER_FIELDS:
                    values[name] = _parse_seed_number(raw[name], name, integer=True)
                for name in CATEGORICAL_FIELDS:
                    values[name] = raw[name]
                quote = CarQuoteInput.from_mapping(values, domain=domain)
                premium = float(_parse_seed_number(raw["premium"], "premium", integer=False))
                if premium < 0:
                    raise ValueError("seed-data premium must be non-negative")
            except ValueError as exc:
                raise ValueError(f"invalid --seed-data row {line_number}: {exc}") from exc
            hashed = row_key(quote)
            if hashed in seen:
                raise ValueError(f"--seed-data contains a duplicate at row {line_number}")
            seen.add(hashed)
            result.append((quote, premium))
    if not result:
        raise ValueError("--seed-data must contain at least one observation")
    return result


class MappingRun:
    """Supported training entry point for new and resumed v1 mapping runs."""

    def __init__(
        self,
        config: MapperConfig,
        *,
        provider: QuoteProvider | None = None,
        run_id: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.domain = config.resolved_domain
        self.run_id = validate_run_id(run_id or default_run_id())
        self.logger = logger or logging.getLogger(__name__)
        self.provider_descriptor = resolve_provider(config.provider.callable, provider)
        self.provider = ProviderExecutor(
            self.provider_descriptor,
            config.provider,
            logger=self.logger,
        )
        self.output_dir = Path(config.artifact.output_dir)
        self.artifact_dir = self.output_dir / self.run_id
        self.run_dir = self.output_dir / config.artifact.state_dir_name / self.run_id
        self.state_database = self.run_dir / "run.sqlite3"
        self.lock_path = self.output_dir / ".locks" / f"{self.run_id}.lock"

    def run(
        self,
        *,
        resume: bool = False,
        seed_data: str | Path | None = None,
    ) -> MappingResult:
        """Execute through atomic artifact publication or resume after a crash."""

        seed_records = None if seed_data is None else load_seed_data(seed_data, self.domain)
        if seed_records is not None and len(seed_records) > self.config.sampling.mapping_budget:
            raise ValueError(
                "--seed-data rows cannot exceed sampling.mapping_budget "
                f"({len(seed_records)} > {self.config.sampling.mapping_budget})"
            )

        with RunLock(self.lock_path):
            if self.artifact_dir.exists():
                if not resume:
                    raise FileExistsError(
                        f"artifact run {self.run_id!r} already exists; choose another run_id"
                    )
                inspection = validate_artifact(self.artifact_dir)
                if inspection["run_id"] != self.run_id:
                    raise PersistenceError(
                        "completed artifact run_id differs from the requested resume run_id"
                    )
                if inspection["config_fingerprint"] != self.config.fingerprint:
                    raise PersistenceError(
                        "completed artifact configuration differs from the resume configuration"
                    )
                if inspection["provider_identity"] != self.provider_descriptor.identity:
                    raise PersistenceError(
                        "completed artifact provider identity differs from the resume provider"
                    )
                if seed_records is not None:
                    with RunStore.open(self.state_database) as completed_store:
                        expected_digest = completed_store.metadata("seed_data_digest")
                    if expected_digest != self._seed_digest(seed_records):
                        raise PersistenceError(
                            "resume --seed-data does not match the completed run import"
                        )
                evaluation = json.loads(
                    (self.artifact_dir / "evaluation.json").read_text(encoding="utf-8")
                )
                split_counts = inspection["dataset_split_counts"]
                return MappingResult(
                    run_id=self.run_id,
                    artifact_dir=self.artifact_dir,
                    state_database=self.state_database,
                    mapping_samples=int(split_counts["mapping"]),
                    evaluation_samples=sum(
                        int(split_counts[name]) for name in ("validation", "calibration", "audit")
                    ),
                    early_stopped=bool(evaluation.get("early_stopped", False)),
                    evaluation_report=evaluation,
                )

            store = self._open_store(resume=resume)
            try:
                self._ensure_initialized(store)
                store.validate_integrity(
                    config_fingerprint=self.config.fingerprint,
                    domain=self.domain,
                    provider_identity=self.provider_descriptor.identity,
                )
                if seed_records is not None:
                    if resume:
                        expected_digest = store.metadata("seed_data_digest")
                        actual_digest = self._seed_digest(seed_records)
                        if expected_digest != actual_digest:
                            raise PersistenceError(
                                "resume --seed-data does not match the original import"
                            )
                    elif store.samples("mapping"):
                        raise PersistenceError("seed data cannot be imported after mapping starts")
                    else:
                        store.register_seed_samples(seed_records)
                        store.set_metadata("seed_data_digest", self._seed_digest(seed_records))
                elif resume and store.metadata("seed_data_digest") is not None:
                    # The imported rows are already durable; the original CSV is
                    # deliberately not required again.
                    pass

                self._quote_pending(
                    store,
                    splits=("validation", "calibration", "audit"),
                )
                self._continue_mapping(store)
                result = self._finalize(store)
                store.set_metadata("completed", True)
                store.set_metadata("artifact_dir", str(result.artifact_dir))
                return result
            finally:
                store.close()

    def resume(self) -> MappingResult:
        """Resume the exact run journal associated with this ``run_id``."""

        return self.run(resume=True)

    def _open_store(self, *, resume: bool) -> RunStore:
        if resume:
            return RunStore.open(self.state_database)
        if self.state_database.exists():
            raise FileExistsError(f"run state {self.state_database} already exists; use map resume")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        store = RunStore.create(
            self.state_database,
            config_fingerprint=self.config.fingerprint,
            config_toml=config_toml(self.config),
            domain=self.domain,
            provider_identity=self.provider_descriptor.identity,
        )
        store.set_metadata("run_id", self.run_id)
        store.set_metadata("run_started_at_utc", datetime.now(UTC).isoformat())
        return store

    def _ensure_initialized(self, store: RunStore) -> None:
        """Recover safely if a process stopped during new-run initialization."""

        if not store.metadata("evaluation_generated", False):
            if store.samples():
                raise PersistenceError("run initialization is incomplete but samples already exist")
            evaluation_rng = _generator(self.config.sampling.seed, 1)
            count = self.config.evaluation.evaluation_budget
            rows = _unique_samples(
                self.domain,
                count=count,
                rng=evaluation_rng,
                used_hashes=set(),
                method="lhs",
            )
            counts = self.config.evaluation.split_counts()
            start = 0
            split_rows: dict[SplitName, list[CarQuoteInput]] = {}
            for split in ("validation", "calibration", "audit"):
                stop = start + counts[split]
                split_rows[split] = rows[start:stop]
                start = stop
            store.register_evaluation_splits(split_rows)

        try:
            store.rng_state("mapping")
        except PersistenceError as exc:
            if store.samples("mapping") or store.batch_history():
                raise PersistenceError(
                    "mapping RNG state is missing or corrupt after mapping started"
                ) from exc
            mapping_rng = _generator(self.config.sampling.seed, 2)
            store.save_rng_state("mapping", mapping_rng.bit_generator.state)

    @staticmethod
    def _seed_digest(records: Sequence[tuple[CarQuoteInput, float]]) -> str:
        payload = [{"row": quote.as_dict(), "premium": premium} for quote, premium in records]
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _quote_pending(
        self,
        store: RunStore,
        *,
        splits: Sequence[SplitName] | None = None,
    ) -> None:
        pending = store.pending_samples(splits)
        if not pending:
            return
        provider_identity = self.provider_descriptor.identity
        to_call: list[tuple[int, CarQuoteInput]] = []
        for record in pending:
            cached = store.cached_quote(provider_identity, record.row_hash)
            if cached is not None:
                store.complete_quote(record.id, record.row_hash, cached, provider_identity)
            else:
                to_call.append((record.id, record.validated_input(self.domain)))
        if to_call:
            self.provider.quote_many(
                to_call,
                on_attempt=store.record_attempt,
                on_complete=lambda sample_id, hashed, premium: store.complete_quote(
                    sample_id,
                    hashed,
                    premium,
                    provider_identity,
                ),
            )

    def _continue_mapping(self, store: RunStore) -> None:
        mapping_rng = _restore_generator(store.rng_state("mapping"))
        self._quote_pending(store, splits=("mapping",))

        current_count = len(store.samples("mapping", completed_only=True))
        initial_target = min(
            self.config.sampling.initial_size,
            self.config.sampling.mapping_budget,
        )
        if current_count < initial_target:
            needed = initial_target - current_count
            initial_rows = _unique_samples(
                self.domain,
                count=needed,
                rng=mapping_rng,
                used_hashes=store.all_row_hashes(),
                method="lhs",
            )
            store.register_samples(
                "mapping",
                initial_rows,
                batch_id=store.next_batch_id(),
                source="initial_lhs",
                mapping_rng_state=mapping_rng.bit_generator.state,
            )
            self._quote_pending(store, splits=("mapping",))

        tracker = self._evaluate_quoted_batches(store)
        while (
            len(store.samples("mapping", completed_only=True)) < self.config.sampling.mapping_budget
            and not tracker.stopped
        ):
            mapping_records = store.samples("mapping", completed_only=True)
            remaining = self.config.sampling.mapping_budget - len(mapping_records)
            count = min(self.config.sampling.batch_size, remaining)
            next_rows = self._propose(mapping_records, count, mapping_rng, store)
            store.register_samples(
                "mapping",
                next_rows,
                batch_id=store.next_batch_id(),
                source=f"{self.config.sampling.strategy}_acquisition",
                mapping_rng_state=mapping_rng.bit_generator.state,
            )
            self._quote_pending(store, splits=("mapping",))
            tracker = self._evaluate_quoted_batches(store)

        if tracker.stopped:
            store.set_metadata("early_stopped", True)
            store.set_metadata(
                "stop_reason",
                "validation MAE confidence bound did not improve within patience",
            )

    def _propose(
        self,
        mapping_records: Sequence[SampleRecord],
        count: int,
        rng: np.random.Generator,
        store: RunStore,
    ) -> list[CarQuoteInput]:
        used = store.all_row_hashes()
        strategy = self.config.sampling.strategy
        if strategy in {"lhs", "random"}:
            return _unique_samples(
                self.domain,
                count=count,
                rng=rng,
                used_hashes=used,
                method=strategy,
            )

        mapping_rows, mapping_targets = _records_data(mapping_records)
        committee = AcquisitionCommittee(
            self.config.model,
            self.domain,
            self.config.sampling.seed,
        ).fit(mapping_rows, mapping_targets)
        pool = _unique_samples(
            self.domain,
            count=self.config.sampling.candidate_pool_size,
            rng=rng,
            used_hashes=used,
            method="lhs",
        )
        pool_rows = [quote.as_dict() for quote in pool]
        _, standard_deviation = committee.predict(pool_rows)
        validation_records = store.samples("validation", completed_only=True)
        validation_rows, validation_targets = _records_data(validation_records)
        validation_predictions, _ = committee.predict(validation_rows)
        residuals = validation_targets - validation_predictions
        context = build_context(
            candidate_rows=pool_rows,
            training_rows=mapping_rows,
            predictive_std=standard_deviation,
            training_residuals=residuals,
            domain=self.domain,
            residual_anchor_rows=validation_rows,
        )
        focused_count = max(
            1,
            min(
                count,
                round(count * self.config.sampling.acquisition.focused_fraction),
            ),
        )
        indices = AcquisitionStrategy(self.config.sampling.acquisition).select(
            context,
            focused_count,
        )
        if len(indices) != focused_count:
            raise RuntimeError(
                f"active acquisition selected {len(indices)} rows, expected {focused_count}"
            )
        focused = [pool[index] for index in indices]
        background_count = count - focused_count
        if background_count == 0:
            return focused
        background = _unique_samples(
            self.domain,
            count=background_count,
            rng=rng,
            used_hashes=used | {row_key(quote) for quote in focused},
            method="lhs",
        )
        return [*focused, *background]

    def _tracker_from_history(self, store: RunStore) -> EarlyStopTracker:
        tracker = EarlyStopTracker()
        early = self.config.model.early_stopping
        for batch in store.batch_history():
            metrics = batch["metrics"]
            if not isinstance(metrics, dict) or "validation_mae" not in metrics:
                continue
            confidence = metrics.get("mae_confidence_interval")
            if not isinstance(confidence, dict):
                raise PersistenceError("batch validation confidence interval is corrupt")
            tracker = tracker.update(
                mae=float(metrics["validation_mae"]),
                lower_bound=float(confidence["lower"]),
                upper_bound=float(confidence["upper"]),
                config=early,
            )
        return tracker

    def _evaluate_quoted_batches(self, store: RunStore) -> EarlyStopTracker:
        tracker = self._tracker_from_history(store)
        validation_records = store.samples("validation", completed_only=True)
        validation_rows, validation_targets = _records_data(validation_records)
        initial_target = self.config.sampling.initial_size

        for batch_id in store.unevaluated_batches():
            all_mapping = [
                record
                for record in store.samples("mapping", completed_only=True)
                if record.batch_id is not None and record.batch_id <= batch_id
            ]
            if len(all_mapping) < initial_target:
                store.mark_batch_evaluated(
                    batch_id,
                    {
                        "deferred_until_initial_size": initial_target,
                        "mapping_samples": len(all_mapping),
                    },
                )
                continue
            mapping_rows, mapping_targets = _records_data(all_mapping)
            _, predictions = fit_monitor_model(
                mapping_rows=mapping_rows,
                mapping_targets=mapping_targets,
                validation_rows=validation_rows,
                config=self.config.model,
                domain=self.domain,
                seed=self.config.sampling.seed + batch_id,
            )
            absolute_errors = np.abs(predictions - validation_targets)
            mae = float(np.mean(absolute_errors))
            intervals = bootstrap_intervals(
                validation_targets,
                predictions,
                iterations=self.config.evaluation.bootstrap_iterations,
                confidence=self.config.model.early_stopping.confidence_level,
                seed=self.config.sampling.seed + 100_003 * (batch_id + 1),
            )
            confidence = intervals["mae"]
            next_tracker = tracker.update(
                mae=mae,
                lower_bound=confidence["lower"],
                upper_bound=confidence["upper"],
                config=self.config.model.early_stopping,
            )
            store.mark_batch_evaluated(
                batch_id,
                {
                    "mapping_samples": len(mapping_rows),
                    "validation_samples": len(validation_rows),
                    "validation_mae": mae,
                    "mae_confidence_interval": confidence,
                    "early_stop_tracker": next_tracker.to_dict(),
                    "leakage_guard": "mapping fit; validation scoring only",
                },
            )
            tracker = next_tracker
            store.set_metadata("early_stop_tracker", tracker.to_dict())
            if tracker.stopped:
                break
        return tracker

    def _finalize(self, store: RunStore) -> MappingResult:
        mapping_records = store.samples("mapping", completed_only=True)
        validation_records = store.samples("validation", completed_only=True)
        calibration_records = store.samples("calibration", completed_only=True)
        audit_records = store.samples("audit", completed_only=True)
        mapping_rows, mapping_targets = _records_data(mapping_records)
        validation_rows, validation_targets = _records_data(validation_records)
        calibration_rows, calibration_targets = _records_data(calibration_records)
        audit_rows, audit_targets = _records_data(audit_records)

        selection = select_model(
            mapping_rows=mapping_rows,
            mapping_targets=mapping_targets,
            validation_rows=validation_rows,
            validation_targets=validation_targets,
            config=self.config.model,
            domain=self.domain,
            seed=self.config.sampling.seed,
        )
        encoder = FeatureEncoder(self.domain)
        selected_kind: ModelKind = selection.selected.spec.kind
        selection_predictions = predict_estimator(
            selection.selected.estimator,
            selected_kind,
            encoder,
            validation_rows,
        )

        refit_rows = [*mapping_rows, *validation_rows]
        refit_targets = np.concatenate((mapping_targets, validation_targets))
        final_estimator = refit_selected(
            selection,
            rows=refit_rows,
            targets=refit_targets,
            config=self.config.model,
            domain=self.domain,
            seed=self.config.sampling.seed + 900_001,
        )
        calibration_predictions = predict_estimator(
            final_estimator,
            selected_kind,
            encoder,
            calibration_rows,
        )
        radius = conformal_radius(
            calibration_targets,
            calibration_predictions,
            coverage=self.config.evaluation.conformal_coverage,
        )
        audit_predictions = predict_estimator(
            final_estimator,
            selected_kind,
            encoder,
            audit_rows,
        )
        audit_lower, audit_upper = conformal_bounds(audit_predictions, radius)
        final_latency = measure_single_row_latency(
            final_estimator,
            selected_kind,
            encoder,
            audit_rows[0],
            self.config.model.latency_repetitions,
        )

        validation_report = regression_report(
            validation_rows,
            validation_targets,
            selection_predictions,
            evaluation=self.config.evaluation,
            seed=self.config.sampling.seed + 4_001,
        )
        calibration_report = regression_report(
            calibration_rows,
            calibration_targets,
            calibration_predictions,
            evaluation=self.config.evaluation,
            seed=self.config.sampling.seed + 4_002,
        )
        audit_report = regression_report(
            audit_rows,
            audit_targets,
            audit_predictions,
            evaluation=self.config.evaluation,
            seed=self.config.sampling.seed + 4_003,
        )
        audit_intervals = interval_report(audit_targets, audit_lower, audit_upper)
        coverage = float(audit_intervals["coverage"])
        coverage_passed = (
            self.config.evaluation.minimum_audit_coverage
            <= coverage
            <= self.config.evaluation.maximum_audit_coverage
        )
        latency_passed = final_latency <= self.config.model.max_p95_latency_ms
        warnings: list[str] = []
        if len(calibration_rows) < 20:
            warnings.append(
                "Conformal calibration contains fewer than 20 rows; interval coverage is unstable."
            )
        if not coverage_passed:
            warnings.append(
                "Independent audit interval coverage is outside the configured "
                f"[{self.config.evaluation.minimum_audit_coverage:.0%}, "
                f"{self.config.evaluation.maximum_audit_coverage:.0%}] gate."
            )
        if not latency_passed:
            warnings.append(
                "Final refit exceeds the configured warm single-row p95 latency ceiling."
            )
        early_stopped = bool(store.metadata("early_stopped", False))
        evaluation_report: dict[str, Any] = {
            "schema_version": 1,
            "evaluation_design": {
                "samples_generated_before_adaptive_mapping": True,
                "split_counts": self.config.evaluation.split_counts(),
                "selection_data": ["mapping", "validation"],
                "validation_used_for_residual_acquisition": True,
                "conformal_data": ["calibration"],
                "final_unbiased_data": ["audit"],
                "calibration_or_audit_used_for_acquisition": False,
                "calibration_or_audit_used_for_tuning_or_early_stopping": False,
            },
            "validation_selection": validation_report,
            "calibration": calibration_report,
            "audit": audit_report,
            "audit_interval": audit_intervals,
            "conformal": {
                "nominal_coverage": self.config.evaluation.conformal_coverage,
                "radius": radius,
            },
            "latency": {
                "warm_single_row_p95_ms": final_latency,
                "ceiling_ms": self.config.model.max_p95_latency_ms,
                "passed": latency_passed,
            },
            "promotion_gates": {
                "audit_coverage_passed": coverage_passed,
                "latency_passed": latency_passed,
                "eligible": coverage_passed and latency_passed,
            },
            "early_stopped": early_stopped,
            "stop_reason": store.metadata("stop_reason"),
            "batch_history": store.batch_history(),
            "warnings": warnings,
        }
        model_version = self._model_version(
            selection.report(),
            mapping_records=mapping_records,
            validation_records=validation_records,
            conformal_radius_value=radius,
        )
        completed_samples = store.samples(completed_only=True)
        artifact_dir = export_artifact(
            output_dir=self.output_dir,
            run_id=self.run_id,
            config=self.config,
            domain=self.domain,
            samples=completed_samples,
            estimator=final_estimator,
            model_kind=selected_kind,
            model_version=model_version,
            conformal_radius=radius,
            evaluation_report=evaluation_report,
            selection_report=selection.report(),
            provider_summary=store.provider_summary(),
            warnings=warnings,
            run_started_at_utc=str(store.metadata("run_started_at_utc")),
        )
        return MappingResult(
            run_id=self.run_id,
            artifact_dir=artifact_dir,
            state_database=self.state_database,
            mapping_samples=len(mapping_records),
            evaluation_samples=(
                len(validation_records) + len(calibration_records) + len(audit_records)
            ),
            early_stopped=early_stopped,
            evaluation_report=evaluation_report,
        )

    def _model_version(
        self,
        selection_report: Mapping[str, Any],
        *,
        mapping_records: Sequence[SampleRecord],
        validation_records: Sequence[SampleRecord],
        conformal_radius_value: float,
    ) -> str:
        selected = dict(selection_report["selected"])
        # Wall-clock measurements are reported but are not part of the stable
        # model identity.
        selected.pop("p95_latency_ms", None)
        selected.pop("latency_eligible", None)
        payload = {
            "artifact_schema": 1,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "threadpoolctl_version": importlib.metadata.version("threadpoolctl"),
            "sampling_seed": self.config.sampling.seed,
            "model_config": self.config.model.model_dump(mode="json"),
            "domain": self.domain.to_dict(),
            "selection": selected,
            "fit_observations": [
                [record.split, record.row_hash, record.premium]
                for record in (*mapping_records, *validation_records)
            ],
            "conformal_coverage": self.config.evaluation.conformal_coverage,
            "conformal_radius": conformal_radius_value,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return f"v1-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"
