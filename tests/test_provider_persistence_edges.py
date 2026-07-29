from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from pricing_mapper.config import ProviderConfig
from pricing_mapper.domain import CarQuoteInput, DomainSpec
from pricing_mapper.exceptions import PersistenceError, ProviderRejected, ProviderUnavailable
from pricing_mapper.persistence import RunStore, _json_dumps, validate_run_id
from pricing_mapper.provider import (
    ProviderAttempt,
    ProviderDescriptor,
    ProviderExecutor,
    reference_car_quote,
    resolve_provider,
)


def _store_with_holdouts(path: Path) -> RunStore:
    domain = DomainSpec.default()
    store = RunStore.create(
        path,
        config_fingerprint="fingerprint",
        config_toml="config_version = 1\n",
        domain=domain,
        provider_identity="provider",
    )
    rows = domain.sample_lhs(3, np.random.default_rng(91))
    store.register_evaluation_splits(
        {
            "validation": [CarQuoteInput.from_mapping(rows[0])],
            "calibration": [CarQuoteInput.from_mapping(rows[1])],
            "audit": [CarQuoteInput.from_mapping(rows[2])],
        }
    )
    return store


def test_provider_resolution_and_identity_edges() -> None:
    assert resolve_provider().identity == "pricing_mapper.reference_car_quote.v1"
    loaded = resolve_provider("pricing_mapper.provider:reference_car_quote")
    assert loaded.callable is reference_car_quote
    with pytest.raises(ValueError, match="either"):
        resolve_provider("pricing_mapper.provider:reference_car_quote", reference_car_quote)
    with pytest.raises(ValueError, match="syntax"):
        resolve_provider("bad")
    with pytest.raises(ValueError, match="cannot import"):
        resolve_provider("does_not_exist_anywhere:quote")
    with pytest.raises(ValueError, match="callable"):
        resolve_provider("pricing_mapper.provider:math")
    with pytest.raises(ValueError, match="callable"):
        resolve_provider(supplied=cast(Any, 42))

    class PlainProvider:
        max_concurrency = "invalid"

        def __call__(self, quote: CarQuoteInput) -> float:
            return 1.0

    descriptor = resolve_provider(supplied=PlainProvider())
    assert descriptor.identity.endswith("PlainProvider")
    assert descriptor.max_concurrency == 1


def test_provider_output_exception_and_retry_exhaustion(valid_row: dict[str, Any]) -> None:
    quote = CarQuoteInput.from_mapping(valid_row)
    for invalid in (True, "100", -1.0, float("inf")):
        executor = ProviderExecutor(
            resolve_provider(supplied=lambda _, result=invalid: result),
            ProviderConfig(max_retries=0),
        )
        with pytest.raises(ProviderRejected):
            executor.quote(quote)

    def unexpected(_: CarQuoteInput) -> float:
        raise KeyError("payload must not be persisted")

    with pytest.raises(ProviderRejected, match="unexpected KeyError"):
        ProviderExecutor(
            resolve_provider(supplied=unexpected),
            ProviderConfig(max_retries=0),
        ).quote(quote)

    def unavailable(_: CarQuoteInput) -> float:
        raise ProviderUnavailable("down")

    with pytest.raises(ProviderUnavailable, match="after 2 attempts"):
        ProviderExecutor(
            resolve_provider(supplied=unavailable),
            ProviderConfig(
                max_retries=1,
                initial_backoff_seconds=0.0,
                maximum_backoff_seconds=0.0,
            ),
        ).quote(quote)


def test_parallel_provider_success_failure_and_declared_limit(
    valid_row: dict[str, Any],
) -> None:
    quote = CarQuoteInput.from_mapping(valid_row)
    with pytest.raises(ValueError, match="exceeds"):
        ProviderExecutor(
            ProviderDescriptor(reference_car_quote, "reference", True, 1),
            ProviderConfig(concurrency=2),
        )

    completed: list[int] = []
    executor = ProviderExecutor(
        resolve_provider(supplied=reference_car_quote),
        ProviderConfig(concurrency=2, max_retries=0),
    )
    results = executor.quote_many(
        [(1, quote), (2, quote)],
        on_complete=lambda sample_id, _hash, _value: completed.append(sample_id),
    )
    assert set(results) == {1, 2}
    assert set(completed) == {1, 2}
    assert executor.quote_many([]) == {}

    calls = 0

    def sometimes(_: CarQuoteInput) -> float:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderRejected("stop")
        return 1.0

    sometimes.thread_safe = True  # type: ignore[attr-defined]
    sometimes.max_concurrency = 2  # type: ignore[attr-defined]
    with pytest.raises(ProviderRejected):
        ProviderExecutor(
            resolve_provider(supplied=sometimes),
            ProviderConfig(concurrency=2, max_retries=0),
        ).quote_many([(1, quote), (2, quote)])


def test_run_store_open_schema_rng_metadata_and_registration_edges(
    tmp_path: Path,
    valid_row: dict[str, Any],
) -> None:
    for bad in ("", "../bad", "_starts-wrong", "a" * 65):
        with pytest.raises(ValueError, match="run_id"):
            validate_run_id(bad)
    assert validate_run_id("valid.run-1") == "valid.run-1"
    with pytest.raises(FileNotFoundError):
        RunStore.open(tmp_path / "missing.sqlite3")
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not sqlite")
    with pytest.raises(PersistenceError, match="cannot open"):
        RunStore.open(corrupt)
    incomplete = tmp_path / "incomplete.sqlite3"
    connection = sqlite3.connect(incomplete)
    connection.execute("PRAGMA user_version = 1")
    connection.close()
    with pytest.raises(PersistenceError, match="missing required tables"):
        RunStore.open(incomplete)

    assert json.loads(_json_dumps(np.asarray([1, 2]))) == [1, 2]
    assert json.loads(_json_dumps(np.int64(2))) == 2
    assert json.loads(_json_dumps(np.float64(2.5))) == 2.5
    with pytest.raises(TypeError, match="serialize"):
        _json_dumps(object())

    database = tmp_path / "state.sqlite3"
    store = _store_with_holdouts(database)
    try:
        with pytest.raises(FileExistsError):
            RunStore.create(
                database,
                config_fingerprint="x",
                config_toml="",
                domain=DomainSpec.default(),
                provider_identity="provider",
            )
        assert store.metadata("absent", "fallback") == "fallback"
        store.set_metadata("custom", {"ok": True})
        assert store.metadata("custom") == {"ok": True}
        with pytest.raises(PersistenceError, match="RNG"):
            store.rng_state("missing")
        state = np.random.default_rng(1).bit_generator.state
        store.save_rng_state("mapping", state)
        assert store.rng_state("mapping") == state
        assert store.register_samples("mapping", [], batch_id=0, source="empty") == []
        with pytest.raises(ValueError, match="unknown"):
            store.register_samples(cast(Any, "bad"), [valid_row], batch_id=None, source="x")
        with pytest.raises(ValueError, match="batch_id"):
            store.register_samples("mapping", [valid_row], batch_id=None, source="x")
        with pytest.raises(ValueError, match="cannot have"):
            store.register_samples("audit", [valid_row], batch_id=3, source="x")
        with pytest.raises(ValueError, match="requires"):
            store.register_evaluation_splits(cast(Any, {"audit": []}))
        with pytest.raises(ValueError, match="at least one"):
            store.register_evaluation_splits(
                cast(
                    Any,
                    {"validation": [], "calibration": [], "audit": []},
                )
            )
        with pytest.raises(PersistenceError, match="duplicate"):
            rows = DomainSpec.default().sample_lhs(1, np.random.default_rng(91))
            quote = CarQuoteInput.from_mapping(rows[0])
            store.register_evaluation_splits(
                {"validation": [quote], "calibration": [quote], "audit": [quote]}
            )
        with pytest.raises(PersistenceError, match="does not exist"):
            store.sample_by_id(99_999)
        assert store.pending_samples([]) == []
        with pytest.raises(ValueError, match="invalid split"):
            store.pending_samples(cast(Any, ["bad"]))
    finally:
        store.close()

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 99")
    connection.commit()
    connection.close()
    with pytest.raises(PersistenceError, match="schema version"):
        RunStore.open(database)


def test_run_store_quote_batch_attempt_and_integrity_edges(
    tmp_path: Path,
    valid_row: dict[str, Any],
) -> None:
    store = _store_with_holdouts(tmp_path / "state.sqlite3")
    quote = CarQuoteInput.from_mapping(valid_row)
    sample = store.register_samples(
        "mapping",
        [quote],
        batch_id=0,
        source="test",
    )[0]
    try:
        with pytest.raises(PersistenceError, match="finite"):
            store.complete_quote(sample.id, sample.row_hash, float("nan"), "provider")
        with pytest.raises(PersistenceError, match="does not exist"):
            store.complete_quote(99_999, sample.row_hash, 1.0, "provider")
        with pytest.raises(PersistenceError, match="does not match"):
            store.complete_quote(sample.id, "0" * 64, 1.0, "provider")
        store.complete_quote(sample.id, sample.row_hash, 100.0, "provider")
        store.complete_quote(sample.id, sample.row_hash, 100.0, "provider")
        with pytest.raises(PersistenceError, match="different"):
            store.complete_quote(sample.id, sample.row_hash, 101.0, "provider")
        assert store.unevaluated_batches() == [0]
        assert store.samples("mapping", completed_only=True, batch_id=0)[0].id == sample.id
        with pytest.raises(PersistenceError, match="current state"):
            store.mark_batch_evaluated(99, {})
        store.mark_batch_evaluated(0, {"validation_mae": 1.0})
        assert store.batch_history()[0]["status"] == "evaluated"
        assert store.next_batch_id() == 1
        assert sample.row_hash in store.all_row_hashes()

        attempt = ProviderAttempt(
            provider_identity="provider",
            row_hash=sample.row_hash,
            attempt_number=1,
            outcome="success",
            latency_ms=1.5,
            error_type=None,
            occurred_at_utc="2026-01-01T00:00:00+00:00",
        )
        store.record_attempt(attempt)
        summary = store.provider_summary()
        assert summary["attempts"] == summary["successes"] == 1
        assert summary["total_latency_ms"] == 1.5

        store.validate_integrity(
            config_fingerprint="fingerprint",
            domain=DomainSpec.default(),
            provider_identity="provider",
        )
        with pytest.raises(PersistenceError, match="provider identity"):
            store.validate_integrity(
                config_fingerprint="fingerprint",
                domain=DomainSpec.default(),
                provider_identity="different",
            )
        versions = store.metadata("runtime_versions")
        store.set_metadata("runtime_versions", {**versions, "numpy": "0.0.0"})
        with pytest.raises(PersistenceError, match="dependency versions"):
            store.validate_integrity(
                config_fingerprint="fingerprint",
                domain=DomainSpec.default(),
                provider_identity="provider",
            )
        store.set_metadata("runtime_versions", versions)
        narrowed_bounds = dict(DomainSpec.default().numeric)
        narrowed_bounds["vehicle_value"] = type(narrowed_bounds["vehicle_value"])(
            2_000.0,
            150_000.0,
            False,
        )
        with pytest.raises(PersistenceError, match="domain snapshot"):
            store.validate_integrity(
                config_fingerprint="fingerprint",
                domain=DomainSpec(
                    numeric=narrowed_bounds,
                    categorical=DomainSpec.default().categorical,
                ),
                provider_identity="provider",
            )
        store.set_metadata("schema_version", 99)
        with pytest.raises(PersistenceError, match="schema version"):
            store.validate_integrity(
                config_fingerprint="fingerprint",
                domain=DomainSpec.default(),
                provider_identity="provider",
            )
    finally:
        store.close()


def test_seed_registration_rejects_bad_or_duplicate_rows(
    tmp_path: Path,
    valid_row: dict[str, Any],
) -> None:
    store = _store_with_holdouts(tmp_path / "state.sqlite3")
    quote = CarQuoteInput.from_mapping(valid_row)
    try:
        assert store.register_seed_samples([]) == []
        with pytest.raises(PersistenceError, match="finite"):
            store.register_seed_samples([(quote, -1.0)])
        store.register_seed_samples([(quote, 100.0)])
        with pytest.raises((PersistenceError, sqlite3.IntegrityError)):
            store.register_seed_samples([(quote, 100.0)], batch_id=1)
    finally:
        store.close()
