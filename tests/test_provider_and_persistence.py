from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from pricing_mapper.config import ProviderConfig
from pricing_mapper.domain import CarQuoteInput, DomainSpec
from pricing_mapper.exceptions import (
    PersistenceError,
    ProviderRejected,
    ProviderUnavailable,
)
from pricing_mapper.persistence import RunLock, RunStore
from pricing_mapper.provider import ProviderExecutor, reference_car_quote, resolve_provider


class RetryProvider:
    provider_id = "test.retry"
    thread_safe = False
    max_concurrency = 1

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self, quote: CarQuoteInput) -> float:
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderUnavailable("temporary")
        return reference_car_quote(quote)


def test_retryable_provider_uses_bounded_backoff(valid_row: dict[str, Any]) -> None:
    provider = RetryProvider(2)
    sleeps: list[float] = []
    attempts = []
    executor = ProviderExecutor(
        resolve_provider(supplied=provider),
        ProviderConfig(
            max_retries=2,
            initial_backoff_seconds=0.1,
            maximum_backoff_seconds=0.15,
        ),
        sleep=sleeps.append,
    )
    value = executor.quote(
        CarQuoteInput.from_mapping(valid_row),
        on_attempt=attempts.append,
    )
    assert value >= 0
    assert provider.calls == 3
    assert sleeps == [0.1, 0.15]
    assert [item.outcome for item in attempts] == [
        "retryable_failure",
        "retryable_failure",
        "success",
    ]


def test_permanent_and_invalid_provider_outputs_are_not_retried(
    valid_row: dict[str, Any],
) -> None:
    quote = CarQuoteInput.from_mapping(valid_row)
    calls = 0

    def rejected(_: CarQuoteInput) -> float:
        nonlocal calls
        calls += 1
        raise ProviderRejected("private quote payload must not escape")

    executor = ProviderExecutor(
        resolve_provider(supplied=rejected),
        ProviderConfig(max_retries=5),
        sleep=lambda _: None,
    )
    with pytest.raises(ProviderRejected) as captured:
        executor.quote(quote)
    assert "private quote payload" not in str(captured.value)
    assert calls == 1

    invalid = ProviderExecutor(
        resolve_provider(supplied=lambda _: float("nan")),
        ProviderConfig(max_retries=5),
        sleep=lambda _: None,
    )
    with pytest.raises(ProviderRejected, match="finite"):
        invalid.quote(quote)


def test_parallel_execution_requires_explicit_declaration() -> None:
    with pytest.raises(ValueError, match="thread_safe"):
        ProviderExecutor(
            resolve_provider(supplied=RetryProvider(0)),
            ProviderConfig(concurrency=2),
        )


def test_sqlite_store_checkpoints_each_quote_without_payload_telemetry(
    tmp_path: Path,
    valid_row: dict[str, Any],
) -> None:
    database = tmp_path / "run.sqlite3"
    domain = DomainSpec.default()
    store = RunStore.create(
        database,
        config_fingerprint="abc",
        config_toml="config_version = 1\n",
        domain=domain,
        provider_identity="test",
    )
    try:
        evaluation = domain.sample_lhs(3, __import__("numpy").random.default_rng(2))
        store.register_evaluation_splits(
            {
                "validation": [CarQuoteInput.from_mapping(evaluation[0])],
                "calibration": [CarQuoteInput.from_mapping(evaluation[1])],
                "audit": [CarQuoteInput.from_mapping(evaluation[2])],
            }
        )
        sample = store.register_samples(
            "mapping",
            [CarQuoteInput.from_mapping(valid_row)],
            batch_id=0,
            source="test",
        )[0]
        store.complete_quote(sample.id, sample.row_hash, 123.45, "test")
        assert store.sample_by_id(sample.id).premium == 123.45
        assert store.cached_quote("test", sample.row_hash) == 123.45
    finally:
        store.close()

    connection = sqlite3.connect(database)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(provider_attempts)")}
        assert "row_json" not in columns
        assert "payload" not in columns
    finally:
        connection.close()


def test_store_detects_fingerprint_and_row_tampering(
    tmp_path: Path,
    valid_row: dict[str, Any],
) -> None:
    database = tmp_path / "run.sqlite3"
    domain = DomainSpec.default()
    store = RunStore.create(
        database,
        config_fingerprint="expected",
        config_toml="config_version = 1\n",
        domain=domain,
        provider_identity="provider",
    )
    evaluation = domain.sample_lhs(3, __import__("numpy").random.default_rng(3))
    store.register_evaluation_splits(
        {
            "validation": [CarQuoteInput.from_mapping(evaluation[0])],
            "calibration": [CarQuoteInput.from_mapping(evaluation[1])],
            "audit": [CarQuoteInput.from_mapping(evaluation[2])],
        }
    )
    sample = store.register_samples(
        "mapping",
        [CarQuoteInput.from_mapping(valid_row)],
        batch_id=0,
        source="test",
    )[0]
    store.complete_quote(sample.id, sample.row_hash, 100.0, "provider")
    with pytest.raises(PersistenceError, match="fingerprint"):
        store.validate_integrity(
            config_fingerprint="different",
            domain=domain,
            provider_identity="provider",
        )
    store.close()

    connection = sqlite3.connect(database)
    connection.execute("UPDATE samples SET row_hash = ? WHERE id = ?", ("0" * 64, sample.id))
    connection.commit()
    connection.close()
    reopened = RunStore.open(database)
    try:
        with pytest.raises(PersistenceError, match="row hash"):
            reopened.validate_integrity(
                config_fingerprint="expected",
                domain=domain,
                provider_identity="provider",
            )
    finally:
        reopened.close()


def test_run_lock_is_exclusive(tmp_path: Path) -> None:
    lock_path = tmp_path / "run.lock"
    with RunLock(lock_path), pytest.raises(PersistenceError, match="locked"):  # noqa: SIM117
        with RunLock(lock_path):
            pass
