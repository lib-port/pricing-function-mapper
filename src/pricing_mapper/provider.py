"""Trusted local quote-provider loading, validation, retry, and telemetry."""

from __future__ import annotations

import importlib
import logging
import math
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from typing import Any

import numpy as np

from pricing_mapper.config import ProviderConfig
from pricing_mapper.domain import V1_VEHICLE_YEAR_MAX, CarQuoteInput, row_key
from pricing_mapper.exceptions import ProviderRejected, ProviderUnavailable
from pricing_mapper.types import QuoteProvider

REFERENCE_RATING_YEAR = V1_VEHICLE_YEAR_MAX


@dataclass(frozen=True)
class ProviderDescriptor:
    """Resolved callable and concurrency declarations."""

    callable: QuoteProvider
    identity: str
    thread_safe: bool
    max_concurrency: int


@dataclass(frozen=True)
class ProviderAttempt:
    """Payload-free telemetry for one provider invocation."""

    provider_identity: str
    row_hash: str
    attempt_number: int
    outcome: str
    latency_ms: float
    error_type: str | None
    occurred_at_utc: str


AttemptCallback = Callable[[ProviderAttempt], None]
CompletionCallback = Callable[[int, str, float], None]


def reference_car_quote(quote: CarQuoteInput) -> float:
    """Deterministic synthetic comprehensive-car reference provider."""

    x = quote.as_dict()
    age = float(x["driver_age"])
    years_licensed = int(x["years_licensed"])
    claims = int(x["claims_5y"])
    convictions = int(x["convictions_5y"])
    postcode_risk = float(x["postcode_risk"])
    annual_km = int(x["annual_km"])
    vehicle_value = float(x["vehicle_value"])
    vehicle_year = int(x["vehicle_year"])
    theft_risk = float(x["theft_risk"])
    excess = int(x["excess"])

    base = 300.0 + 0.0105 * vehicle_value
    base *= 1.0 + 0.62 * postcode_risk

    vehicle_age = max(0, REFERENCE_RATING_YEAR - vehicle_year)
    base *= 1.0 + 0.010 * min(vehicle_age, 10) + 0.020 * max(0, vehicle_age - 10)

    if age < 21:
        base *= 1.95
    elif age < 25:
        base *= 1.52
    elif age < 35:
        base *= 1.18
    elif age >= 60:
        base *= 1.10

    base *= 1.0 - 0.06 * np.tanh((years_licensed - 3) / 6.0)
    base *= (1.0 + 0.22 * claims + 0.28 * convictions) * (1.0 + 0.10 * postcode_risk * (claims > 0))
    base *= (1.0 + 0.55 * theft_risk) * {
        "garage": 0.92,
        "driveway": 1.00,
        "street": 1.13,
    }[str(x["parking"])]
    base *= {"private": 0.98, "commute": 1.00, "business": 1.13}[str(x["usage"])]
    base *= 1.0 + 0.11 * np.tanh((annual_km - 12_000) / 12_000.0)
    base *= {"none": 1.00, "basic": 1.04, "premium": 1.08}[str(x["hire_car"])]
    if x["windscreen"] == "yes":
        base *= 1.02
    if x["rating"] == "agreed":
        base *= 1.015
    base *= 1.0 - 0.16 * np.tanh((excess - 600) / 700.0)
    return float(np.round(max(base, 260.0), 2))


# Explicit provider declarations permit reproducible bounded parallel execution.
reference_car_quote.provider_id = "pricing_mapper.reference_car_quote.v1"  # type: ignore[attr-defined]
reference_car_quote.thread_safe = True  # type: ignore[attr-defined]
reference_car_quote.max_concurrency = 32  # type: ignore[attr-defined]


def _callable_identity(provider: QuoteProvider, configured_name: str | None) -> str:
    explicit = getattr(provider, "provider_id", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if configured_name:
        return configured_name
    module = getattr(provider, "__module__", provider.__class__.__module__)
    qualname = getattr(provider, "__qualname__", provider.__class__.__qualname__)
    return f"{module}:{qualname}"


def resolve_provider(
    configured: str | None = None,
    supplied: QuoteProvider | None = None,
) -> ProviderDescriptor:
    """Resolve one trusted provider and its declared execution capabilities."""

    if supplied is not None and configured is not None:
        raise ValueError("supply either provider.callable or a QuoteProvider object, not both")

    provider: Any
    if supplied is not None:
        provider = supplied
    elif configured is None:
        provider = reference_car_quote
    else:
        target = configured.strip()
        if target.count(":") != 1:
            raise ValueError("provider.callable must use 'module:function' syntax")
        module_name, attribute_name = target.split(":", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ValueError(f"cannot import quote-provider module {module_name!r}: {exc}") from exc
        provider = getattr(module, attribute_name, None)
        if not callable(provider):
            raise ValueError(f"quote provider {target!r} did not resolve to a callable")

    if not callable(provider):
        raise ValueError("QuoteProvider must be callable")
    identity = _callable_identity(provider, configured)
    thread_safe = getattr(provider, "thread_safe", False) is True
    raw_limit = getattr(provider, "max_concurrency", 1)
    max_concurrency = int(raw_limit) if isinstance(raw_limit, int) and raw_limit >= 1 else 1
    return ProviderDescriptor(
        callable=provider,
        identity=identity,
        thread_safe=thread_safe,
        max_concurrency=max_concurrency,
    )


class ProviderExecutor:
    """Validate provider I/O and apply bounded retry policy."""

    def __init__(
        self,
        descriptor: ProviderDescriptor,
        config: ProviderConfig,
        *,
        logger: logging.Logger | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if config.concurrency > 1 and not descriptor.thread_safe:
            raise ValueError(
                "provider concurrency > 1 requires the provider to declare thread_safe=True"
            )
        if config.concurrency > descriptor.max_concurrency:
            raise ValueError(
                f"provider concurrency {config.concurrency} exceeds the declared "
                f"limit {descriptor.max_concurrency}"
            )
        self.descriptor = descriptor
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._sleep = sleep

    @staticmethod
    def _validate_output(raw: Any) -> float:
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
            raise ProviderRejected("quote provider must return a numeric premium")
        try:
            value = float(raw)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ProviderRejected("quote provider must return a numeric premium") from exc
        if not math.isfinite(value) or value < 0:
            raise ProviderRejected("quote provider must return a finite, non-negative premium")
        return value

    def quote(
        self,
        quote: CarQuoteInput,
        *,
        on_attempt: AttemptCallback | None = None,
    ) -> float:
        """Invoke one quote with retries only for ``ProviderUnavailable``."""

        quote = CarQuoteInput.from_mapping(quote)
        hashed_row = row_key(quote)
        total_attempts = self.config.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            started = time.perf_counter()
            try:
                raw = self.descriptor.callable(quote)
            except ProviderUnavailable as exc:
                latency_ms = (time.perf_counter() - started) * 1_000.0
                self._record(
                    hashed_row,
                    attempt,
                    "retryable_failure",
                    latency_ms,
                    type(exc).__name__,
                    on_attempt,
                )
                if attempt >= total_attempts:
                    raise ProviderUnavailable(
                        f"provider {self.descriptor.identity!r} remained unavailable "
                        f"after {total_attempts} attempts"
                    ) from exc
                delay = min(
                    self.config.maximum_backoff_seconds,
                    self.config.initial_backoff_seconds * (2 ** (attempt - 1)),
                )
                if delay > 0:
                    self._sleep(delay)
                continue
            except ProviderRejected as exc:
                latency_ms = (time.perf_counter() - started) * 1_000.0
                self._record(
                    hashed_row,
                    attempt,
                    "permanent_failure",
                    latency_ms,
                    type(exc).__name__,
                    on_attempt,
                )
                raise ProviderRejected(
                    f"provider {self.descriptor.identity!r} permanently rejected "
                    f"row hash {hashed_row[:12]}"
                ) from exc
            except Exception as exc:
                latency_ms = (time.perf_counter() - started) * 1_000.0
                self._record(
                    hashed_row,
                    attempt,
                    "permanent_failure",
                    latency_ms,
                    type(exc).__name__,
                    on_attempt,
                )
                raise ProviderRejected(
                    f"provider {self.descriptor.identity!r} raised an unexpected "
                    f"{type(exc).__name__}"
                ) from exc
            try:
                value = self._validate_output(raw)
            except ProviderRejected as exc:
                latency_ms = (time.perf_counter() - started) * 1_000.0
                self._record(
                    hashed_row,
                    attempt,
                    "permanent_failure",
                    latency_ms,
                    type(exc).__name__,
                    on_attempt,
                )
                raise
            latency_ms = (time.perf_counter() - started) * 1_000.0
            self._record(
                hashed_row,
                attempt,
                "success",
                latency_ms,
                None,
                on_attempt,
            )
            return value
        raise AssertionError("provider retry loop exhausted without returning or raising")

    def _record(
        self,
        hashed_row: str,
        attempt: int,
        outcome: str,
        latency_ms: float,
        error_type: str | None,
        callback: AttemptCallback | None,
    ) -> None:
        record = ProviderAttempt(
            provider_identity=self.descriptor.identity,
            row_hash=hashed_row,
            attempt_number=attempt,
            outcome=outcome,
            latency_ms=latency_ms,
            error_type=error_type,
            occurred_at_utc=datetime.now(UTC).isoformat(),
        )
        if callback is not None:
            callback(record)
        if outcome != "success":
            self.logger.warning(
                "Quote provider %s: %s for row hash %s (attempt %d)",
                self.descriptor.identity,
                outcome,
                hashed_row[:12],
                attempt,
            )

    def quote_many(
        self,
        items: Iterable[tuple[int, CarQuoteInput]],
        *,
        on_attempt: AttemptCallback | None = None,
        on_complete: CompletionCallback | None = None,
    ) -> dict[int, float]:
        """Quote rows sequentially by default, or with declared safe concurrency."""

        materialized = list(items)
        results: dict[int, float] = {}
        if self.config.concurrency == 1:
            for sample_id, quote in materialized:
                value = self.quote(quote, on_attempt=on_attempt)
                results[sample_id] = value
                if on_complete is not None:
                    on_complete(sample_id, row_key(quote), value)
            return results

        with ThreadPoolExecutor(
            max_workers=self.config.concurrency,
            thread_name_prefix="quote-provider",
        ) as pool:
            futures: dict[Future[float], tuple[int, CarQuoteInput]] = {
                pool.submit(self.quote, quote, on_attempt=on_attempt): (sample_id, quote)
                for sample_id, quote in materialized
            }
            try:
                for future in as_completed(futures):
                    sample_id, quote = futures[future]
                    value = future.result()
                    results[sample_id] = value
                    if on_complete is not None:
                        on_complete(sample_id, row_key(quote), value)
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
        return results
