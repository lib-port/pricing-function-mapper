from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from pricing_mapper.config import (
    BoundsConfig,
    DomainConfig,
    EvaluationConfig,
    MapperConfig,
    OllamaConfig,
    OptimizerConfig,
    SamplingConfig,
    config_json_schema,
    config_toml,
    load_config,
)
from pricing_mapper.domain import (
    FIELD_ORDER,
    V1_VEHICLE_YEAR_MAX,
    CarQuoteInput,
    DomainSpec,
    canonicalize_sample,
    default_numeric_bounds,
    row_key,
)
from pricing_mapper.exceptions import ConfigurationError, DomainValidationError
from pricing_mapper.provider import REFERENCE_RATING_YEAR


def test_car_quote_input_is_exact_and_strict(valid_row: dict[str, object]) -> None:
    quote = CarQuoteInput.from_mapping(valid_row)
    assert tuple(quote.as_dict()) == FIELD_ORDER

    missing = dict(valid_row)
    missing.pop("rating")
    with pytest.raises(DomainValidationError, match="rating"):
        CarQuoteInput.from_mapping(missing)

    unknown = dict(valid_row, typo=1)
    with pytest.raises(DomainValidationError, match="typo"):
        CarQuoteInput.from_mapping(unknown)

    boolean = dict(valid_row, claims_5y=False)
    with pytest.raises(DomainValidationError):
        CarQuoteInput.from_mapping(boolean)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("driver_age", 16.99),
        ("vehicle_value", 200_001.0),
        ("postcode_risk", float("nan")),
        ("theft_risk", float("inf")),
        ("vehicle_year", 9999),
    ],
)
def test_inference_rejects_out_of_domain_without_clipping(
    valid_row: dict[str, object],
    field: str,
    value: object,
) -> None:
    row = dict(valid_row)
    row[field] = value
    with pytest.raises(DomainValidationError):
        CarQuoteInput.from_mapping(row)


def test_v1_vehicle_year_ceiling_is_release_stable(valid_row: dict[str, object]) -> None:
    assert V1_VEHICLE_YEAR_MAX == 2026
    assert REFERENCE_RATING_YEAR == V1_VEHICLE_YEAR_MAX
    assert default_numeric_bounds()["vehicle_year"] == (1998.0, 2026.0)

    maximum = dict(valid_row, vehicle_year=V1_VEHICLE_YEAR_MAX)
    assert CarQuoteInput.from_mapping(maximum).vehicle_year == V1_VEHICLE_YEAR_MAX

    future = dict(valid_row, vehicle_year=V1_VEHICLE_YEAR_MAX + 1)
    with pytest.raises(DomainValidationError, match="vehicle_year"):
        CarQuoteInput.from_mapping(future)


def test_cross_field_tenure_is_rejected(valid_row: dict[str, object]) -> None:
    row = dict(valid_row, driver_age=18.0, years_licensed=5)
    with pytest.raises(DomainValidationError, match="years_licensed"):
        CarQuoteInput.from_mapping(row)


def test_internal_sample_canonicalization_is_explicit(valid_row: dict[str, object]) -> None:
    row = dict(valid_row, driver_age=12.0, years_licensed=99, postcode_risk=3.0)
    canonical = canonicalize_sample(row, DomainSpec.default())
    assert canonical["driver_age"] == 17.0
    assert canonical["years_licensed"] == 1
    assert canonical["postcode_risk"] == 1.0


@given(st.integers(min_value=0, max_value=40), st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=30)
def test_lhs_samples_are_valid_unique_and_deterministic(count: int, seed: int) -> None:
    domain = DomainSpec.default()
    first = domain.sample_lhs(count, np.random.default_rng(seed))
    second = domain.sample_lhs(count, np.random.default_rng(seed))
    assert first == second
    assert len({row_key(row) for row in first}) == len(first)
    for row in first:
        CarQuoteInput.from_mapping(row, domain=domain)


def test_domain_specific_bounds_reject_not_clip(valid_row: dict[str, object]) -> None:
    default = DomainConfig()
    bounds = dict(default.bounds)
    bounds["vehicle_value"] = BoundsConfig(low=10_000.0, high=50_000.0)
    domain = DomainConfig(bounds=bounds).to_domain()
    with pytest.raises(DomainValidationError, match="outside"):
        CarQuoteInput.from_mapping(dict(valid_row, vehicle_value=60_000.0), domain=domain)


def test_toml_round_trip_and_unknown_rejection(tmp_path: Path) -> None:
    config = MapperConfig()
    path = tmp_path / "config.toml"
    path.write_text(config_toml(config), encoding="utf-8")
    assert load_config(path) == config

    bad = tmp_path / "bad.toml"
    bad.write_text("config_version = 1\nunknown = true\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown"):
        load_config(bad)

    legacy = tmp_path / "legacy.json"
    legacy.write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="TOML"):
        load_config(legacy)


def test_evaluation_allocation_is_exact() -> None:
    evaluation = EvaluationConfig(evaluation_budget=11)
    counts = evaluation.split_counts()
    assert counts == {"validation": 5, "calibration": 3, "audit": 3}
    assert sum(counts.values()) == 11
    with pytest.raises(ValidationError, match="sum"):
        EvaluationConfig(
            validation_fraction=0.5,
            calibration_fraction=0.3,
            audit_fraction=0.3,
        )


def test_generated_config_schema_forbids_unknown_settings() -> None:
    schema = config_json_schema()
    assert schema["additionalProperties"] is False
    serialized = json.dumps(schema)
    assert "mapping_budget" in serialized
    assert "max_p95_latency_ms" in serialized
    assert "required_digest" in serialized


def test_ollama_optimizer_is_opt_in_pinned_and_fingerprinted() -> None:
    digest = "sha256:" + "a" * 64
    ollama = OllamaConfig(required_digest=digest)
    configured = MapperConfig(
        sampling=SamplingConfig(strategy="bayesian"),
        optimizer=OptimizerConfig(ollama=ollama),
    )
    assert configured.optimizer.ollama == ollama
    assert configured.fingerprint != MapperConfig().fingerprint
    with pytest.raises(ValidationError, match=r"requires sampling\.strategy"):
        MapperConfig(optimizer=OptimizerConfig(ollama=ollama))
    with pytest.raises(ValidationError, match="full lowercase"):
        OllamaConfig(required_digest="6fd349357287")
    with pytest.raises(ValidationError, match="origin"):
        OllamaConfig(required_digest=digest, endpoint="http://user:pass@localhost/path")
    with pytest.raises(ValidationError, match="60"):
        OllamaConfig(required_digest=digest, timeout_seconds=61.0)
