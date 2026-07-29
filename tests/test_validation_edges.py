from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from pricing_mapper.config import (
    AcquisitionConfig,
    ArtifactConfig,
    BoundsConfig,
    DomainConfig,
    EvaluationConfig,
    MapperConfig,
    ModelConfig,
    ProviderConfig,
    SamplingConfig,
    load_config,
    write_config,
)
from pricing_mapper.domain import (
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
    DEFAULT_CATEGORIES,
    INTEGER_FIELDS,
    CarQuoteInput,
    DomainSpec,
    NumericDomain,
    canonicalize_sample,
    rows_json_schema,
)
from pricing_mapper.exceptions import ConfigurationError, DomainValidationError


def test_numeric_and_domain_snapshot_validation_edges(valid_row: dict[str, Any]) -> None:
    with pytest.raises(DomainValidationError, match="low < high"):
        NumericDomain(1.0, 1.0, False)
    with pytest.raises(DomainValidationError, match="integral"):
        NumericDomain(0.5, 2.0, True)

    default = DomainSpec.default()
    with pytest.raises(DomainValidationError, match="numeric fields"):
        DomainSpec(numeric={}, categorical=default.categorical)
    with pytest.raises(DomainValidationError, match="categorical fields"):
        DomainSpec(numeric=default.numeric, categorical={})

    wrong_integer = dict(default.numeric)
    wrong_integer["annual_km"] = NumericDomain(1_000.0, 60_000.0, False)
    with pytest.raises(DomainValidationError, match="integer domain"):
        DomainSpec(numeric=wrong_integer, categorical=default.categorical)

    wrong_continuous = dict(default.numeric)
    wrong_continuous["driver_age"] = NumericDomain(17.0, 90.0, True)
    with pytest.raises(DomainValidationError, match="continuous domain"):
        DomainSpec(numeric=wrong_continuous, categorical=default.categorical)

    duplicate_categories = dict(default.categorical)
    duplicate_categories["usage"] = ("private", "private")
    with pytest.raises(DomainValidationError, match="unique"):
        DomainSpec(numeric=default.numeric, categorical=duplicate_categories)

    infeasible = dict(default.numeric)
    infeasible["years_licensed"] = NumericDomain(2.0, 70.0, True)
    infeasible["driver_age"] = NumericDomain(17.0, 18.0, False)
    with pytest.raises(DomainValidationError, match="infeasible"):
        DomainSpec(numeric=infeasible, categorical=default.categorical)

    restored = DomainSpec.from_dict(default.to_dict())
    assert restored == default
    for invalid_snapshot in (
        {"schema_version": 2, "numeric": {}, "categorical": {}},
        {"schema_version": 1, "numeric": {}, "categorical": {}, "unknown": True},
        {
            **default.to_dict(),
            "numeric": {
                **default.to_dict()["numeric"],
                "annual_km": {
                    "low": 1_000.0,
                    "high": 60_000.0,
                    "integer": "true",
                },
            },
        },
    ):
        with pytest.raises(DomainValidationError, match="snapshot"):
            DomainSpec.from_dict(invalid_snapshot)
    with pytest.raises(DomainValidationError, match="snapshot"):
        DomainSpec.from_dict({"numeric": [], "categorical": {}})
    with pytest.raises(DomainValidationError, match="snapshot"):
        DomainSpec.from_dict({"numeric": {}, "categorical": {}})

    quote = CarQuoteInput.from_mapping(valid_row)
    assert CarQuoteInput.from_mapping(quote, domain=default) == quote
    invalid_integer = quote.model_copy(update={"annual_km": 1_000.5})
    with pytest.warns(UserWarning), pytest.raises(DomainValidationError, match="integer"):
        default.validate(invalid_integer)
    invalid_tenure = quote.model_copy(update={"driver_age": 18.0, "years_licensed": 20})
    with pytest.raises(DomainValidationError, match="years_licensed"):
        CarQuoteInput.from_mapping(invalid_tenure, domain=default)
    invalid_category = quote.model_copy(update={"usage": "fleet"})
    with pytest.raises(DomainValidationError, match="levels"):
        default.validate(invalid_category)


def test_sampling_and_internal_canonicalization_error_edges(
    valid_row: dict[str, Any],
) -> None:
    domain = DomainSpec.default()
    rng = np.random.default_rng(4)
    assert domain.sample_lhs(0, rng) == []
    assert domain.sample_random(0, rng) == []
    for invalid in (-1, True, 1.2):
        with pytest.raises(ValueError, match="non-negative"):
            domain.sample_lhs(cast(Any, invalid), rng)
        with pytest.raises(ValueError, match="non-negative"):
            domain.sample_random(cast(Any, invalid), rng)
    random_rows = domain.sample_random(5, rng)
    assert len(random_rows) == 5
    for row in random_rows:
        CarQuoteInput.from_mapping(row)

    missing = dict(valid_row)
    missing.pop("rating")
    with pytest.raises(DomainValidationError, match="missing"):
        canonicalize_sample(missing, domain)
    bad_category = dict(valid_row, parking="carport")
    with pytest.raises(DomainValidationError, match="parking"):
        canonicalize_sample(bad_category, domain)
    bad_number = dict(valid_row, driver_age=True)
    with pytest.raises(DomainValidationError, match="finite"):
        canonicalize_sample(bad_number, domain)
    overflow = dict(valid_row, vehicle_value=10**10_000)
    with pytest.raises(DomainValidationError, match="finite"):
        canonicalize_sample(overflow, domain)
    assert rows_json_schema()["additionalProperties"] is False


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BoundsConfig(low=2.0, high=1.0),
        lambda: AcquisitionConfig(weights={"unknown": 1.0}),
        lambda: AcquisitionConfig(weights={"uncertainty": -1.0}),
        lambda: AcquisitionConfig(weights={"uncertainty": 0.0}),
        lambda: SamplingConfig(mapping_budget=2, initial_size=3),
        lambda: ModelConfig(monotonic_constraints={"unknown": 1}),
        lambda: ModelConfig(monotonic_constraints={"vehicle_value": 2}),
        lambda: EvaluationConfig(
            minimum_audit_coverage=0.96,
            maximum_audit_coverage=0.95,
        ),
        lambda: EvaluationConfig(
            evaluation_budget=3,
            validation_fraction=0.98,
            calibration_fraction=0.01,
            audit_fraction=0.01,
        ),
        lambda: ProviderConfig(callable="not-a-target"),
        lambda: ProviderConfig(initial_backoff_seconds=2.0, maximum_backoff_seconds=1.0),
        lambda: ArtifactConfig(output_dir=""),
        lambda: ArtifactConfig(state_dir_name="."),
        lambda: ArtifactConfig(state_dir_name="../state"),
        lambda: ArtifactConfig(model_card_title=" "),
    ],
)
def test_nested_config_rejects_invalid_contracts(factory: Any) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_domain_config_rejects_missing_expanded_fractional_and_infeasible() -> None:
    default = DomainConfig()
    missing = dict(default.bounds)
    missing.pop("excess")
    with pytest.raises(ValidationError, match="mismatch"):
        DomainConfig(bounds=missing)

    expanded = dict(default.bounds)
    expanded["driver_age"] = BoundsConfig(low=16.0, high=90.0)
    with pytest.raises(ValidationError, match="within"):
        DomainConfig(bounds=expanded)

    fractional = dict(default.bounds)
    fractional["annual_km"] = BoundsConfig(low=1_000.5, high=60_000.0)
    with pytest.raises(ValidationError, match="integer"):
        DomainConfig(bounds=fractional)

    infeasible = dict(default.bounds)
    infeasible["driver_age"] = BoundsConfig(low=17.0, high=18.0)
    infeasible["years_licensed"] = BoundsConfig(low=2.0, high=3.0)
    with pytest.raises(ValidationError, match="infeasible"):
        DomainConfig(bounds=infeasible)


def test_config_file_io_failures_and_writer(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_config(tmp_path / "missing.toml")
    malformed = tmp_path / "malformed.toml"
    malformed.write_text("[broken", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_config(malformed)

    target = tmp_path / "written.toml"
    write_config(target, MapperConfig())
    assert load_config(target) == MapperConfig()
    assert set(DEFAULT_CATEGORIES) == set(CATEGORICAL_FIELDS)
    assert set(CONTINUOUS_FIELDS) | set(INTEGER_FIELDS) == set(
        MapperConfig().resolved_domain.numeric
    )
