"""Strict comprehensive-car-insurance input validation and internal sampling."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pricing_mapper.exceptions import DomainValidationError

Usage = Literal["private", "commute", "business"]
Parking = Literal["garage", "driveway", "street"]
HireCar = Literal["none", "basic", "premium"]
Windscreen = Literal["no", "yes"]
Rating = Literal["market", "agreed"]

FIELD_ORDER: tuple[str, ...] = (
    "driver_age",
    "years_licensed",
    "vehicle_year",
    "vehicle_value",
    "annual_km",
    "claims_5y",
    "convictions_5y",
    "postcode_risk",
    "theft_risk",
    "excess",
    "usage",
    "parking",
    "hire_car",
    "windscreen",
    "rating",
)
CONTINUOUS_FIELDS: tuple[str, ...] = (
    "driver_age",
    "vehicle_value",
    "postcode_risk",
    "theft_risk",
)
INTEGER_FIELDS: tuple[str, ...] = (
    "years_licensed",
    "vehicle_year",
    "annual_km",
    "claims_5y",
    "convictions_5y",
    "excess",
)
CATEGORICAL_FIELDS: tuple[str, ...] = (
    "usage",
    "parking",
    "hire_car",
    "windscreen",
    "rating",
)

DEFAULT_CATEGORIES: dict[str, tuple[str, ...]] = {
    "usage": ("private", "commute", "business"),
    "parking": ("garage", "driveway", "street"),
    "hire_car": ("none", "basic", "premium"),
    "windscreen": ("no", "yes"),
    "rating": ("market", "agreed"),
}

V1_VEHICLE_YEAR_MAX = 2026


def default_numeric_bounds() -> dict[str, tuple[float, float]]:
    """Return a fresh copy of the supported v1 numeric bounds."""

    return {
        "driver_age": (17.0, 90.0),
        "years_licensed": (0.0, 70.0),
        "vehicle_year": (1998.0, float(V1_VEHICLE_YEAR_MAX)),
        "vehicle_value": (2_000.0, 200_000.0),
        "annual_km": (1_000.0, 60_000.0),
        "claims_5y": (0.0, 6.0),
        "convictions_5y": (0.0, 6.0),
        "postcode_risk": (0.0, 1.0),
        "theft_risk": (0.0, 1.0),
        "excess": (0.0, 5_000.0),
    }


class CarQuoteInput(BaseModel):
    """The stable, strict 15-field public input model.

    Configuration-specific bounds are checked by :class:`DomainSpec`. This base
    model enforces the complete v1 car-insurance schema and its supported outer
    limits. Unknown fields, booleans in numeric fields, NaN, infinity, and
    infeasible licence tenure are rejected.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )

    driver_age: float = Field(ge=17.0, le=90.0)
    years_licensed: int = Field(ge=0, le=70)
    vehicle_year: int = Field(ge=1998, le=V1_VEHICLE_YEAR_MAX)
    vehicle_value: float = Field(ge=2_000.0, le=200_000.0)
    annual_km: int = Field(ge=1_000, le=60_000)
    claims_5y: int = Field(ge=0, le=6)
    convictions_5y: int = Field(ge=0, le=6)
    postcode_risk: float = Field(ge=0.0, le=1.0)
    theft_risk: float = Field(ge=0.0, le=1.0)
    excess: int = Field(ge=0, le=5_000)
    usage: Usage
    parking: Parking
    hire_car: HireCar
    windscreen: Windscreen
    rating: Rating

    @model_validator(mode="after")
    def validate_cross_field_rules(self) -> Self:
        maximum_tenure = max(0, math.floor(self.driver_age) - 16)
        if self.years_licensed > maximum_tenure:
            raise ValueError(
                "years_licensed cannot exceed floor(driver_age) - 16 " f"(maximum {maximum_tenure})"
            )
        return self

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | CarQuoteInput,
        *,
        domain: DomainSpec | None = None,
    ) -> CarQuoteInput:
        """Validate a mapping without coercion or out-of-domain clipping."""

        try:
            raw = value.as_dict() if isinstance(value, cls) else dict(value)
            result = cls.model_validate(raw, strict=True)
        except (TypeError, ValueError) as exc:
            raise DomainValidationError(f"Invalid car quote input: {exc}") from exc
        if domain is not None:
            domain.validate(result)
        return result

    def as_dict(self) -> dict[str, Any]:
        """Return fields in the canonical, stable v1 order."""

        raw = self.model_dump(mode="python")
        return {name: raw[name] for name in FIELD_ORDER}


@dataclass(frozen=True)
class NumericDomain:
    """Resolved bounds for one numeric field."""

    low: float
    high: float
    integer: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high) or self.low >= self.high:
            raise DomainValidationError("numeric domain bounds must be finite with low < high")
        if self.integer and (not self.low.is_integer() or not self.high.is_integer()):
            raise DomainValidationError("integer domain bounds must be integral")


@dataclass(frozen=True)
class DomainSpec:
    """A fully resolved, artifact-stable domain snapshot."""

    numeric: dict[str, NumericDomain]
    categorical: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if set(self.numeric) != set(CONTINUOUS_FIELDS) | set(INTEGER_FIELDS):
            raise DomainValidationError("domain numeric fields do not match the v1 schema")
        if set(self.categorical) != set(CATEGORICAL_FIELDS):
            raise DomainValidationError("domain categorical fields do not match the v1 schema")
        for name in INTEGER_FIELDS:
            if not self.numeric[name].integer:
                raise DomainValidationError(f"{name} must use an integer domain")
        for name in CONTINUOUS_FIELDS:
            if self.numeric[name].integer:
                raise DomainValidationError(f"{name} must use a continuous domain")
        for name, levels in self.categorical.items():
            if not levels or len(levels) != len(set(levels)):
                raise DomainValidationError(f"{name} must have unique categorical levels")

        age_low = self.numeric["driver_age"].low
        licence_low = self.numeric["years_licensed"].low
        if licence_low > max(0, math.floor(age_low) - 16):
            raise DomainValidationError(
                "domain is infeasible: years_licensed.low exceeds the tenure "
                "allowed at driver_age.low"
            )

    @classmethod
    def default(cls) -> DomainSpec:
        bounds = default_numeric_bounds()
        return cls(
            numeric={
                name: NumericDomain(
                    low=float(bounds[name][0]),
                    high=float(bounds[name][1]),
                    integer=name in INTEGER_FIELDS,
                )
                for name in (*CONTINUOUS_FIELDS, *INTEGER_FIELDS)
            },
            categorical=dict(DEFAULT_CATEGORIES),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DomainSpec:
        try:
            if not isinstance(value, Mapping):
                raise TypeError("domain snapshot must be an object")
            expected_keys = {"schema_version", "numeric", "categorical"}
            if set(value) != expected_keys:
                raise ValueError("domain snapshot has missing or unknown fields")
            schema_version = value["schema_version"]
            if type(schema_version) is not int or schema_version != 1:
                raise ValueError(f"unsupported domain schema version {schema_version!r}")
            raw_numeric = value["numeric"]
            raw_categorical = value["categorical"]
            if not isinstance(raw_numeric, Mapping) or not isinstance(raw_categorical, Mapping):
                raise TypeError("numeric and categorical must be objects")
            numeric: dict[str, NumericDomain] = {}
            for name, item in raw_numeric.items():
                if not isinstance(name, str) or not isinstance(item, Mapping):
                    raise TypeError("numeric domain entries must be named objects")
                if set(item) != {"low", "high", "integer"}:
                    raise ValueError(f"numeric domain entry {name!r} has invalid fields")
                integer = item["integer"]
                if type(integer) is not bool:
                    raise TypeError(f"numeric domain entry {name!r}.integer must be a boolean")
                numeric[name] = NumericDomain(
                    low=float(item["low"]),
                    high=float(item["high"]),
                    integer=integer,
                )
            categorical: dict[str, tuple[str, ...]] = {}
            for name, levels in raw_categorical.items():
                if not isinstance(name, str) or not isinstance(levels, (list, tuple)):
                    raise TypeError("categorical domain entries must be named arrays")
                if any(not isinstance(level, str) for level in levels):
                    raise TypeError(f"categorical domain entry {name!r} must contain strings")
                categorical[name] = tuple(levels)
            return cls(numeric=numeric, categorical=categorical)
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainValidationError(f"Invalid domain snapshot: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "numeric": {
                name: {
                    "low": item.low,
                    "high": item.high,
                    "integer": item.integer,
                }
                for name, item in self.numeric.items()
            },
            "categorical": {name: list(levels) for name, levels in self.categorical.items()},
        }

    def validate(self, row: CarQuoteInput) -> None:
        values = row.as_dict()
        for name, bounds in self.numeric.items():
            raw = values[name]
            number = float(raw)
            if number < bounds.low or number > bounds.high:
                raise DomainValidationError(
                    f"{name}={raw!r} is outside the configured domain "
                    f"[{bounds.low:g}, {bounds.high:g}]"
                )
            if bounds.integer and not number.is_integer():
                raise DomainValidationError(f"{name} must be an integer")
        for name, levels in self.categorical.items():
            if values[name] not in levels:
                raise DomainValidationError(
                    f"{name}={values[name]!r} is outside the configured levels {list(levels)!r}"
                )
        maximum_tenure = max(0, math.floor(float(values["driver_age"])) - 16)
        if int(values["years_licensed"]) > maximum_tenure:
            raise DomainValidationError(
                "years_licensed cannot exceed floor(driver_age) - 16 " f"(maximum {maximum_tenure})"
            )

    def sample_lhs(self, n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Generate internal Latin-hypercube samples and canonicalize constraints."""

        if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
            raise ValueError("n must be a non-negative integer")
        count = int(n)
        if count == 0:
            return []

        dimensions = len(CONTINUOUS_FIELDS)
        cut = np.linspace(0.0, 1.0, count + 1)
        offsets = rng.uniform(size=(count, dimensions))
        points = cut[:-1, None] + (cut[1:] - cut[:-1])[:, None] * offsets
        for column in range(dimensions):
            rng.shuffle(points[:, column])

        rows: list[dict[str, Any]] = []
        for index in range(count):
            row: dict[str, Any] = {}
            for column, name in enumerate(CONTINUOUS_FIELDS):
                bounds = self.numeric[name]
                row[name] = bounds.low + points[index, column] * (bounds.high - bounds.low)
            for name in INTEGER_FIELDS:
                bounds = self.numeric[name]
                row[name] = int(rng.integers(int(bounds.low), int(bounds.high) + 1))
            for name in CATEGORICAL_FIELDS:
                levels = self.categorical[name]
                row[name] = levels[int(rng.integers(0, len(levels)))]
            rows.append(canonicalize_sample(row, self))
        return rows

    def sample_random(self, n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Generate independent internal uniform samples."""

        if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
            raise ValueError("n must be a non-negative integer")
        rows: list[dict[str, Any]] = []
        for _ in range(int(n)):
            row: dict[str, Any] = {}
            for name in CONTINUOUS_FIELDS:
                bounds = self.numeric[name]
                row[name] = float(rng.uniform(bounds.low, bounds.high))
            for name in INTEGER_FIELDS:
                bounds = self.numeric[name]
                row[name] = int(rng.integers(int(bounds.low), int(bounds.high) + 1))
            for name in CATEGORICAL_FIELDS:
                levels = self.categorical[name]
                row[name] = levels[int(rng.integers(0, len(levels)))]
            rows.append(canonicalize_sample(row, self))
        return rows


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise DomainValidationError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DomainValidationError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise DomainValidationError(f"{field} must be a finite number")
    return number


def canonicalize_sample(row: Mapping[str, Any], domain: DomainSpec) -> dict[str, Any]:
    """Canonicalize an internally sampled point.

    This function deliberately clips and rounds generated points. Public
    inference paths must use :meth:`CarQuoteInput.from_mapping` instead.
    """

    missing = sorted(set(FIELD_ORDER) - set(row))
    unknown = sorted(set(row) - set(FIELD_ORDER))
    if missing or unknown:
        raise DomainValidationError(
            f"sample fields do not match schema; missing={missing}, unknown={unknown}"
        )

    result: dict[str, Any] = {}
    for name in CONTINUOUS_FIELDS:
        bounds = domain.numeric[name]
        number = _finite_number(row[name], name)
        result[name] = float(np.clip(number, bounds.low, bounds.high))
    for name in INTEGER_FIELDS:
        bounds = domain.numeric[name]
        number = _finite_number(row[name], name)
        result[name] = int(np.clip(round(number), int(bounds.low), int(bounds.high)))
    for name in CATEGORICAL_FIELDS:
        levels = domain.categorical[name]
        value = row[name]
        if value not in levels:
            raise DomainValidationError(f"{name} must be one of {list(levels)!r}")
        result[name] = value

    maximum_tenure = max(0, math.floor(float(result["driver_age"])) - 16)
    licence_bounds = domain.numeric["years_licensed"]
    feasible_maximum = min(int(licence_bounds.high), maximum_tenure)
    if feasible_maximum < int(licence_bounds.low):
        raise DomainValidationError("sample is infeasible for years_licensed")
    result["years_licensed"] = int(
        np.clip(
            int(result["years_licensed"]),
            int(licence_bounds.low),
            feasible_maximum,
        )
    )

    # A final strict pass is intentional: internal canonicalization must never
    # permit an invalid provider payload.
    validated = CarQuoteInput.from_mapping(result, domain=domain)
    return validated.as_dict()


def row_key(row: Mapping[str, Any] | CarQuoteInput) -> str:
    """Return a stable SHA-256 key without exposing the quote payload."""

    values = row.as_dict() if isinstance(row, CarQuoteInput) else dict(row)
    canonical = json.dumps(
        {name: values[name] for name in FIELD_ORDER},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rows_json_schema() -> dict[str, Any]:
    """Return the generated public input JSON Schema."""

    schema = CarQuoteInput.model_json_schema()
    schema["$id"] = "https://pricing-function-mapper.local/schema/car-quote-input-v1.json"
    schema["title"] = "CarQuoteInput v1"
    return schema
