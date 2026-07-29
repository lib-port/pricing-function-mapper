from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral, Real
from typing import Any

import numpy as np

DOMAIN_CONTINUOUS_NAMES: tuple[str, ...] = (
    "driver_age",
    "postcode_risk",
    "vehicle_value",
    "theft_risk",
)
DOMAIN_INTEGER_NAMES: tuple[str, ...] = (
    "years_licensed",
    "vehicle_year",
    "annual_km",
    "claims_5y",
    "convictions_5y",
    "excess",
)
DOMAIN_CATEGORICAL_NAMES: tuple[str, ...] = (
    "usage",
    "parking",
    "hire_car",
    "windscreen",
    "rating",
)

CONTINUOUS_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "driver_age": (17.0, 90.0),
    "postcode_risk": (0.0, 1.0),
    "vehicle_value": (2000.0, 200000.0),
    "theft_risk": (0.0, 1.0),
}
INTEGER_DEFAULT_BOUNDS: dict[str, tuple[int, int]] = {
    "years_licensed": (0, 70),
    "annual_km": (1000, 60000),
    "claims_5y": (0, 6),
    "convictions_5y": (0, 6),
    "excess": (0, 5000),
}
CATEGORICAL_DEFAULT_LEVELS: dict[str, tuple[Any, ...]] = {
    "usage": ("private", "commute", "business"),
    "parking": ("garage", "driveway", "street"),
    "hire_car": ("none", "basic", "premium"),
    "windscreen": ("no", "yes"),
    "rating": ("market", "agreed"),
}


@dataclass(frozen=True)
class ContinuousVar:
    name: str
    low: float
    high: float


@dataclass(frozen=True)
class IntegerVar:
    name: str
    low: int
    high: int


@dataclass(frozen=True)
class CategoricalVar:
    name: str
    levels: tuple[Any, ...]


@dataclass(frozen=True)
class DomainSpec:
    continuous: tuple[ContinuousVar, ...]
    integers: tuple[IntegerVar, ...]
    categorical: tuple[CategoricalVar, ...]

    def sample_lhs(self, n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Latin hypercube for continuous vars; uniform for integers/categoricals."""
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
            raise ValueError("n must be a non-negative integer")
        n = int(n)

        xs: list[dict[str, Any]] = []
        cont = self.continuous
        d = len(cont)

        if d > 0:
            cut = np.linspace(0, 1, n + 1)
            u = rng.uniform(size=(n, d))
            a = cut[:n]
            b = cut[1 : n + 1]
            pts = a[:, None] + (b - a)[:, None] * u
            for j in range(d):
                rng.shuffle(pts[:, j])
        else:
            pts = np.zeros((n, 0))

        for i in range(n):
            x: dict[str, Any] = {}
            for j, v in enumerate(cont):
                x[v.name] = float(v.low + pts[i, j] * (v.high - v.low))
            for iv in self.integers:
                x[iv.name] = int(rng.integers(iv.low, iv.high + 1))
            for cv in self.categorical:
                level_index = int(rng.integers(0, len(cv.levels)))
                x[cv.name] = cv.levels[level_index]
            xs.append(x)
        return xs


def _current_utc_year() -> int:
    return datetime.now(UTC).year


def _integer_default_bounds() -> dict[str, tuple[int, int]]:
    return {
        "years_licensed": INTEGER_DEFAULT_BOUNDS["years_licensed"],
        "vehicle_year": (1998, _current_utc_year()),
        "annual_km": INTEGER_DEFAULT_BOUNDS["annual_km"],
        "claims_5y": INTEGER_DEFAULT_BOUNDS["claims_5y"],
        "convictions_5y": INTEGER_DEFAULT_BOUNDS["convictions_5y"],
        "excess": INTEGER_DEFAULT_BOUNDS["excess"],
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _integer_bound(value: Any, label: str) -> int:
    number = _finite_number(value, label)
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer.")
    return int(number)


def validate_domain_overrides(overrides: dict[str, Any]) -> None:
    """Validate domain override structure, bounds, and categorical levels."""
    if not isinstance(overrides, dict):
        raise ValueError("domain_overrides must be a dictionary.")

    allowed_buckets = {"continuous", "integers", "categorical"}
    unknown_buckets = set(overrides) - allowed_buckets
    if unknown_buckets:
        raise ValueError(f"Unknown domain_overrides keys: {sorted(unknown_buckets)}")

    for bucket in allowed_buckets:
        if bucket in overrides and not isinstance(overrides[bucket], dict):
            raise ValueError(f"domain_overrides.{bucket} must be a dictionary.")

    for name, cfg in overrides.get("continuous", {}).items():
        if name not in DOMAIN_CONTINUOUS_NAMES:
            raise ValueError(f"Unknown continuous override '{name}'.")
        if not isinstance(cfg, dict) or set(cfg) != {"low", "high"}:
            raise ValueError(f"continuous override '{name}' must provide only low/high.")
        low = _finite_number(cfg["low"], f"continuous override '{name}'.low")
        high = _finite_number(cfg["high"], f"continuous override '{name}'.high")
        if low >= high:
            raise ValueError(f"continuous override '{name}' has low >= high.")

    for name, cfg in overrides.get("integers", {}).items():
        if name not in DOMAIN_INTEGER_NAMES:
            raise ValueError(f"Unknown integers override '{name}'.")
        if not isinstance(cfg, dict) or set(cfg) != {"low", "high"}:
            raise ValueError(f"integers override '{name}' must provide only low/high.")
        low = _integer_bound(cfg["low"], f"integers override '{name}'.low")
        high = _integer_bound(cfg["high"], f"integers override '{name}'.high")
        if low >= high:
            raise ValueError(f"integers override '{name}' has low >= high.")

    scalar_types = (str, int, float, bool, type(None))
    for name, levels in overrides.get("categorical", {}).items():
        if name not in DOMAIN_CATEGORICAL_NAMES:
            raise ValueError(f"Unknown categorical override '{name}'.")
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError(
                f"categorical override '{name}' must be a list with at least 2 levels."
            )
        if any(not isinstance(level, scalar_types) for level in levels):
            raise ValueError(f"categorical override '{name}' levels must be JSON scalar values.")
        if any(isinstance(level, float) and not np.isfinite(level) for level in levels):
            raise ValueError(f"categorical override '{name}' levels must be finite when numeric.")
        if len(set(levels)) != len(levels):
            raise ValueError(f"categorical override '{name}' contains duplicate levels.")

    continuous_bounds = dict(CONTINUOUS_DEFAULT_BOUNDS)
    integer_bounds = _integer_default_bounds()
    for name, cfg in overrides.get("continuous", {}).items():
        continuous_bounds[name] = (float(cfg["low"]), float(cfg["high"]))
    for name, cfg in overrides.get("integers", {}).items():
        integer_bounds[name] = (int(cfg["low"]), int(cfg["high"]))

    minimum_age = continuous_bounds["driver_age"][0]
    minimum_years_licensed = integer_bounds["years_licensed"][0]
    maximum_feasible_at_minimum_age = max(0, int(minimum_age) - 16)
    if minimum_years_licensed > maximum_feasible_at_minimum_age:
        raise ValueError(
            "domain_overrides are infeasible: years_licensed.low exceeds the "
            "maximum permitted at driver_age.low."
        )


def _coerce_finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"'{name}' must be numeric, not boolean.")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' must be numeric.") from exc
    if not np.isfinite(number):
        raise ValueError(f"'{name}' must be finite.")
    return number


def _coerce_finite_int(value: Any, name: str) -> int:
    number = _coerce_finite_float(value, name)
    return round(number)


def canonicalize_comp_car_input(
    x: dict[str, Any],
    domain: DomainSpec | None = None,
) -> dict[str, Any]:
    """Validate a complete input row, then apply constraints and bound clipping."""
    if not isinstance(x, dict):
        raise ValueError("Input row must be a dictionary.")
    x = dict(x)

    current_year = _current_utc_year()
    if domain is None:
        cont_bounds = dict(CONTINUOUS_DEFAULT_BOUNDS)
        int_bounds = _integer_default_bounds()
        categorical_levels: dict[str, list[Any]] = {
            name: list(levels) for name, levels in CATEGORICAL_DEFAULT_LEVELS.items()
        }
    else:
        cont_bounds = {v.name: (float(v.low), float(v.high)) for v in domain.continuous}
        int_bounds = {v.name: (int(v.low), int(v.high)) for v in domain.integers}
        categorical_levels = {v.name: list(v.levels) for v in domain.categorical}

    expected_fields = set(cont_bounds) | set(int_bounds) | set(categorical_levels)
    missing = sorted(expected_fields - set(x))
    unknown = sorted(set(x) - expected_fields)
    if missing:
        raise ValueError(f"Input row is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"Input row contains unknown fields: {unknown}")

    age_low, age_high = cont_bounds.get("driver_age", (17.0, 90.0))
    age = _coerce_finite_float(x["driver_age"], "driver_age")
    x["driver_age"] = float(np.clip(age, age_low, age_high))
    max_years_licensed = max(0, int(x["driver_age"]) - 16)
    yl_low, yl_high = int_bounds.get("years_licensed", (0, 70))
    yl_max = min(yl_high, max_years_licensed)
    if yl_max < yl_low:
        raise ValueError(
            "Domain is infeasible for this row: years_licensed.low exceeds the "
            "maximum permitted by driver_age."
        )
    years_licensed = _coerce_finite_int(x["years_licensed"], "years_licensed")
    x["years_licensed"] = int(np.clip(years_licensed, yl_low, yl_max))

    year_low, year_high = int_bounds.get("vehicle_year", (1998, current_year))
    vehicle_year = _coerce_finite_int(x["vehicle_year"], "vehicle_year")
    x["vehicle_year"] = int(np.clip(vehicle_year, year_low, year_high))
    value_low, value_high = cont_bounds.get("vehicle_value", (2000.0, 200000.0))
    vehicle_value = _coerce_finite_float(x["vehicle_value"], "vehicle_value")
    x["vehicle_value"] = float(np.clip(vehicle_value, value_low, value_high))
    km_low, km_high = int_bounds.get("annual_km", (1000, 60000))
    annual_km = _coerce_finite_int(x["annual_km"], "annual_km")
    x["annual_km"] = int(np.clip(annual_km, km_low, km_high))
    claims_low, claims_high = int_bounds.get("claims_5y", (0, 6))
    claims = _coerce_finite_int(x["claims_5y"], "claims_5y")
    x["claims_5y"] = int(np.clip(claims, claims_low, claims_high))
    conv_low, conv_high = int_bounds.get("convictions_5y", (0, 6))
    convictions = _coerce_finite_int(x["convictions_5y"], "convictions_5y")
    x["convictions_5y"] = int(np.clip(convictions, conv_low, conv_high))
    post_low, post_high = cont_bounds.get("postcode_risk", (0.0, 1.0))
    postcode_risk = _coerce_finite_float(x["postcode_risk"], "postcode_risk")
    x["postcode_risk"] = float(np.clip(postcode_risk, post_low, post_high))
    theft_low, theft_high = cont_bounds.get("theft_risk", (0.0, 1.0))
    theft_risk = _coerce_finite_float(x["theft_risk"], "theft_risk")
    x["theft_risk"] = float(np.clip(theft_risk, theft_low, theft_high))
    excess_low, excess_high = int_bounds.get("excess", (0, 5000))
    excess = _coerce_finite_int(x["excess"], "excess")
    x["excess"] = int(np.clip(excess, excess_low, excess_high))

    for name, levels in categorical_levels.items():
        if x[name] not in levels:
            raise ValueError(f"'{name}' must be one of {levels!r}; received {x[name]!r}.")

    return x


def build_comp_car_domain(overrides: dict[str, Any] | None = None) -> DomainSpec:
    """Build domain with optional per-variable bound/levels overrides from config."""
    if overrides is None:
        overrides = {}
    validate_domain_overrides(overrides)

    cont_defaults = dict(CONTINUOUS_DEFAULT_BOUNDS)
    int_defaults = _integer_default_bounds()
    cat_defaults = {name: list(levels) for name, levels in CATEGORICAL_DEFAULT_LEVELS.items()}

    for name, cfg in overrides.get("continuous", {}).items():
        if name in cont_defaults:
            cont_defaults[name] = (float(cfg["low"]), float(cfg["high"]))

    for name, cfg in overrides.get("integers", {}).items():
        if name in int_defaults:
            int_defaults[name] = (int(cfg["low"]), int(cfg["high"]))

    for name, levels in overrides.get("categorical", {}).items():
        if name in cat_defaults:
            cat_defaults[name] = list(levels)

    return DomainSpec(
        continuous=(
            ContinuousVar("driver_age", *cont_defaults["driver_age"]),
            ContinuousVar("postcode_risk", *cont_defaults["postcode_risk"]),
            ContinuousVar("vehicle_value", *cont_defaults["vehicle_value"]),
            ContinuousVar("theft_risk", *cont_defaults["theft_risk"]),
        ),
        integers=(
            IntegerVar("years_licensed", *int_defaults["years_licensed"]),
            IntegerVar("vehicle_year", *int_defaults["vehicle_year"]),
            IntegerVar("annual_km", *int_defaults["annual_km"]),
            IntegerVar("claims_5y", *int_defaults["claims_5y"]),
            IntegerVar("convictions_5y", *int_defaults["convictions_5y"]),
            IntegerVar("excess", *int_defaults["excess"]),
        ),
        categorical=(
            CategoricalVar("usage", tuple(cat_defaults["usage"])),
            CategoricalVar("parking", tuple(cat_defaults["parking"])),
            CategoricalVar("hire_car", tuple(cat_defaults["hire_car"])),
            CategoricalVar("windscreen", tuple(cat_defaults["windscreen"])),
            CategoricalVar("rating", tuple(cat_defaults["rating"])),
        ),
    )
