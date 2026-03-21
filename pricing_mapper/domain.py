from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
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
    levels: list[Any]


@dataclass
class DomainSpec:
    continuous: list[ContinuousVar]
    integers: list[IntegerVar]
    categorical: list[CategoricalVar]

    def sample_lhs(self, n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Latin hypercube for continuous vars; uniform for integers/categoricals."""
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
                x[cv.name] = rng.choice(cv.levels)
            xs.append(x)
        return xs


def _current_utc_year() -> int:
    return datetime.now(UTC).year


def canonicalize_comp_car_input(
    x: dict[str, Any],
    domain: DomainSpec | None = None,
) -> dict[str, Any]:
    """Apply product constraints and bound clipping."""
    x = dict(x)

    current_year = _current_utc_year()
    if domain is None:
        cont_bounds: dict[str, tuple[float, float]] = {
            "driver_age": (17.0, 90.0),
            "postcode_risk": (0.0, 1.0),
            "vehicle_value": (2000.0, 200000.0),
            "theft_risk": (0.0, 1.0),
        }
        int_bounds: dict[str, tuple[int, int]] = {
            "years_licensed": (0, 70),
            "vehicle_year": (1998, current_year),
            "annual_km": (1000, 60000),
            "claims_5y": (0, 6),
            "convictions_5y": (0, 6),
            "excess": (0, 5000),
        }
    else:
        cont_bounds = {v.name: (float(v.low), float(v.high)) for v in domain.continuous}
        int_bounds = {v.name: (int(v.low), int(v.high)) for v in domain.integers}

    age_low, age_high = cont_bounds.get("driver_age", (17.0, 90.0))
    x["driver_age"] = float(np.clip(float(x["driver_age"]), age_low, age_high))
    max_years_licensed = max(0, int(x["driver_age"]) - 16)
    yl_low, yl_high = int_bounds.get("years_licensed", (0, 70))
    x["years_licensed"] = int(
        np.clip(int(x["years_licensed"]), yl_low, min(yl_high, max_years_licensed))
    )

    year_low, year_high = int_bounds.get("vehicle_year", (1998, current_year))
    x["vehicle_year"] = int(np.clip(int(x["vehicle_year"]), year_low, year_high))
    value_low, value_high = cont_bounds.get("vehicle_value", (2000.0, 200000.0))
    x["vehicle_value"] = float(np.clip(float(x["vehicle_value"]), value_low, value_high))
    km_low, km_high = int_bounds.get("annual_km", (1000, 60000))
    x["annual_km"] = int(np.clip(int(x["annual_km"]), km_low, km_high))
    claims_low, claims_high = int_bounds.get("claims_5y", (0, 6))
    x["claims_5y"] = int(np.clip(int(x["claims_5y"]), claims_low, claims_high))
    conv_low, conv_high = int_bounds.get("convictions_5y", (0, 6))
    x["convictions_5y"] = int(np.clip(int(x["convictions_5y"]), conv_low, conv_high))
    post_low, post_high = cont_bounds.get("postcode_risk", (0.0, 1.0))
    x["postcode_risk"] = float(np.clip(float(x["postcode_risk"]), post_low, post_high))
    theft_low, theft_high = cont_bounds.get("theft_risk", (0.0, 1.0))
    x["theft_risk"] = float(np.clip(float(x["theft_risk"]), theft_low, theft_high))
    excess_low, excess_high = int_bounds.get("excess", (0, 5000))
    x["excess"] = int(np.clip(int(x["excess"]), excess_low, excess_high))

    return x


def build_comp_car_domain(overrides: dict[str, Any] | None = None) -> DomainSpec:
    """Build domain with optional per-variable bound/levels overrides from config."""
    overrides = overrides or {}
    current_year = _current_utc_year()

    cont_defaults = {
        "driver_age": (17.0, 90.0),
        "postcode_risk": (0.0, 1.0),
        "vehicle_value": (2000.0, 200000.0),
        "theft_risk": (0.0, 1.0),
    }
    int_defaults = {
        "years_licensed": (0, 70),
        "vehicle_year": (1998, current_year),
        "annual_km": (1000, 60000),
        "claims_5y": (0, 6),
        "convictions_5y": (0, 6),
        "excess": (0, 5000),
    }
    cat_defaults = {
        "usage": ["private", "commute", "business"],
        "parking": ["garage", "driveway", "street"],
        "hire_car": ["none", "basic", "premium"],
        "windscreen": ["no", "yes"],
        "rating": ["market", "agreed"],
    }

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
        continuous=[
            ContinuousVar("driver_age", *cont_defaults["driver_age"]),
            ContinuousVar("postcode_risk", *cont_defaults["postcode_risk"]),
            ContinuousVar("vehicle_value", *cont_defaults["vehicle_value"]),
            ContinuousVar("theft_risk", *cont_defaults["theft_risk"]),
        ],
        integers=[
            IntegerVar("years_licensed", *int_defaults["years_licensed"]),
            IntegerVar("vehicle_year", *int_defaults["vehicle_year"]),
            IntegerVar("annual_km", *int_defaults["annual_km"]),
            IntegerVar("claims_5y", *int_defaults["claims_5y"]),
            IntegerVar("convictions_5y", *int_defaults["convictions_5y"]),
            IntegerVar("excess", *int_defaults["excess"]),
        ],
        categorical=[
            CategoricalVar("usage", cat_defaults["usage"]),
            CategoricalVar("parking", cat_defaults["parking"]),
            CategoricalVar("hire_car", cat_defaults["hire_car"]),
            CategoricalVar("windscreen", cat_defaults["windscreen"]),
            CategoricalVar("rating", cat_defaults["rating"]),
        ],
    )
