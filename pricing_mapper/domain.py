from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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


def canonicalize_comp_car_input(x: dict[str, Any]) -> dict[str, Any]:
    """Apply product constraints and bound clipping."""
    x = dict(x)

    x["driver_age"] = float(np.clip(x["driver_age"], 17, 90))
    x["years_licensed"] = int(np.clip(x["years_licensed"], 0, int(x["driver_age"]) - 16))
    x["vehicle_year"] = int(np.clip(x["vehicle_year"], 1998, 2026))
    x["vehicle_value"] = float(np.clip(x["vehicle_value"], 2000, 200000))
    x["annual_km"] = int(np.clip(x["annual_km"], 1000, 60000))
    x["claims_5y"] = int(np.clip(x["claims_5y"], 0, 6))
    x["convictions_5y"] = int(np.clip(x["convictions_5y"], 0, 6))
    x["postcode_risk"] = float(np.clip(x["postcode_risk"], 0.0, 1.0))
    x["theft_risk"] = float(np.clip(x["theft_risk"], 0.0, 1.0))
    x["excess"] = int(np.clip(x["excess"], 0, 5000))

    return x


def build_comp_car_domain(overrides: dict[str, Any] | None = None) -> DomainSpec:
    """Build domain with optional per-variable bound/levels overrides from config."""
    overrides = overrides or {}

    cont_defaults = {
        "driver_age": (17.0, 90.0),
        "postcode_risk": (0.0, 1.0),
        "vehicle_value": (2000.0, 200000.0),
        "theft_risk": (0.0, 1.0),
    }
    int_defaults = {
        "years_licensed": (0, 70),
        "vehicle_year": (1998, 2026),
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
