from __future__ import annotations

import importlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np


def mock_comp_car_quote(x: dict[str, Any]) -> float:
    age = x["driver_age"]
    yrs_lic = x["years_licensed"]
    claims = x["claims_5y"]
    convictions = x["convictions_5y"]
    postcode_risk = x["postcode_risk"]
    annual_km = x["annual_km"]
    value = x["vehicle_value"]
    year = x["vehicle_year"]
    theft = x["theft_risk"]
    garage = x["parking"]
    usage = x["usage"]
    excess = x["excess"]
    hire = x["hire_car"]
    wind = x["windscreen"]
    rating = x["rating"]

    base = 300.0 + 0.0105 * value
    base *= 1.0 + 0.62 * postcode_risk

    current_year = datetime.now(UTC).year
    veh_age = max(0, current_year - year)
    base *= 1.0 + 0.010 * min(veh_age, 10) + 0.020 * max(0, veh_age - 10)

    if age < 21:
        base *= 1.95
    elif age < 25:
        base *= 1.52
    elif age < 35:
        base *= 1.18
    elif age < 60:
        base *= 1.00
    else:
        base *= 1.10

    base *= 1.0 - 0.06 * np.tanh((yrs_lic - 3) / 6.0)
    base *= (1.0 + 0.22 * claims + 0.28 * convictions) * (1.0 + 0.10 * postcode_risk * (claims > 0))

    park_mult = {"garage": 0.92, "driveway": 1.00, "street": 1.13}[garage]
    base *= (1.0 + 0.55 * theft) * park_mult

    usage_mult = {"private": 0.98, "commute": 1.00, "business": 1.13}[usage]
    base *= usage_mult
    base *= 1.0 + 0.11 * np.tanh((annual_km - 12000) / 12000.0)

    hire_mult = {"none": 1.00, "basic": 1.04, "premium": 1.08}[hire]
    base *= hire_mult
    if wind == "yes":
        base *= 1.02
    if rating == "agreed":
        base *= 1.015

    base *= 1.0 - 0.16 * np.tanh((excess - 600) / 700.0)

    premium = max(base, 260.0)
    return float(np.round(premium, 2))


def load_quote_fn(provider: str | None) -> Callable[[dict[str, Any]], float]:
    """Load quote function from `module:function` path, or default mock provider."""
    if not provider:
        return mock_comp_car_quote

    if ":" not in provider:
        raise ValueError("quote_provider must use 'module:function' format")

    mod_name, fn_name = provider.split(":", 1)
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"quote provider '{provider}' did not resolve to a callable")
    return fn
