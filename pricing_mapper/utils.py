from __future__ import annotations

import json
import math
from collections.abc import Callable
from numbers import Integral
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pricing_mapper.domain import DomainSpec, canonicalize_comp_car_input


def stable_key(x: dict[str, Any]) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def top_k_desc_idx(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or arr.size == 0:
        return np.array([], dtype=np.intp)
    k = min(k, arr.size)
    idx = np.argpartition(arr, -k)[-k:]
    return idx[np.argsort(arr[idx])[::-1]]


def allocate_integer_counts(total: int, weights: tuple[float, ...]) -> tuple[int, ...]:
    """Allocate an exact integer total using the largest-remainder method."""
    if isinstance(total, bool) or not isinstance(total, Integral) or total < 0:
        raise ValueError("total must be a non-negative integer")
    total = int(total)
    if not weights:
        raise ValueError("weights cannot be empty")

    weight_array = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0):
        raise ValueError("weights must be finite and non-negative")
    if not np.isclose(float(weight_array.sum()), 1.0):
        raise ValueError("weights must sum to 1.0")

    raw = weight_array * total
    counts = np.floor(raw).astype(int)
    remaining = total - int(counts.sum())
    if remaining > 0:
        fractional = raw - counts
        order = np.argsort(-fractional, kind="stable")
        counts[order[:remaining]] += 1
    return tuple(int(value) for value in counts)


def min_dist2_to_train(
    x_pool: np.ndarray,
    x_train: np.ndarray,
    backend: str = "knn",
    chunk: int = 2000,
) -> np.ndarray:
    if backend not in {"brute", "knn"}:
        raise ValueError("backend must be one of: brute, knn")
    if chunk <= 0:
        raise ValueError("chunk must be > 0")
    if x_train.shape[0] == 0:
        raise ValueError("x_train must contain at least one row")
    if x_pool.shape[0] == 0:
        return np.empty((0,), dtype=float)

    if backend == "knn":
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(x_train)
        dist, _ = nn.kneighbors(x_pool, return_distance=True)
        return np.square(dist[:, 0])

    dmin = np.full((x_pool.shape[0],), np.inf, dtype=float)
    for i in range(0, x_pool.shape[0], chunk):
        a = x_pool[i : i + chunk]
        dist2 = ((a[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2)
        dmin[i : i + chunk] = dist2.min(axis=1)
    return dmin


def pick_unique(
    candidates: list[dict[str, Any]],
    used_keys: set[str],
    k: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        key = stable_key(candidate)
        if key not in used_keys:
            used_keys.add(key)
            out.append(candidate)
            if len(out) >= k:
                break
    return out


def propose_pool(domain: DomainSpec, n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    return [canonicalize_comp_car_input(x, domain) for x in domain.sample_lhs(n=n, rng=rng)]


def propose_segment_pool(
    domain: DomainSpec,
    n: int,
    rng: np.random.Generator,
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    """Sample candidates directly inside independently constrained ranges."""
    rows = domain.sample_lhs(n=n, rng=rng)

    years_var = next(var for var in domain.integers if var.name == "years_licensed")
    years_rule_raw = constraints.get("years_licensed")
    minimum_years = years_var.low
    if years_rule_raw is not None:
        years_rule = years_rule_raw if isinstance(years_rule_raw, dict) else {"eq": years_rule_raw}
        minimum_value = (
            years_rule["eq"] if "eq" in years_rule else years_rule.get("min", years_var.low)
        )
        minimum_years = math.ceil(float(minimum_value))
    minimum_driver_age = 16 + minimum_years

    for row in rows:
        for cont_var in domain.continuous:
            raw_rule = constraints.get(cont_var.name)
            adjust_driver_age = cont_var.name == "driver_age" and minimum_driver_age > cont_var.low
            if raw_rule is None and not adjust_driver_age:
                continue
            rule = (
                raw_rule
                if isinstance(raw_rule, dict)
                else ({"eq": raw_rule} if raw_rule is not None else {})
            )
            if "eq" in rule:
                row[cont_var.name] = float(rule["eq"])
            else:
                low = max(float(cont_var.low), float(rule.get("min", cont_var.low)))
                high = min(float(cont_var.high), float(rule.get("max", cont_var.high)))
                if cont_var.name == "driver_age":
                    low = max(low, float(minimum_driver_age))
                row[cont_var.name] = float(rng.uniform(low, high))

        for int_var in domain.integers:
            raw_rule = constraints.get(int_var.name)
            if raw_rule is None:
                continue
            rule = raw_rule if isinstance(raw_rule, dict) else {"eq": raw_rule}
            if "eq" in rule:
                row[int_var.name] = int(rule["eq"])
            else:
                low = max(
                    int_var.low,
                    math.ceil(float(rule.get("min", int_var.low))),
                )
                high = min(
                    int_var.high,
                    math.floor(float(rule.get("max", int_var.high))),
                )
                if int_var.name == "years_licensed":
                    high = min(high, max(0, int(float(row["driver_age"])) - 16))
                row[int_var.name] = int(rng.integers(low, high + 1))

        for cat_var in domain.categorical:
            raw_rule = constraints.get(cat_var.name)
            if raw_rule is None:
                continue
            rule = raw_rule if isinstance(raw_rule, dict) else {"eq": raw_rule}
            if "eq" in rule:
                row[cat_var.name] = rule["eq"]
            elif "in" in rule:
                levels = rule["in"]
                row[cat_var.name] = levels[int(rng.integers(0, len(levels)))]

    return [canonicalize_comp_car_input(row, domain) for row in rows]


def jitter_around(
    x0: dict[str, Any],
    domain: DomainSpec,
    rng: np.random.Generator,
    n: int,
    cont_sigma: float = 0.08,
    int_sigma: float = 0.10,
    p_cat_flip: float = 0.15,
) -> list[dict[str, Any]]:
    xs: list[dict[str, Any]] = []
    for _ in range(n):
        x = dict(x0)

        for v in domain.continuous:
            span = v.high - v.low
            z = (float(x[v.name]) - v.low) / span
            z2 = float(np.clip(z + rng.normal(0, cont_sigma), 0.0, 1.0))
            x[v.name] = v.low + z2 * span

        for iv in domain.integers:
            span = iv.high - iv.low
            z = (int(x[iv.name]) - iv.low) / max(1, span)
            z2 = float(np.clip(z + rng.normal(0, int_sigma), 0.0, 1.0))
            x[iv.name] = round(iv.low + z2 * span)

        for cv in domain.categorical:
            if rng.uniform() < p_cat_flip:
                level_index = int(rng.integers(0, len(cv.levels)))
                x[cv.name] = cv.levels[level_index]

        xs.append(canonicalize_comp_car_input(x, domain))
    return xs


def binary_search_breakpoint(
    x_base: dict[str, Any],
    var_name: str,
    low: float,
    high: float,
    predict_fn: Callable[[list[dict[str, Any]]], np.ndarray],
    domain: DomainSpec | None = None,
    max_queries: int = 6,
    threshold: float = 40.0,
) -> list[dict[str, Any]]:
    if max_queries < 0:
        raise ValueError("max_queries must be >= 0")
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    if high < low:
        raise ValueError("high must be >= low")

    x_low = dict(x_base)
    x_high = dict(x_base)
    x_low[var_name] = low
    x_high[var_name] = high
    x_low = canonicalize_comp_car_input(x_low, domain)
    x_high = canonicalize_comp_car_input(x_high, domain)

    pred = predict_fn([x_low, x_high])
    if abs(float(pred[1] - pred[0])) < threshold:
        return []

    points: list[dict[str, Any]] = []
    a, b = low, high
    for _ in range(max_queries):
        mid = (a + b) / 2.0
        x_mid = dict(x_base)
        x_mid[var_name] = mid
        x_mid = canonicalize_comp_car_input(x_mid, domain)
        points.append(x_mid)

        pa = dict(x_base)
        pm = dict(x_base)
        pb = dict(x_base)
        pa[var_name] = a
        pm[var_name] = mid
        pb[var_name] = b
        pa = canonicalize_comp_car_input(pa, domain)
        pm = canonicalize_comp_car_input(pm, domain)
        pb = canonicalize_comp_car_input(pb, domain)
        pred_ab = predict_fn([pa, pm, pb])

        left = abs(float(pred_ab[1] - pred_ab[0]))
        right = abs(float(pred_ab[2] - pred_ab[1]))
        if left >= right:
            b = mid
        else:
            a = mid

    return points


def summarize_metrics(y: np.ndarray) -> tuple[float, float]:
    if y.size == 0:
        raise ValueError("Cannot summarize an empty array")
    return float(y.mean()), float(y.std())
