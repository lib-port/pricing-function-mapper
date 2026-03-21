from __future__ import annotations

import json
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


def min_dist2_to_train(
    x_pool: np.ndarray,
    x_train: np.ndarray,
    backend: str = "brute",
    chunk: int = 2000,
) -> np.ndarray:
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
            x[iv.name] = int(round(iv.low + z2 * span))

        for cv in domain.categorical:
            if rng.uniform() < p_cat_flip:
                x[cv.name] = rng.choice(cv.levels)

        xs.append(canonicalize_comp_car_input(x, domain))
    return xs


def binary_search_breakpoint(
    x_base: dict[str, Any],
    var_name: str,
    low: float,
    high: float,
    predict_fn,
    domain: DomainSpec | None = None,
    max_queries: int = 6,
    threshold: float = 40.0,
) -> list[dict[str, Any]]:
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
        pred_ab = predict_fn([pa, pm, pb])

        left = abs(float(pred_ab[1] - pred_ab[0]))
        right = abs(float(pred_ab[2] - pred_ab[1]))
        if left >= right:
            b = mid
        else:
            a = mid

    return points


def summarize_metrics(y: np.ndarray) -> tuple[float, float]:
    return float(y.mean()), float(y.std())
