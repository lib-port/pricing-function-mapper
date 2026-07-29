from __future__ import annotations

from typing import Any
from weakref import ReferenceType, ref

import numpy as np

from pricing_mapper.domain import DomainSpec


class FeatureEncoder:
    def __init__(self, domain: DomainSpec) -> None:
        self.cont_names = [v.name for v in domain.continuous]
        self.cont_low = np.asarray([v.low for v in domain.continuous], dtype=float)
        self.cont_inv_span = np.asarray(
            [1.0 / (v.high - v.low) if (v.high - v.low) != 0 else 0.0 for v in domain.continuous],
            dtype=float,
        )

        self.int_names = [v.name for v in domain.integers]
        self.int_low = np.asarray([v.low for v in domain.integers], dtype=float)
        self.int_inv_span = np.asarray(
            [1.0 / (v.high - v.low) if (v.high - v.low) != 0 else 0.0 for v in domain.integers],
            dtype=float,
        )

        self.cat_names = [cv.name for cv in domain.categorical]
        self.cat_levels = [list(cv.levels) for cv in domain.categorical]
        self.cat_maps = [{lvl: i for i, lvl in enumerate(lvls)} for lvls in self.cat_levels]

        cols: list[str] = []
        cols.extend(self.cont_names)
        cols.extend(self.int_names)
        for cv_name, lvls in zip(self.cat_names, self.cat_levels, strict=True):
            cols.extend([f"{cv_name}={lvl}" for lvl in lvls])
        self.cols = tuple(cols)

    def encode(self, rows: list[dict[str, Any]]) -> np.ndarray:
        n = len(rows)
        parts: list[np.ndarray] = []

        if self.cont_names:
            a_cont = np.empty((n, len(self.cont_names)), dtype=float)
            for j, name in enumerate(self.cont_names):
                col = np.fromiter((float(x[name]) for x in rows), dtype=float, count=n)
                a_cont[:, j] = (col - self.cont_low[j]) * self.cont_inv_span[j]
            parts.append(a_cont)

        if self.int_names:
            a_int = np.empty((n, len(self.int_names)), dtype=float)
            for j, name in enumerate(self.int_names):
                col = np.fromiter((int(x[name]) for x in rows), dtype=float, count=n)
                a_int[:, j] = (col - self.int_low[j]) * self.int_inv_span[j]
            parts.append(a_int)

        if self.cat_names:
            row_idx = np.arange(n, dtype=np.intp)
            for name, lvls, level_to_idx in zip(
                self.cat_names,
                self.cat_levels,
                self.cat_maps,
                strict=True,
            ):
                try:
                    idx = np.fromiter(
                        (level_to_idx[x[name]] for x in rows),
                        dtype=np.intp,
                        count=n,
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown or missing categorical value for '{name}': {exc}"
                    ) from exc
                one_hot = np.zeros((n, len(lvls)), dtype=float)
                one_hot[row_idx, idx] = 1.0
                parts.append(one_hot)

        if not parts:
            return np.zeros((n, 0), dtype=float)
        return np.concatenate(parts, axis=1)


_ENCODER_CACHE: dict[int, tuple[ReferenceType[DomainSpec], FeatureEncoder]] = {}


def get_encoder(domain: DomainSpec) -> FeatureEncoder:
    key = id(domain)
    cached = _ENCODER_CACHE.get(key)
    if cached is not None and cached[0]() is domain:
        return cached[1]

    def remove_cached_encoder(
        domain_ref: ReferenceType[DomainSpec],
        cache_key: int = key,
    ) -> None:
        current = _ENCODER_CACHE.get(cache_key)
        if current is not None and current[0] is domain_ref:
            _ENCODER_CACHE.pop(cache_key, None)

    encoder = FeatureEncoder(domain)
    domain_ref = ref(domain, remove_cached_encoder)
    _ENCODER_CACHE[key] = (domain_ref, encoder)
    return encoder


def encode_features(domain: DomainSpec, rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    encoder = get_encoder(domain)
    return encoder.encode(rows), list(encoder.cols)
