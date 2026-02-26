from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from pricing_mapper.config import MapperConfig
from pricing_mapper.domain import DomainSpec, canonicalize_comp_car_input
from pricing_mapper.encoding import get_encoder
from pricing_mapper.models import HGB_AVAILABLE, BootstrappedRF, MonotoneHGBWrapper
from pricing_mapper.utils import (
    binary_search_breakpoint,
    jitter_around,
    min_dist2_to_train,
    pick_unique,
    propose_pool,
    stable_key,
    summarize_metrics,
    top_k_desc_idx,
)

STATE_SCHEMA_VERSION = 2


@dataclass
class RunStats:
    samples: int
    budget: int
    mean: float
    std: float
    monotone_enabled: bool
    elapsed_seconds: float
    profile_seconds: dict[str, float]


class ActiveQuoteMapper:
    def __init__(
        self,
        domain: DomainSpec,
        quote_fn: Callable[[dict[str, Any]], float],
        cfg: MapperConfig,
        logger: logging.Logger | None = None,
    ):
        self.domain = domain
        self.quote_fn = quote_fn
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.rng = np.random.default_rng(cfg.seed)

        self.x_rows: list[dict[str, Any]] = []
        self.y_vals: list[float] = []
        self.cache: dict[str, float] = {}

        self.encoder = get_encoder(domain)
        self.profile_seconds: dict[str, float] = {
            "fit": 0.0,
            "pool_generate": 0.0,
            "predict_pool": 0.0,
            "distance": 0.0,
            "cv_residuals": 0.0,
            "local_scoring": 0.0,
            "breakpoint_search": 0.0,
            "proposal_total": 0.0,
            "run_total": 0.0,
        }

        self.rf = BootstrappedRF(
            n_models=cfg.rf_n_models,
            seed=cfg.seed,
            n_estimators=cfg.rf_n_estimators,
            n_jobs=cfg.rf_n_jobs,
        )

        self.use_monotone = False
        self.hgb: MonotoneHGBWrapper | None = None
        self.monotone_vars = {
            "vehicle_value": +1,
            "postcode_risk": +1,
            "theft_risk": +1,
            "claims_5y": +1,
            "convictions_5y": +1,
            "excess": -1,
            "annual_km": +1,
        }
        if cfg.use_monotone_if_available and HGB_AVAILABLE:
            try:
                monotonic_cst: list[int] = []
                for cont_var in self.domain.continuous:
                    monotonic_cst.append(int(self.monotone_vars.get(cont_var.name, 0)))
                for iv in self.domain.integers:
                    monotonic_cst.append(int(self.monotone_vars.get(iv.name, 0)))
                for cat_var in self.domain.categorical:
                    monotonic_cst.extend([0] * len(cat_var.levels))
                self.hgb = MonotoneHGBWrapper(monotonic_cst=monotonic_cst, seed=cfg.seed)
                self.use_monotone = True
            except Exception as exc:
                self.logger.warning("Monotone model unavailable: %s", exc)

        self.var_bounds: dict[str, tuple[float, float]] = {}
        for cv in self.domain.continuous:
            self.var_bounds[cv.name] = (cv.low, cv.high)
        for iv in self.domain.integers:
            self.var_bounds[iv.name] = (float(iv.low), float(iv.high))

        self._last_fit_n = 0
        self._fitted = False

    def _add_profile(self, key: str, elapsed: float) -> None:
        self.profile_seconds[key] = self.profile_seconds.get(key, 0.0) + float(elapsed)

    def query(self, row: dict[str, Any]) -> float:
        row = canonicalize_comp_car_input(row)
        key = stable_key(row)
        if key in self.cache:
            return self.cache[key]
        value = float(self.quote_fn(row))
        self.cache[key] = value
        return value

    def add_samples(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            row = canonicalize_comp_car_input(row)
            key = stable_key(row)
            value = self.cache.get(key)
            if value is None:
                value = self.query(row)
            self.x_rows.append(row)
            self.y_vals.append(value)

    def _fit_models(self) -> tuple[np.ndarray, np.ndarray]:
        x_train = self.encoder.encode(self.x_rows)
        y_train = np.asarray(self.y_vals, dtype=float)

        t0 = perf_counter()
        self.rf.fit(x_train, y_train)
        if self.use_monotone and self.hgb is not None:
            self.hgb.fit(x_train, y_train)
        fit_elapsed = perf_counter() - t0

        self._add_profile("fit", fit_elapsed)
        self._last_fit_n = len(self.x_rows)
        self._fitted = True
        self.logger.info("Model fit complete on %d samples in %.2fs", len(self.x_rows), fit_elapsed)
        return x_train, y_train

    def _ensure_models(self, force_refit: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted or force_refit:
            return self._fit_models()
        x_train = self.encoder.encode(self.x_rows)
        y_train = np.asarray(self.y_vals, dtype=float)
        return x_train, y_train

    def _predict(self, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        x_eval = self.encoder.encode(rows)
        mu_rf, sigma_rf = self.rf.predict_mean_std(x_eval)
        if self.use_monotone and self.hgb is not None:
            return self.hgb.predict(x_eval), sigma_rf
        return mu_rf, sigma_rf

    def _cv_residuals(self, x_train: np.ndarray, y_train: np.ndarray, k: int = 5) -> np.ndarray:
        if len(y_train) < max(30, k * 10):
            return np.zeros_like(y_train)

        n = len(y_train)
        max_n = min(self.cfg.cv_subsample_max, n)
        if max_n < n:
            subsample_idx = self.rng.choice(n, size=max_n, replace=False)
            x_use = x_train[subsample_idx]
            y_use = y_train[subsample_idx]
        else:
            subsample_idx = np.arange(n)
            x_use = x_train
            y_use = y_train

        folds = min(k, max(2, len(y_use) // 20))
        splitter = KFold(n_splits=folds, shuffle=True, random_state=123)
        preds = np.zeros_like(y_use, dtype=float)

        for tr, te in splitter.split(x_use):
            model = RandomForestRegressor(
                n_estimators=self.cfg.rf_n_estimators,
                min_samples_leaf=2,
                random_state=777,
                n_jobs=self.cfg.rf_n_jobs,
            )
            model.fit(x_use[tr], y_use[tr])
            preds[te] = model.predict(x_use[te])

        out = np.zeros_like(y_train)
        out[subsample_idx] = y_use - preds
        return out

    def propose_next_batch(
        self,
        batch_size: int,
        force_refit: bool = False,
    ) -> list[dict[str, Any]]:
        t_propose = perf_counter()
        x_train, y_train = self._ensure_models(force_refit=force_refit)
        used = {stable_key(x) for x in self.x_rows}

        t0 = perf_counter()
        pool = propose_pool(self.domain, n=self.cfg.pool_size, rng=self.rng)
        self._add_profile("pool_generate", perf_counter() - t0)

        t0 = perf_counter()
        _, sigma = self._predict(pool)
        self._add_profile("predict_pool", perf_counter() - t0)

        # Lightweight update heuristic for non-refit rounds: slightly widen uncertainty.
        stale = max(0, len(self.x_rows) - self._last_fit_n)
        if stale > 0 and not force_refit:
            sigma = sigma * (1.0 + 0.01 * min(10, stale))

        x_pool = self.encoder.encode(pool)

        t0 = perf_counter()
        dmin = min_dist2_to_train(x_pool, x_train, backend=self.cfg.distance_backend)
        self._add_profile("distance", perf_counter() - t0)

        score_unc = sigma
        score_bnd = sigma * np.log1p(dmin)

        t0 = perf_counter()
        resid = self._cv_residuals(x_train, y_train, k=5)
        self._add_profile("cv_residuals", perf_counter() - t0)

        top_resid_idx = top_k_desc_idx(np.abs(resid), max(5, min(25, len(resid) // 10)))
        local_points: list[dict[str, Any]] = []
        for idx in top_resid_idx:
            local_points.extend(jitter_around(self.x_rows[int(idx)], self.domain, self.rng, n=25))

        t0 = perf_counter()
        if local_points:
            _, sigma_local = self._predict(local_points)
            x_local = self.encoder.encode(local_points)
            dmin_local = min_dist2_to_train(x_local, x_train, backend=self.cfg.distance_backend)
            score_err_local = sigma_local * np.log1p(dmin_local)
        else:
            score_err_local = np.array([], dtype=float)
        self._add_profile("local_scoring", perf_counter() - t0)

        t0 = perf_counter()
        bp_points: list[dict[str, Any]] = []
        anchors = [pool[i] for i in top_k_desc_idx(score_bnd, 40)]
        for var in self.cfg.breakpoint_vars:
            if len(bp_points) >= max(2, int(batch_size * self.cfg.acquisition_mix[3]) * 2):
                break
            bounds = self.var_bounds.get(var)
            if bounds is None:
                continue
            low, high = bounds
            self.rng.shuffle(anchors)
            for anchor in anchors[:2]:
                bp_points.extend(
                    binary_search_breakpoint(
                        x_base=anchor,
                        var_name=var,
                        low=low,
                        high=high,
                        predict_fn=lambda rows: self._predict(rows)[0],
                        max_queries=5,
                        threshold=45.0,
                    )
                )
        self._add_profile("breakpoint_search", perf_counter() - t0)

        n_unc = int(round(batch_size * self.cfg.acquisition_mix[0]))
        n_bnd = int(round(batch_size * self.cfg.acquisition_mix[1]))
        n_err = int(round(batch_size * self.cfg.acquisition_mix[2]))
        n_bp = batch_size - n_unc - n_bnd - n_err

        picks: list[dict[str, Any]] = []

        if n_unc > 0:
            idx = top_k_desc_idx(score_unc, n_unc * 10)
            picks.extend(pick_unique([pool[i] for i in idx], used, n_unc))

        if len(picks) < batch_size and n_bnd > 0:
            idx = top_k_desc_idx(score_bnd, n_bnd * 10)
            picks.extend(pick_unique([pool[i] for i in idx], used, n_bnd))

        if len(picks) < batch_size and n_err > 0 and local_points:
            idx = top_k_desc_idx(score_err_local, n_err * 10)
            picks.extend(pick_unique([local_points[i] for i in idx], used, n_err))

        if len(picks) < batch_size and n_bp > 0 and bp_points:
            self.rng.shuffle(bp_points)
            picks.extend(pick_unique(bp_points, used, n_bp))

        if len(picks) < batch_size:
            idx = top_k_desc_idx(score_bnd, (batch_size - len(picks)) * 10)
            picks.extend(pick_unique([pool[i] for i in idx], used, batch_size - len(picks)))

        elapsed = perf_counter() - t_propose
        self._add_profile("proposal_total", elapsed)
        self.logger.info("Candidate proposal built in %.2fs", elapsed)
        return picks[:batch_size]

    def save_state(self, path: str | Path) -> None:
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "x_rows": self.x_rows,
            "y_vals": self.y_vals,
            "cache": self.cache,
            "rng_state": self.rng.bit_generator.state,
            "cfg": asdict(self.cfg),
            "profile_seconds": self.profile_seconds,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")))
        tmp.replace(target)

    def _migrate_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        version = int(payload.get("schema_version", 1))
        if version == 1:
            payload = dict(payload)
            payload["schema_version"] = 2
            payload.setdefault("profile_seconds", {})
            return payload
        if version == STATE_SCHEMA_VERSION:
            return payload
        raise ValueError(f"Unsupported state schema version: {version}")

    def load_state(self, path: str | Path) -> None:
        target = Path(path)
        try:
            payload = json.loads(target.read_text())
            payload = self._migrate_state(payload)
        except Exception as exc:
            raise ValueError(f"Failed to load state from {target}: {exc}") from exc

        required = ("x_rows", "y_vals", "cache", "rng_state")
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(f"State file missing required keys: {missing}")

        self.x_rows = payload["x_rows"]
        self.y_vals = [float(v) for v in payload["y_vals"]]
        self.cache = {k: float(v) for k, v in payload["cache"].items()}
        self.rng.bit_generator.state = payload["rng_state"]

        loaded_profile = payload.get("profile_seconds", {})
        if isinstance(loaded_profile, dict):
            for key, val in loaded_profile.items():
                self.profile_seconds[key] = float(val)

        self._fitted = False
        self._last_fit_n = 0

    def run(self) -> tuple[pd.DataFrame, RunStats]:
        start = perf_counter()

        if self.cfg.resume and Path(self.cfg.state_path).exists():
            self.load_state(self.cfg.state_path)
            self.logger.info(
                "Resumed from %s with %d samples",
                self.cfg.state_path,
                len(self.x_rows),
            )

        if not self.x_rows:
            init = propose_pool(self.domain, n=min(self.cfg.init_n, self.cfg.budget), rng=self.rng)
            self.add_samples(init)
            self.logger.info("Initialized with %d samples", len(init))

        batch_count = 0
        while len(self.x_rows) < self.cfg.budget:
            need = min(self.cfg.batch_size, self.cfg.budget - len(self.x_rows))
            should_refit = (batch_count % max(1, self.cfg.refit_every_batches)) == 0

            next_batch = self.propose_next_batch(batch_size=need, force_refit=should_refit)
            self.add_samples(next_batch)
            batch_count += 1

            y = np.asarray(self.y_vals, dtype=float)
            mean, std = summarize_metrics(y)
            self.logger.info(
                "Samples: %d/%d | mean=%.2f | std=%.2f | monotone=%s",
                len(self.x_rows),
                self.cfg.budget,
                mean,
                std,
                "ON" if self.use_monotone else "OFF",
            )

            if self.cfg.checkpoint_every_batches > 0 and (
                batch_count % self.cfg.checkpoint_every_batches == 0
            ):
                self.save_state(self.cfg.state_path)

        df = pd.DataFrame(self.x_rows)
        df["premium"] = np.asarray(self.y_vals, dtype=float)

        mean, std = summarize_metrics(df["premium"].to_numpy(dtype=float))
        total_elapsed = float(perf_counter() - start)
        self.profile_seconds["run_total"] = total_elapsed

        stats = RunStats(
            samples=len(df),
            budget=self.cfg.budget,
            mean=mean,
            std=std,
            monotone_enabled=self.use_monotone,
            elapsed_seconds=total_elapsed,
            profile_seconds=dict(self.profile_seconds),
        )
        return df, stats
