from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from copy import deepcopy
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
    allocate_integer_counts,
    binary_search_breakpoint,
    jitter_around,
    min_dist2_to_train,
    pick_unique,
    propose_pool,
    propose_segment_pool,
    stable_key,
    summarize_metrics,
    top_k_desc_idx,
)

STATE_SCHEMA_VERSION = 3
RESUME_CFG_COMPAT_KEYS: tuple[str, ...] = (
    "seed",
    "quote_provider",
    "domain_overrides",
    "use_monotone_if_available",
    "rf_n_models",
    "rf_n_estimators",
)


@dataclass
class RunStats:
    samples: int
    budget: int
    mean: float
    std: float
    monotone_enabled: bool
    completed_budget: bool
    early_stopped: bool
    stop_reason: str | None
    elapsed_seconds: float
    profile_seconds: dict[str, float]


@dataclass(frozen=True)
class _ModelFitState:
    last_fit_n: int
    rf_rng_state: Mapping[str, Any]
    last_fit_rf_rng_state: Mapping[str, Any] | None


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
                monotonic_cst = [
                    int(self.monotone_vars.get(cont_var.name, 0))
                    for cont_var in self.domain.continuous
                ]
                monotonic_cst.extend(
                    int(self.monotone_vars.get(int_var.name, 0)) for int_var in self.domain.integers
                )
                for cat_var in self.domain.categorical:
                    monotonic_cst.extend([0] * len(cat_var.levels))
                self.hgb = MonotoneHGBWrapper(monotonic_cst=monotonic_cst, seed=cfg.seed)
                self.use_monotone = True
            except (RuntimeError, TypeError, ValueError) as exc:
                self.logger.warning("Monotone model unavailable: %s", exc)

        self.var_bounds: dict[str, tuple[float, float]] = {}
        for cv in self.domain.continuous:
            self.var_bounds[cv.name] = (cv.low, cv.high)
        for iv in self.domain.integers:
            self.var_bounds[iv.name] = (float(iv.low), float(iv.high))

        self._last_fit_n = 0
        self._last_fit_rf_rng_state: Mapping[str, Any] | None = None
        self._fitted = False
        self.batch_count = 0
        self.best_fit_mae = float("inf")
        self.stale_batches = 0
        self._state_early_stopped = False
        self._state_stop_reason: str | None = None
        self._state_model_fit_snapshot: _ModelFitState | None = None

    def _add_profile(self, key: str, elapsed: float) -> None:
        self.profile_seconds[key] = self.profile_seconds.get(key, 0.0) + float(elapsed)

    @staticmethod
    def _normalize01(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.array([], dtype=float)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo:
            return np.zeros_like(arr, dtype=float)
        return (arr - lo) / (hi - lo)

    def row_in_segment(self, row: dict[str, Any]) -> bool:
        constraints = self.cfg.segment_constraints
        if not constraints:
            return True

        for var, raw_rule in constraints.items():
            val = row[var]
            rule = raw_rule if isinstance(raw_rule, dict) else {"eq": raw_rule}

            if "eq" in rule and val != rule["eq"]:
                return False
            if "in" in rule and val not in set(rule["in"]):
                return False
            if "min" in rule and float(val) < float(rule["min"]):
                return False
            if "max" in rule and float(val) > float(rule["max"]):
                return False
        return True

    def _segment_mask(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([self.row_in_segment(row) for row in rows], dtype=bool)

    def _focus_stage_active(self) -> bool:
        if not self.cfg.staged_mapping_enabled:
            return False
        cutoff = max(1, round(self.cfg.budget * self.cfg.staged_stage1_fraction))
        return len(self.x_rows) >= cutoff

    def _build_candidate_pool(self, x_train: np.ndarray) -> list[dict[str, Any]]:
        pool = propose_pool(self.domain, n=self.cfg.pool_size, rng=self.rng)

        if self._focus_stage_active():
            _, sigma0 = self._predict(pool)
            x_pool0 = self.encoder.encode(pool)
            dmin0 = min_dist2_to_train(x_pool0, x_train, backend=self.cfg.distance_backend)
            score0 = sigma0 * np.log1p(dmin0)
            n_anchor = max(5, min(30, self.cfg.pool_size // 100))
            focus_points: list[dict[str, Any]] = []
            for idx in top_k_desc_idx(score0, n_anchor):
                focus_points.extend(
                    jitter_around(
                        pool[int(idx)],
                        self.domain,
                        self.rng,
                        n=self.cfg.staged_focus_jitter_per_anchor,
                    )
                )
            if focus_points:
                pool.extend(focus_points)

        if self.cfg.segment_focus_enabled and self.cfg.segment_constraints:
            tries = 0
            seg_count = int(self._segment_mask(pool).sum())
            while (
                seg_count < self.cfg.segment_min_candidates
                and tries < self.cfg.segment_pool_max_tries
            ):
                remaining = self.cfg.segment_min_candidates - seg_count
                extra_n = min(self.cfg.pool_size, max(50, remaining * 2))
                pool.extend(
                    propose_segment_pool(
                        self.domain,
                        n=extra_n,
                        rng=self.rng,
                        constraints=self.cfg.segment_constraints,
                    )
                )
                seg_count = int(self._segment_mask(pool).sum())
                tries += 1

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in pool:
            key = stable_key(row)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    def query(self, row: dict[str, Any]) -> float:
        row = canonicalize_comp_car_input(row, self.domain)
        key = stable_key(row)
        if key in self.cache:
            return self.cache[key]
        raw_value = self.quote_fn(row)
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Quote provider must return a numeric value.") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError("Quote provider must return a finite, non-negative value.")
        self.cache[key] = value
        return value

    def add_samples(self, rows: list[dict[str, Any]]) -> int:
        """Add unique canonical rows and return the number added."""
        added = 0
        existing = {stable_key(row) for row in self.x_rows}
        for raw_row in rows:
            row = canonicalize_comp_car_input(raw_row, self.domain)
            key = stable_key(row)
            if key in existing:
                continue
            value = self.cache.get(key)
            if value is None:
                value = self.query(row)
            self.x_rows.append(row)
            self.y_vals.append(value)
            existing.add(key)
            added += 1
        return added

    def _fit_models(self, fit_n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        fit_n = len(self.x_rows) if fit_n is None else fit_n
        if fit_n <= 0 or fit_n > len(self.x_rows):
            raise ValueError("fit_n must be between 1 and the number of samples")
        x_train = self.encoder.encode(self.x_rows[:fit_n])
        y_train = np.asarray(self.y_vals[:fit_n], dtype=float)

        t0 = perf_counter()
        self._last_fit_rf_rng_state = deepcopy(self.rf.rng.bit_generator.state)
        self.rf.fit(x_train, y_train)
        if self.use_monotone and self.hgb is not None:
            self.hgb.fit(x_train, y_train)
        fit_elapsed = perf_counter() - t0

        self._add_profile("fit", fit_elapsed)
        self._last_fit_n = fit_n
        self._fitted = True
        self.logger.info("Model fit complete on %d samples in %.2fs", fit_n, fit_elapsed)
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
        if max_n < max(30, k * 10):
            return np.zeros_like(y_train)
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
        n_unc, n_bnd, n_err, n_bp = allocate_integer_counts(
            batch_size,
            self.cfg.acquisition_mix,
        )
        x_train, y_train = self._ensure_models(force_refit=force_refit)
        used = {stable_key(x) for x in self.x_rows}

        t0 = perf_counter()
        pool = self._build_candidate_pool(x_train)
        self._add_profile("pool_generate", perf_counter() - t0)
        if not pool:
            raise RuntimeError("Candidate generation produced an empty pool.")

        t0 = perf_counter()
        mu, sigma = self._predict(pool)
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

        if self.cfg.segment_focus_enabled and self.cfg.segment_constraints:
            seg_mask = self._segment_mask(pool)
            if seg_mask.any():
                mu_norm = self._normalize01(mu)
                sigma_norm = self._normalize01(sigma)
                segment_priority = seg_mask.astype(float) * (
                    1.0
                    + self.cfg.segment_target_weight * mu_norm
                    + self.cfg.segment_sigma_weight * sigma_norm
                )
                score_unc = score_unc * segment_priority
                score_bnd = score_bnd * segment_priority

        local_points: list[dict[str, Any]] = []
        score_err_local = np.array([], dtype=float)
        if n_err > 0:
            t0 = perf_counter()
            resid = self._cv_residuals(x_train, y_train, k=5)
            self._add_profile("cv_residuals", perf_counter() - t0)

            resid_score = np.abs(resid)
            if self.cfg.segment_focus_enabled and self.cfg.segment_constraints:
                train_seg_mask = self._segment_mask(self.x_rows)
                resid_score = resid_score * (1.0 + 0.5 * train_seg_mask.astype(float))
            top_resid_idx = top_k_desc_idx(
                resid_score,
                max(5, min(25, len(resid) // 10)),
            )
            for idx in top_resid_idx:
                local_points.extend(
                    jitter_around(
                        self.x_rows[int(idx)],
                        self.domain,
                        self.rng,
                        n=25,
                    )
                )

            t0 = perf_counter()
            if local_points:
                _, sigma_local = self._predict(local_points)
                x_local = self.encoder.encode(local_points)
                dmin_local = min_dist2_to_train(
                    x_local,
                    x_train,
                    backend=self.cfg.distance_backend,
                )
                score_err_local = sigma_local * np.log1p(dmin_local)
                if self.cfg.segment_focus_enabled and self.cfg.segment_constraints:
                    mu_local, _ = self._predict(local_points)
                    seg_local = self._segment_mask(local_points)
                    if seg_local.any():
                        mu_local_norm = self._normalize01(mu_local)
                        sigma_local_norm = self._normalize01(sigma_local)
                        local_priority = seg_local.astype(float) * (
                            1.0
                            + self.cfg.segment_target_weight * mu_local_norm
                            + self.cfg.segment_sigma_weight * sigma_local_norm
                        )
                        score_err_local = score_err_local * local_priority
            self._add_profile("local_scoring", perf_counter() - t0)

        bp_points: list[dict[str, Any]] = []
        if n_bp > 0:
            t0 = perf_counter()
            anchors = [pool[i] for i in top_k_desc_idx(score_bnd, 40)]
            if self.cfg.segment_focus_enabled and self.cfg.segment_constraints:
                seg_anchors = [row for row in anchors if self.row_in_segment(row)]
                if seg_anchors:
                    anchors = seg_anchors
            for var in self.cfg.breakpoint_vars:
                if len(bp_points) >= max(2, n_bp * 2):
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
                            domain=self.domain,
                            max_queries=5,
                            threshold=45.0,
                        )
                    )
            if self.cfg.segment_focus_enabled and self.cfg.segment_constraints:
                bp_points = [row for row in bp_points if self.row_in_segment(row)]
            self._add_profile("breakpoint_search", perf_counter() - t0)

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
        model_fit_state = self._state_model_fit_snapshot or _ModelFitState(
            last_fit_n=self._last_fit_n,
            rf_rng_state=deepcopy(self.rf.rng.bit_generator.state),
            last_fit_rf_rng_state=deepcopy(self._last_fit_rf_rng_state),
        )
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "x_rows": self.x_rows,
            "y_vals": self.y_vals,
            "cache": self.cache,
            "rng_state": self.rng.bit_generator.state,
            "rf_rng_state": model_fit_state.rf_rng_state,
            "last_fit_rf_rng_state": model_fit_state.last_fit_rf_rng_state,
            "last_fit_n": model_fit_state.last_fit_n,
            "monotone_enabled": self.use_monotone,
            "batch_count": self.batch_count,
            "best_fit_mae": (self.best_fit_mae if np.isfinite(self.best_fit_mae) else None),
            "stale_batches": self.stale_batches,
            "early_stopped": self._state_early_stopped,
            "stop_reason": self._state_stop_reason,
            "cfg": asdict(self.cfg),
            "domain": asdict(self.domain),
            "profile_seconds": self.profile_seconds,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(target)

    def _migrate_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        version = payload.get("schema_version", 1)
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("State schema_version must be an integer.")
        if version == 1:
            payload = dict(payload)
            payload.setdefault("profile_seconds", {})
            version = 2
        if version == 2:
            payload = dict(payload)
            payload["schema_version"] = 3
            payload.setdefault("rf_rng_state", None)
            payload.setdefault("last_fit_rf_rng_state", None)
            payload.setdefault("last_fit_n", 0)
            payload.setdefault("monotone_enabled", None)
            payload.setdefault("batch_count", 0)
            payload.setdefault("best_fit_mae", None)
            payload.setdefault("stale_batches", 0)
            payload.setdefault("early_stopped", False)
            payload.setdefault("stop_reason", None)
            payload.setdefault("domain", None)
            return payload
        if version == STATE_SCHEMA_VERSION:
            return payload
        raise ValueError(f"Unsupported state schema version: {version}")

    def load_state(self, path: str | Path) -> None:
        target = Path(path)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("State payload must be a JSON object.")
            payload = self._migrate_state(payload)
        except Exception as exc:
            raise ValueError(f"Failed to load state from {target}: {exc}") from exc

        required = (
            "x_rows",
            "y_vals",
            "cache",
            "rng_state",
            "rf_rng_state",
            "last_fit_rf_rng_state",
            "last_fit_n",
            "monotone_enabled",
            "batch_count",
            "best_fit_mae",
            "stale_batches",
            "early_stopped",
            "stop_reason",
            "cfg",
            "domain",
        )
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(f"State file missing required keys: {missing}")

        saved_cfg = payload.get("cfg")
        if not isinstance(saved_cfg, dict):
            raise ValueError("State file config must be an object.")
        current_cfg = asdict(self.cfg)
        mismatched = [
            key for key in RESUME_CFG_COMPAT_KEYS if saved_cfg.get(key) != current_cfg.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"State/config mismatch for resume. Incompatible keys: {', '.join(mismatched)}"
            )
        saved_domain = payload["domain"]
        current_domain = json.loads(json.dumps(asdict(self.domain)))
        if saved_domain is not None and saved_domain != current_domain:
            raise ValueError("State/domain mismatch for resume.")

        raw_rows = payload["x_rows"]
        raw_values = payload["y_vals"]
        raw_cache = payload["cache"]
        raw_rng_state = payload["rng_state"]
        raw_rf_rng_state = payload["rf_rng_state"]
        raw_last_fit_rf_rng_state = payload["last_fit_rf_rng_state"]
        if not isinstance(raw_rows, list):
            raise ValueError("State x_rows must be a list.")
        if not isinstance(raw_values, list):
            raise ValueError("State y_vals must be a list.")
        if len(raw_rows) != len(raw_values):
            raise ValueError("State x_rows and y_vals lengths do not match.")
        if len(raw_rows) > self.cfg.budget:
            raise ValueError(
                "State contains more samples than the configured budget; "
                "increase budget before resuming."
            )
        if not isinstance(raw_cache, dict):
            raise ValueError("State cache must be an object.")
        if not isinstance(raw_rng_state, dict):
            raise ValueError("State rng_state must be an object.")

        rows: list[dict[str, Any]] = []
        values: list[float] = []
        seen: set[str] = set()
        for index, (raw_row, raw_value) in enumerate(zip(raw_rows, raw_values, strict=True)):
            if not isinstance(raw_row, dict):
                raise ValueError(f"State x_rows[{index}] must be an object.")
            row = canonicalize_comp_car_input(raw_row, self.domain)
            key = stable_key(row)
            if key in seen:
                raise ValueError("State x_rows contains duplicate canonical rows.")
            seen.add(key)
            try:
                value = float(raw_value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(f"State y_vals[{index}] must be numeric.") from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"State y_vals[{index}] must be finite and non-negative.")
            rows.append(row)
            values.append(value)

        cache: dict[str, float] = {}
        for key, raw_value in raw_cache.items():
            if not isinstance(key, str):
                raise ValueError("State cache keys must be strings.")
            try:
                value = float(raw_value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError("State cache values must be numeric.") from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError("State cache values must be finite and non-negative.")
            cache[key] = value

        for row, expected in zip(rows, values, strict=True):
            cached = cache.get(stable_key(row))
            if cached is None or cached != expected:
                raise ValueError("State cache does not match sampled rows and values.")

        test_rng = np.random.default_rng()
        try:
            test_rng.bit_generator.state = raw_rng_state
        except (TypeError, ValueError) as exc:
            raise ValueError("State rng_state is invalid.") from exc

        last_fit_n = payload["last_fit_n"]
        batch_count = payload["batch_count"]
        stale_batches = payload["stale_batches"]
        for name, value in (
            ("last_fit_n", last_fit_n),
            ("batch_count", batch_count),
            ("stale_batches", stale_batches),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"State {name} must be a non-negative integer.")
        if last_fit_n > len(rows):
            raise ValueError("State last_fit_n exceeds the number of sampled rows.")
        if batch_count > len(rows):
            raise ValueError("State batch_count exceeds the number of sampled rows.")
        if stale_batches > batch_count:
            raise ValueError("State stale_batches exceeds batch_count.")

        state_early_stopped = payload["early_stopped"]
        state_stop_reason = payload["stop_reason"]
        if not isinstance(state_early_stopped, bool):
            raise ValueError("State early_stopped must be a boolean.")
        if state_stop_reason is not None and not isinstance(state_stop_reason, str):
            raise ValueError("State stop_reason must be a string or null.")
        if state_early_stopped and not state_stop_reason:
            raise ValueError("State stop_reason is required when early_stopped is true.")

        best_fit_mae_raw = payload["best_fit_mae"]
        if best_fit_mae_raw is None:
            best_fit_mae = float("inf")
        else:
            try:
                best_fit_mae = float(best_fit_mae_raw)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError("State best_fit_mae must be numeric or null.") from exc
            if not np.isfinite(best_fit_mae) or best_fit_mae < 0:
                raise ValueError("State best_fit_mae must be finite and non-negative.")

        for name, rng_state in (
            ("rf_rng_state", raw_rf_rng_state),
            ("last_fit_rf_rng_state", raw_last_fit_rf_rng_state),
        ):
            if rng_state is None:
                continue
            if not isinstance(rng_state, dict):
                raise ValueError(f"State {name} must be an object or null.")
            test_rf_rng = np.random.default_rng()
            try:
                test_rf_rng.bit_generator.state = rng_state
            except (TypeError, ValueError) as exc:
                raise ValueError(f"State {name} is invalid.") from exc
        if last_fit_n > 0 and (raw_rf_rng_state is None or raw_last_fit_rf_rng_state is None):
            raise ValueError("State is missing RF RNG data for its last model fit.")

        monotone_enabled = payload["monotone_enabled"]
        if monotone_enabled is not None and not isinstance(monotone_enabled, bool):
            raise ValueError("State monotone_enabled must be a boolean or null.")
        if monotone_enabled is not None and monotone_enabled != self.use_monotone:
            raise ValueError(
                "State/config mismatch for resume: monotone model availability changed."
            )

        loaded_profile = payload.get("profile_seconds", {})
        if not isinstance(loaded_profile, dict):
            raise ValueError("State profile_seconds must be an object.")
        profile: dict[str, float] = {}
        for key, raw_value in loaded_profile.items():
            if not isinstance(key, str):
                raise ValueError("State profile_seconds keys must be strings.")
            try:
                value = float(raw_value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError("State profile_seconds values must be numeric.") from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError("State profile_seconds values must be finite and non-negative.")
            profile[key] = value

        self.x_rows = rows
        self.y_vals = values
        self.cache = cache
        self.rng.bit_generator.state = raw_rng_state
        self.profile_seconds.update(profile)
        self.batch_count = batch_count
        self.best_fit_mae = best_fit_mae
        self.stale_batches = stale_batches
        self._state_early_stopped = state_early_stopped
        self._state_stop_reason = state_stop_reason
        self._state_model_fit_snapshot = None

        if last_fit_n > 0:
            self.rf.rng.bit_generator.state = raw_last_fit_rf_rng_state
            self._fit_models(fit_n=last_fit_n)
            if self.rf.rng.bit_generator.state != raw_rf_rng_state:
                raise ValueError("State RF RNG data is inconsistent with the reconstructed model.")
        else:
            self._fitted = False
            self._last_fit_n = 0
            self._last_fit_rf_rng_state = None
            if raw_rf_rng_state is not None:
                self.rf.rng.bit_generator.state = raw_rf_rng_state

    def run(self) -> tuple[pd.DataFrame, RunStats]:
        start = perf_counter()
        early_stopped = False
        stop_reason: str | None = None
        self._state_early_stopped = False
        self._state_stop_reason = None
        self._state_model_fit_snapshot = None

        if self.cfg.resume:
            state_path = Path(self.cfg.state_path)
            if not state_path.is_file():
                raise FileNotFoundError(f"Cannot resume: state file does not exist: {state_path}")
            self.load_state(state_path)
            self.logger.info(
                "Resumed from %s with %d samples",
                self.cfg.state_path,
                len(self.x_rows),
            )
            if self._state_early_stopped and self.cfg.early_stop_patience_batches > 0:
                early_stopped = True
                stop_reason = self._state_stop_reason
                self.logger.info("Checkpoint had already stopped early: %s", stop_reason)
            else:
                self._state_early_stopped = False
                self._state_stop_reason = None

        if not self.x_rows:
            init = propose_pool(self.domain, n=min(self.cfg.init_n, self.cfg.budget), rng=self.rng)
            added = self.add_samples(init)
            if added == 0:
                raise RuntimeError("Initial sampling produced no unique rows.")
            self.logger.info("Initialized with %d samples", added)

        while len(self.x_rows) < self.cfg.budget and not early_stopped:
            need = min(self.cfg.batch_size, self.cfg.budget - len(self.x_rows))
            should_refit = (self.batch_count % max(1, self.cfg.refit_every_batches)) == 0

            next_batch = self.propose_next_batch(batch_size=need, force_refit=should_refit)
            if not next_batch:
                raise RuntimeError(
                    "Unable to propose a new unique candidate; the configured domain "
                    "or pool may be too small for the requested budget."
                )
            added = self.add_samples(next_batch)
            if added == 0:
                raise RuntimeError(
                    "Candidate proposal made no progress; the configured domain "
                    "or pool may be too small for the requested budget."
                )
            self.batch_count += 1

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

            if self.cfg.early_stop_patience_batches > 0:
                _, y_fit = self._ensure_models(force_refit=True)
                mu_fit, _ = self._predict(self.x_rows)
                fit_mae = float(np.mean(np.abs(mu_fit - y_fit)))
                if not np.isfinite(self.best_fit_mae):
                    self.best_fit_mae = fit_mae
                    self.stale_batches = 0
                else:
                    rel_improve = (self.best_fit_mae - fit_mae) / max(
                        self.best_fit_mae,
                        1e-9,
                    )
                    if rel_improve >= self.cfg.early_stop_min_rel_improvement:
                        self.best_fit_mae = fit_mae
                        self.stale_batches = 0
                    else:
                        self.stale_batches += 1

                if (
                    self.batch_count >= self.cfg.early_stop_min_batches
                    and self.stale_batches >= self.cfg.early_stop_patience_batches
                ):
                    early_stopped = True
                    stop_reason = (
                        "early_stop_plateau:"
                        f" mae={fit_mae:.4f}, best={self.best_fit_mae:.4f}, "
                        f"stale_batches={self.stale_batches}"
                    )
                    self.logger.info("Stopping early: %s", stop_reason)
                    self._state_early_stopped = True
                    self._state_stop_reason = stop_reason

            if self.cfg.checkpoint_every_batches > 0 and (
                self.batch_count % self.cfg.checkpoint_every_batches == 0
            ):
                self.save_state(self.cfg.state_path)

            if early_stopped:
                break

        self._state_model_fit_snapshot = _ModelFitState(
            last_fit_n=self._last_fit_n,
            rf_rng_state=deepcopy(self.rf.rng.bit_generator.state),
            last_fit_rf_rng_state=deepcopy(self._last_fit_rf_rng_state),
        )
        if not self._fitted or self._last_fit_n != len(self.x_rows):
            self._fit_models()

        df = pd.DataFrame(self.x_rows)
        df["premium"] = np.asarray(self.y_vals, dtype=float)

        mean, std = summarize_metrics(df["premium"].to_numpy(dtype=float))
        current_elapsed = float(perf_counter() - start)
        total_elapsed = self.profile_seconds.get("run_total", 0.0) + current_elapsed
        self.profile_seconds["run_total"] = total_elapsed

        stats = RunStats(
            samples=len(df),
            budget=self.cfg.budget,
            mean=mean,
            std=std,
            monotone_enabled=self.use_monotone,
            completed_budget=len(df) >= self.cfg.budget,
            early_stopped=early_stopped,
            stop_reason=stop_reason,
            elapsed_seconds=total_elapsed,
            profile_seconds=dict(self.profile_seconds),
        )
        if self.cfg.checkpoint_every_batches > 0 or self.cfg.resume:
            self.save_state(self.cfg.state_path)
        return df, stats
