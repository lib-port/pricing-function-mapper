"""Held-out regression metrics, bootstrap bounds, and split-conformal intervals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pricing_mapper.config import EarlyStoppingConfig, EvaluationConfig


def _aligned_arrays(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("actual and predicted values must be aligned one-dimensional arrays")
    if y_true.size == 0:
        raise ValueError("evaluation requires at least one row")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("evaluation values must be finite")
    return y_true, y_pred


def point_metrics(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> dict[str, float | None]:
    """Compute production regression metrics without training-set shortcuts."""

    y_true, y_pred = _aligned_arrays(actual, predicted)
    errors = y_pred - y_true
    absolute = np.abs(errors)
    denominator = float(np.sum(np.abs(y_true)))
    centered = y_true - float(np.mean(y_true))
    total_variance = float(np.sum(centered * centered))
    residual_variance = float(np.sum(errors * errors))
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "wape": None if denominator == 0 else float(np.sum(absolute) / denominator),
        "r2": None if total_variance == 0 else float(1.0 - residual_variance / total_variance),
        "p95_absolute_error": float(np.quantile(absolute, 0.95, method="higher")),
        "max_absolute_error": float(np.max(absolute)),
    }


def bootstrap_intervals(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap confidence intervals for MAE and RMSE."""

    y_true, y_pred = _aligned_arrays(actual, predicted)
    base = point_metrics(y_true, y_pred)
    if iterations == 0:
        exact: dict[str, dict[str, float]] = {}
        for name in ("mae", "rmse"):
            value = base[name]
            if value is None:
                raise AssertionError(f"{name} cannot be undefined")
            exact[name] = {"lower": value, "upper": value}
        return exact
    rng = np.random.default_rng(seed)
    size = len(y_true)
    mae_values = np.empty(iterations, dtype=float)
    rmse_values = np.empty(iterations, dtype=float)
    errors = y_pred - y_true
    for iteration in range(iterations):
        indices = rng.integers(0, size, size=size)
        sampled_errors = errors[indices]
        mae_values[iteration] = float(np.mean(np.abs(sampled_errors)))
        rmse_values[iteration] = float(np.sqrt(np.mean(sampled_errors * sampled_errors)))
    alpha = 1.0 - confidence
    return {
        "mae": {
            "lower": float(np.quantile(mae_values, alpha / 2.0)),
            "upper": float(np.quantile(mae_values, 1.0 - alpha / 2.0)),
        },
        "rmse": {
            "lower": float(np.quantile(rmse_values, alpha / 2.0)),
            "upper": float(np.quantile(rmse_values, 1.0 - alpha / 2.0)),
        },
    }


def regression_report(
    rows: Sequence[Mapping[str, Any]],
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    *,
    evaluation: EvaluationConfig,
    seed: int,
) -> dict[str, Any]:
    """Compute overall metrics, bootstrap bounds, and predefined risk slices."""

    y_true, y_pred = _aligned_arrays(actual, predicted)
    if len(rows) != len(y_true):
        raise ValueError("risk-slice rows must align with evaluation arrays")
    overall = point_metrics(y_true, y_pred)
    intervals = bootstrap_intervals(
        y_true,
        y_pred,
        iterations=evaluation.bootstrap_iterations,
        confidence=evaluation.bootstrap_confidence,
        seed=seed,
    )
    slice_masks: dict[str, np.ndarray] = {
        "young_driver_under_25": np.asarray(
            [float(row["driver_age"]) < 25.0 for row in rows],
            dtype=bool,
        ),
        "senior_driver_60_plus": np.asarray(
            [float(row["driver_age"]) >= 60.0 for row in rows],
            dtype=bool,
        ),
        "claims_history": np.asarray(
            [int(row["claims_5y"]) > 0 for row in rows],
            dtype=bool,
        ),
        "high_postcode_risk": np.asarray(
            [float(row["postcode_risk"]) >= 0.70 for row in rows],
            dtype=bool,
        ),
        "high_value_vehicle": np.asarray(
            [float(row["vehicle_value"]) >= 100_000.0 for row in rows],
            dtype=bool,
        ),
        "business_usage": np.asarray(
            [row["usage"] == "business" for row in rows],
            dtype=bool,
        ),
        "street_parking": np.asarray(
            [row["parking"] == "street" for row in rows],
            dtype=bool,
        ),
    }
    slices: dict[str, Any] = {}
    for name, mask in slice_masks.items():
        count = int(np.sum(mask))
        slices[name] = {
            "count": count,
            "metrics": None if count == 0 else point_metrics(y_true[mask], y_pred[mask]),
        }
    return {
        "count": len(y_true),
        "metrics": overall,
        "bootstrap_confidence": evaluation.bootstrap_confidence,
        "bootstrap_intervals": intervals,
        "risk_slices": slices,
    }


def conformal_radius(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    *,
    coverage: float,
) -> float:
    """Finite-sample split-conformal absolute-residual quantile."""

    y_true, y_pred = _aligned_arrays(actual, predicted)
    residuals = np.sort(np.abs(y_true - y_pred))
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * coverage))
    return float(residuals[max(0, rank - 1)])


def conformal_bounds(
    predicted: Sequence[float] | np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predicted, dtype=float)
    if predictions.ndim != 1 or not np.all(np.isfinite(predictions)):
        raise ValueError("predictions must be a finite one-dimensional array")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("conformal radius must be finite and non-negative")
    return np.maximum(0.0, predictions - radius), predictions + radius


def interval_report(
    actual: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    y_true = np.asarray(actual, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if (
        y_true.ndim != 1
        or y_true.size == 0
        or y_true.shape != low.shape
        or y_true.shape != high.shape
        or not np.all(np.isfinite(y_true))
        or not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or np.any(low > high)
    ):
        raise ValueError("invalid interval evaluation arrays")
    covered = (y_true >= low) & (y_true <= high)
    widths = high - low
    return {
        "count": len(y_true),
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "p95_width": float(np.quantile(widths, 0.95, method="higher")),
    }


@dataclass(frozen=True)
class EarlyStopTracker:
    """Serializable early-stop state driven solely by validation confidence bounds."""

    batches_seen: int = 0
    best_mae: float | None = None
    best_lower_bound: float | None = None
    stale_batches: int = 0
    stopped: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> EarlyStopTracker:
        if raw is None:
            return cls()
        return cls(
            batches_seen=int(raw.get("batches_seen", 0)),
            best_mae=(None if raw.get("best_mae") is None else float(raw["best_mae"])),
            best_lower_bound=(
                None if raw.get("best_lower_bound") is None else float(raw["best_lower_bound"])
            ),
            stale_batches=int(raw.get("stale_batches", 0)),
            stopped=bool(raw.get("stopped", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batches_seen": self.batches_seen,
            "best_mae": self.best_mae,
            "best_lower_bound": self.best_lower_bound,
            "stale_batches": self.stale_batches,
            "stopped": self.stopped,
        }

    def update(
        self,
        *,
        mae: float,
        lower_bound: float,
        upper_bound: float,
        config: EarlyStoppingConfig,
    ) -> EarlyStopTracker:
        if not all(
            math.isfinite(value) and value >= 0 for value in (mae, lower_bound, upper_bound)
        ):
            raise ValueError("early-stop metrics must be finite and non-negative")
        if lower_bound > upper_bound:
            raise ValueError("early-stop confidence bounds are inverted")

        batches_seen = self.batches_seen + 1
        if self.best_mae is None or self.best_lower_bound is None:
            return EarlyStopTracker(
                batches_seen=batches_seen,
                best_mae=mae,
                best_lower_bound=lower_bound,
                stale_batches=0,
                stopped=False,
            )

        threshold = self.best_lower_bound * (1.0 - config.minimum_relative_improvement)
        significant_improvement = upper_bound < threshold
        stale = 0 if significant_improvement else self.stale_batches + 1
        best_mae = mae if significant_improvement else self.best_mae
        best_lower = lower_bound if significant_improvement else self.best_lower_bound
        stopped = (
            config.patience_batches > 0
            and batches_seen >= config.minimum_batches
            and stale >= config.patience_batches
        )
        return EarlyStopTracker(
            batches_seen=batches_seen,
            best_mae=best_mae,
            best_lower_bound=best_lower,
            stale_batches=stale,
            stopped=stopped,
        )
