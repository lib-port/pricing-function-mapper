"""Deterministic acquisition committee and held-out sklearn model selection."""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import ThreadpoolController  # type: ignore[import-untyped]

from pricing_mapper.config import ModelConfig
from pricing_mapper.domain import CATEGORICAL_FIELDS, FIELD_ORDER, CarQuoteInput, DomainSpec
from pricing_mapper.encoding import FeatureEncoder

ModelKind = Literal["hist_gradient_boosting", "extra_trees"]
_SKLEARN_PARALLEL_WARNING = (
    r"`sklearn\.utils\.parallel\.delayed` should be used with "
    r"`sklearn\.utils\.parallel\.Parallel`"
)
_THREADPOOL_CONTROLLER = ThreadpoolController()


@dataclass(frozen=True)
class CandidateSpec:
    """Reconstructible estimator family and sampled hyperparameters."""

    name: str
    kind: ModelKind
    parameters: dict[str, Any]
    monotonic: bool


@dataclass(frozen=True)
class CandidateResult:
    """One validation-scored, latency-measured fitted candidate."""

    spec: CandidateSpec
    estimator: BaseEstimator
    validation_mae: float
    p95_latency_ms: float
    eligible: bool

    def report(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "kind": self.spec.kind,
            "parameters": self.spec.parameters,
            "monotonic": self.spec.monotonic,
            "validation_mae": self.validation_mae,
            "p95_latency_ms": self.p95_latency_ms,
            "latency_eligible": self.eligible,
        }


@dataclass(frozen=True)
class SelectionResult:
    selected: CandidateResult
    candidates: tuple[CandidateResult, ...]
    monotonic_tradeoff: dict[str, Any]

    def report(self) -> dict[str, Any]:
        return {
            "selected": self.selected.report(),
            "candidates": [candidate.report() for candidate in self.candidates],
            "monotonic_constraint_tradeoff": self.monotonic_tradeoff,
            "selection_rule": "lowest validation MAE satisfying the p95 latency ceiling",
        }


class AcquisitionCommittee:
    """Small deterministic RF committee used only for acquisition uncertainty."""

    def __init__(self, config: ModelConfig, domain: DomainSpec, seed: int) -> None:
        self.config = config
        self.domain = domain
        self.seed = seed
        self.encoder = FeatureEncoder(domain)
        self.models: list[RandomForestRegressor] = []

    def fit(self, rows: Sequence[Mapping[str, Any]], targets: np.ndarray) -> AcquisitionCommittee:
        if len(rows) == 0 or len(rows) != len(targets):
            raise ValueError("committee training rows and targets must be non-empty and aligned")
        features = self.encoder.scaled_one_hot(rows)
        self.models = []
        # The seed depends only on stable run state, not call order, which makes
        # crash/resume model fitting byte-for-byte deterministic in behavior.
        base_seed = (self.seed + 1_000_003 * len(rows)) % (2**32 - 1)
        for index in range(self.config.committee_size):
            model = RandomForestRegressor(
                n_estimators=self.config.committee_estimators,
                min_samples_leaf=max(1, min(3, len(rows) // 10 or 1)),
                max_features=0.8,
                bootstrap=True,
                random_state=int((base_seed + 7_919 * index) % (2**32 - 1)),
                n_jobs=self.config.n_jobs,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=_SKLEARN_PARALLEL_WARNING,
                    category=UserWarning,
                )
                model.fit(features, targets)
            self.models.append(model)
        return self

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("acquisition committee has not been fitted")
        if not rows:
            empty = np.empty(0, dtype=float)
            return empty, empty
        features = self.encoder.scaled_one_hot(rows)
        predictions = np.vstack([model.predict(features) for model in self.models])
        return predictions.mean(axis=0), predictions.std(axis=0, ddof=0)

    def training_residuals(
        self,
        rows: Sequence[Mapping[str, Any]],
        targets: np.ndarray,
    ) -> np.ndarray:
        mean, _ = self.predict(rows)
        residuals = np.asarray(targets, dtype=float) - mean
        return np.asarray(residuals, dtype=float)


class BayesianAcquisitionModel:
    """Deterministic Gaussian process used only for Bayesian acquisition scores."""

    def __init__(self, domain: DomainSpec, seed: int) -> None:
        self.domain = domain
        self.seed = seed
        self.encoder = FeatureEncoder(domain)
        self.model: GaussianProcessRegressor | None = None

    def fit(
        self,
        rows: Sequence[Mapping[str, Any]],
        targets: np.ndarray,
    ) -> BayesianAcquisitionModel:
        if len(rows) == 0 or len(rows) != len(targets):
            raise ValueError("Bayesian training rows and targets must be non-empty and aligned")
        features = self.encoder.scaled_one_hot(rows)
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=1.0,
            length_scale_bounds="fixed",
            nu=1.5,
        ) + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            optimizer=None,
            normalize_y=True,
            random_state=self.seed,
            copy_X_train=True,
        )
        self.model.fit(features, np.asarray(targets, dtype=float))
        return self

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Bayesian acquisition model has not been fitted")
        if not rows:
            empty = np.empty(0, dtype=float)
            return empty, empty
        mean, standard_deviation = self.model.predict(
            self.encoder.scaled_one_hot(rows),
            return_std=True,
        )
        mean_array = np.asarray(mean, dtype=float)
        standard_deviation_array = np.asarray(standard_deviation, dtype=float)
        if not np.all(np.isfinite(mean_array)) or not np.all(np.isfinite(standard_deviation_array)):
            raise RuntimeError("Bayesian acquisition model returned non-finite predictions")
        return mean_array, standard_deviation_array


def _choice(rng: np.random.Generator, values: Sequence[Any]) -> Any:
    return values[int(rng.integers(0, len(values)))]


def candidate_specs(config: ModelConfig, seed: int) -> tuple[CandidateSpec, ...]:
    """Create a bounded deterministic randomized candidate set."""

    rng = np.random.default_rng(np.random.SeedSequence([seed, 31_337]))
    specs: list[CandidateSpec] = []
    for iteration in range(config.search_iterations):
        hgb_parameters = {
            "learning_rate": float(_choice(rng, (0.035, 0.05, 0.075, 0.10))),
            "max_leaf_nodes": int(_choice(rng, (15, 23, 31, 47))),
            "min_samples_leaf": int(_choice(rng, (5, 10, 15, 20))),
            "l2_regularization": float(_choice(rng, (0.0, 0.1, 0.5, 1.0))),
            "max_iter": config.hgb_max_iter,
        }
        specs.append(
            CandidateSpec(
                name=f"hgb-unconstrained-{iteration}",
                kind="hist_gradient_boosting",
                parameters=dict(hgb_parameters),
                monotonic=False,
            )
        )
        specs.append(
            CandidateSpec(
                name=f"hgb-monotonic-{iteration}",
                kind="hist_gradient_boosting",
                parameters=dict(hgb_parameters),
                monotonic=True,
            )
        )
        specs.append(
            CandidateSpec(
                name=f"extra-trees-{iteration}",
                kind="extra_trees",
                parameters={
                    "n_estimators": config.extra_trees_estimators,
                    "min_samples_leaf": int(_choice(rng, (1, 2, 3, 5))),
                    "max_features": float(_choice(rng, (0.65, 0.8, 1.0))),
                    "max_depth": _choice(rng, (None, 12, 18, 28)),
                },
                monotonic=False,
            )
        )
    return tuple(specs)


def build_estimator(
    spec: CandidateSpec,
    *,
    config: ModelConfig,
    domain: DomainSpec,
    seed: int,
    training_size: int,
) -> BaseEstimator:
    """Build one estimator without carrying validation or domain objects into it."""

    if spec.kind == "hist_gradient_boosting":
        parameters = dict(spec.parameters)
        parameters["min_samples_leaf"] = max(
            1,
            min(int(parameters["min_samples_leaf"]), max(1, training_size // 3)),
        )
        constraints = [
            config.monotonic_constraints.get(name, 0) if spec.monotonic else 0
            for name in FIELD_ORDER
        ]
        estimator = HistGradientBoostingRegressor(
            **parameters,
            categorical_features=np.asarray(
                [name in CATEGORICAL_FIELDS for name in FIELD_ORDER],
                dtype=bool,
            ),
            monotonic_cst=constraints,
            early_stopping=False,
            random_state=seed,
        )
        return estimator

    if spec.kind == "extra_trees":
        categorical = list(CATEGORICAL_FIELDS)
        numeric = [name for name in FIELD_ORDER if name not in CATEGORICAL_FIELDS]
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(
                        categories=[list(domain.categorical[name]) for name in categorical],
                        handle_unknown="error",
                        sparse_output=False,
                        dtype=np.float64,
                    ),
                    categorical,
                ),
                ("numeric", "passthrough", numeric),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        regressor = ExtraTreesRegressor(
            **spec.parameters,
            random_state=seed,
            n_jobs=config.n_jobs,
            bootstrap=False,
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ]
        )
    raise ValueError(f"unsupported model kind: {spec.kind}")


def _fit_estimator(
    estimator: BaseEstimator,
    kind: ModelKind,
    encoder: FeatureEncoder,
    rows: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
) -> BaseEstimator:
    if kind == "hist_gradient_boosting":
        with _THREADPOOL_CONTROLLER.limit(limits=1, user_api="openmp"):
            estimator.fit(encoder.ordinal(rows), targets)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=_SKLEARN_PARALLEL_WARNING,
                category=UserWarning,
            )
            estimator.fit(encoder.frame(rows), targets)
    return estimator


def predict_estimator(
    estimator: BaseEstimator,
    kind: ModelKind,
    encoder: FeatureEncoder,
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Predict with code-native preprocessing selected by artifact metadata."""

    if not rows:
        return np.empty(0, dtype=float)
    if kind == "hist_gradient_boosting":
        with _THREADPOOL_CONTROLLER.limit(limits=1, user_api="openmp"):
            raw = estimator.predict(encoder.ordinal(rows))
    elif kind == "extra_trees":
        raw = estimator.predict(encoder.frame(rows))
    else:
        raise ValueError(f"unsupported model kind: {kind}")
    predictions = np.asarray(raw, dtype=float)
    if predictions.shape != (len(rows),) or not np.all(np.isfinite(predictions)):
        raise RuntimeError("fitted estimator returned invalid predictions")
    return np.asarray(np.maximum(predictions, 0.0), dtype=float)


def measure_single_row_latency(
    estimator: BaseEstimator,
    kind: ModelKind,
    encoder: FeatureEncoder,
    row: Mapping[str, Any],
    repetitions: int,
) -> float:
    """Measure warm single-row p95 prediction latency."""

    def predict_one() -> None:
        validated = CarQuoteInput.from_mapping(row, domain=encoder.domain).as_dict()
        predict_estimator(estimator, kind, encoder, [validated])

    for _ in range(3):
        predict_one()
    timings = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        started = time.perf_counter_ns()
        predict_one()
        timings[index] = (time.perf_counter_ns() - started) / 1_000_000.0
    return float(np.quantile(timings, 0.95, method="higher"))


def select_model(
    *,
    mapping_rows: Sequence[Mapping[str, Any]],
    mapping_targets: np.ndarray,
    validation_rows: Sequence[Mapping[str, Any]],
    validation_targets: np.ndarray,
    config: ModelConfig,
    domain: DomainSpec,
    seed: int,
) -> SelectionResult:
    """Tune on mapping rows and select using validation rows only."""

    if not mapping_rows or len(mapping_rows) != len(mapping_targets):
        raise ValueError("model selection requires aligned non-empty mapping data")
    if not validation_rows or len(validation_rows) != len(validation_targets):
        raise ValueError("model selection requires aligned non-empty validation data")

    encoder = FeatureEncoder(domain)
    results: list[CandidateResult] = []
    for index, spec in enumerate(candidate_specs(config, seed)):
        estimator = build_estimator(
            spec,
            config=config,
            domain=domain,
            seed=(seed + 10_007 * (index + 1)) % (2**32 - 1),
            training_size=len(mapping_rows),
        )
        fitted = _fit_estimator(estimator, spec.kind, encoder, mapping_rows, mapping_targets)
        predictions = predict_estimator(fitted, spec.kind, encoder, validation_rows)
        validation_mae = float(np.mean(np.abs(predictions - validation_targets)))
        latency = measure_single_row_latency(
            fitted,
            spec.kind,
            encoder,
            validation_rows[0],
            config.latency_repetitions,
        )
        results.append(
            CandidateResult(
                spec=spec,
                estimator=fitted,
                validation_mae=validation_mae,
                p95_latency_ms=latency,
                eligible=latency <= config.max_p95_latency_ms,
            )
        )

    eligible = [candidate for candidate in results if candidate.eligible]
    if not eligible:
        fastest = min(results, key=lambda candidate: candidate.p95_latency_ms)
        raise RuntimeError(
            "no candidate satisfies model.max_p95_latency_ms; fastest candidate "
            f"{fastest.spec.name!r} measured {fastest.p95_latency_ms:.3f} ms"
        )
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate.validation_mae,
            candidate.p95_latency_ms,
            candidate.spec.name,
        ),
    )

    constrained = [
        candidate
        for candidate in results
        if candidate.spec.kind == "hist_gradient_boosting" and candidate.spec.monotonic
    ]
    unconstrained = [
        candidate
        for candidate in results
        if candidate.spec.kind == "hist_gradient_boosting" and not candidate.spec.monotonic
    ]
    best_constrained = min(constrained, key=lambda candidate: candidate.validation_mae)
    best_unconstrained = min(unconstrained, key=lambda candidate: candidate.validation_mae)
    delta = best_constrained.validation_mae - best_unconstrained.validation_mae
    tradeoff = {
        "tested_on": "validation",
        "best_constrained_mae": best_constrained.validation_mae,
        "best_unconstrained_mae": best_unconstrained.validation_mae,
        "mae_delta_constrained_minus_unconstrained": delta,
        "constraint_improved_accuracy": delta < 0,
        "constraints": config.monotonic_constraints,
    }
    return SelectionResult(
        selected=selected,
        candidates=tuple(results),
        monotonic_tradeoff=tradeoff,
    )


def refit_selected(
    selection: SelectionResult,
    *,
    rows: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
    config: ModelConfig,
    domain: DomainSpec,
    seed: int,
) -> BaseEstimator:
    """Refit selected hyperparameters on mapping plus validation rows."""

    estimator = build_estimator(
        selection.selected.spec,
        config=config,
        domain=domain,
        seed=seed,
        training_size=len(rows),
    )
    return _fit_estimator(
        estimator,
        selection.selected.spec.kind,
        FeatureEncoder(domain),
        rows,
        targets,
    )


def fit_monitor_model(
    *,
    mapping_rows: Sequence[Mapping[str, Any]],
    mapping_targets: np.ndarray,
    validation_rows: Sequence[Mapping[str, Any]],
    config: ModelConfig,
    domain: DomainSpec,
    seed: int,
) -> tuple[BaseEstimator, np.ndarray]:
    """Fit a fixed deterministic ExtraTrees monitor for held-out early stopping."""

    spec = CandidateSpec(
        name="early-stop-monitor",
        kind="extra_trees",
        parameters={
            "n_estimators": max(20, min(80, config.extra_trees_estimators)),
            "min_samples_leaf": 2,
            "max_features": 0.8,
            "max_depth": 18,
        },
        monotonic=False,
    )
    estimator = build_estimator(
        spec,
        config=config,
        domain=domain,
        seed=seed,
        training_size=len(mapping_rows),
    )
    encoder = FeatureEncoder(domain)
    fitted = _fit_estimator(estimator, spec.kind, encoder, mapping_rows, mapping_targets)
    return fitted, predict_estimator(fitted, spec.kind, encoder, validation_rows)


def clone_and_fit(
    estimator: BaseEstimator,
    kind: ModelKind,
    encoder: FeatureEncoder,
    rows: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
) -> BaseEstimator:
    """Typed clone helper used by evaluation tests and diagnostics."""

    return _fit_estimator(clone(estimator), kind, encoder, rows, targets)


def validate_loaded_estimator(estimator: Any, kind: ModelKind) -> BaseEstimator:
    """Reject an estimator whose concrete sklearn family conflicts with metadata."""

    if kind == "hist_gradient_boosting":
        if not isinstance(estimator, HistGradientBoostingRegressor):
            raise ValueError("artifact model is not a HistGradientBoostingRegressor")
    elif kind == "extra_trees":
        if not isinstance(estimator, Pipeline):
            raise ValueError("artifact ExtraTrees model is not a sklearn Pipeline")
        if set(estimator.named_steps) != {"preprocessor", "regressor"}:
            raise ValueError("artifact ExtraTrees pipeline has unexpected steps")
        if not isinstance(estimator.named_steps["regressor"], ExtraTreesRegressor):
            raise ValueError("artifact pipeline does not contain ExtraTreesRegressor")
    else:
        raise ValueError(f"unsupported artifact model kind: {kind}")
    if not isinstance(estimator, BaseEstimator):
        raise ValueError("artifact model is not a fitted sklearn estimator")
    try:
        check_is_fitted(estimator)
    except NotFittedError as exc:
        raise ValueError("artifact sklearn estimator is not fitted") from exc
    return estimator


def finite_nonnegative_targets(values: Sequence[float]) -> np.ndarray:
    targets = np.asarray(values, dtype=float)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(targets)) or np.any(targets < 0):
        raise ValueError("targets must be finite and non-negative")
    return targets


def validation_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    if predictions.shape != targets.shape or predictions.size == 0:
        raise ValueError("validation arrays must be aligned and non-empty")
    value = float(np.mean(np.abs(predictions - targets)))
    if not math.isfinite(value):
        raise RuntimeError("validation MAE is non-finite")
    return value
