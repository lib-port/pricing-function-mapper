"""Pluggable active-acquisition scores with greedy diversity selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pricing_mapper.config import AcquisitionConfig
from pricing_mapper.domain import DomainSpec
from pricing_mapper.encoding import FeatureEncoder


def normalize01(values: np.ndarray) -> np.ndarray:
    """Normalize finite values to [0, 1] without producing NaN."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array.copy()
    if not np.all(np.isfinite(array)):
        raise ValueError("acquisition scores must be finite")
    low = float(np.min(array))
    high = float(np.max(array))
    if high <= low:
        return np.zeros_like(array)
    return (array - low) / (high - low)


@dataclass(frozen=True)
class AcquisitionContext:
    candidate_rows: Sequence[Mapping[str, Any]]
    training_rows: Sequence[Mapping[str, Any]]
    candidate_features: np.ndarray
    training_features: np.ndarray
    residual_anchor_features: np.ndarray
    predictive_std: np.ndarray
    training_residuals: np.ndarray
    domain: DomainSpec


class AcquisitionScore(Protocol):
    """Protocol implemented by every independently testable score component."""

    name: str

    def score(self, context: AcquisitionContext) -> np.ndarray:
        """Return one finite score per candidate."""


class UncertaintyScore:
    name = "uncertainty"

    def score(self, context: AcquisitionContext) -> np.ndarray:
        return normalize01(context.predictive_std)


class ResidualScore:
    name = "residual"

    def score(self, context: AcquisitionContext) -> np.ndarray:
        if context.residual_anchor_features.shape[0] == 0 or context.training_residuals.size == 0:
            return np.zeros(len(context.candidate_rows), dtype=float)
        if context.residual_anchor_features.shape[0] != context.training_residuals.size:
            raise ValueError("residual anchors and residual values must be aligned")
        nearest = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nearest.fit(context.residual_anchor_features)
        indices = nearest.kneighbors(
            context.candidate_features,
            return_distance=False,
        )[:, 0]
        return normalize01(np.abs(context.training_residuals[indices]))


class BreakpointScore:
    name = "breakpoint"

    _BREAKPOINTS: ClassVar[dict[str, tuple[float, ...]]] = {
        "driver_age": (21.0, 25.0, 35.0, 60.0),
        "years_licensed": (3.0,),
        "vehicle_year": (),
        "annual_km": (12_000.0,),
        "excess": (600.0,),
    }

    def score(self, context: AcquisitionContext) -> np.ndarray:
        values = np.zeros(len(context.candidate_rows), dtype=float)
        for index, row in enumerate(context.candidate_rows):
            total = 0.0
            for name, breakpoints in self._BREAKPOINTS.items():
                if not breakpoints:
                    continue
                bounds = context.domain.numeric[name]
                scale = max(bounds.high - bounds.low, 1.0)
                distance = min(abs(float(row[name]) - point) for point in breakpoints) / scale
                total += float(np.exp(-40.0 * distance))
            values[index] = total
        return normalize01(values)


class DiversityScore:
    name = "diversity"

    def score(self, context: AcquisitionContext) -> np.ndarray:
        if len(context.training_rows) == 0:
            return np.ones(len(context.candidate_rows), dtype=float)
        nearest = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nearest.fit(context.training_features)
        distances = nearest.kneighbors(
            context.candidate_features,
            return_distance=True,
        )[
            0
        ][:, 0]
        return normalize01(distances)


DEFAULT_SCORE_COMPONENTS: tuple[AcquisitionScore, ...] = (
    UncertaintyScore(),
    ResidualScore(),
    BreakpointScore(),
    DiversityScore(),
)


class AcquisitionStrategy:
    """Weighted component composition followed by greedy max-min diversity."""

    def __init__(
        self,
        config: AcquisitionConfig,
        components: Sequence[AcquisitionScore] = DEFAULT_SCORE_COMPONENTS,
    ) -> None:
        available = {component.name: component for component in components}
        missing = sorted(set(config.weights) - set(available))
        if missing:
            raise ValueError(f"missing acquisition component implementations: {missing}")
        self.config = config
        self.components = available

    def combined_scores(
        self, context: AcquisitionContext
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        component_values: dict[str, np.ndarray] = {}
        combined = np.zeros(len(context.candidate_rows), dtype=float)
        weight_total = sum(self.config.weights.values())
        for name, weight in self.config.weights.items():
            values = self.components[name].score(context)
            if values.shape != combined.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"acquisition component {name!r} returned invalid scores")
            component_values[name] = values
            combined += (weight / weight_total) * values
        return normalize01(combined), component_values

    def select(self, context: AcquisitionContext, count: int) -> list[int]:
        if count < 0:
            raise ValueError("acquisition selection count cannot be negative")
        if count == 0 or len(context.candidate_rows) == 0:
            return []
        base, _ = self.combined_scores(context)
        return select_from_base_scores(
            context,
            base,
            count,
            diversity_weight=self.config.greedy_diversity_weight,
        )


def select_from_base_scores(
    context: AcquisitionContext,
    base_scores: np.ndarray,
    count: int,
    *,
    diversity_weight: float,
) -> list[int]:
    """Greedily select local scores with bounded max-min diversity."""

    if count < 0:
        raise ValueError("acquisition selection count cannot be negative")
    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError("acquisition diversity weight must be within [0, 1]")
    base = np.asarray(base_scores, dtype=float)
    expected = (len(context.candidate_rows),)
    if base.shape != expected or not np.all(np.isfinite(base)):
        raise ValueError("base acquisition scores must be finite and aligned")
    if count == 0 or len(context.candidate_rows) == 0:
        return []
    count = min(count, len(context.candidate_rows))
    available = np.ones(len(base), dtype=bool)
    selected: list[int] = []
    min_distance_to_selected = np.full(len(base), np.inf, dtype=float)

    for _ in range(count):
        if selected:
            diversity = normalize01(min_distance_to_selected)
            scores = (1.0 - diversity_weight) * base + diversity_weight * diversity
        else:
            scores = base.copy()
        scores[~available] = -np.inf
        chosen = int(np.argmax(scores))
        if not np.isfinite(scores[chosen]):
            break
        selected.append(chosen)
        available[chosen] = False
        differences = context.candidate_features - context.candidate_features[chosen]
        distances = np.sqrt(np.sum(differences * differences, axis=1))
        min_distance_to_selected = np.minimum(min_distance_to_selected, distances)
    return selected


def build_context(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    predictive_std: np.ndarray,
    training_residuals: np.ndarray,
    domain: DomainSpec,
    residual_anchor_rows: Sequence[Mapping[str, Any]] | None = None,
) -> AcquisitionContext:
    encoder = FeatureEncoder(domain)
    anchor_rows = training_rows if residual_anchor_rows is None else residual_anchor_rows
    return AcquisitionContext(
        candidate_rows=candidate_rows,
        training_rows=training_rows,
        candidate_features=encoder.scaled_one_hot(candidate_rows),
        training_features=encoder.scaled_one_hot(training_rows),
        residual_anchor_features=encoder.scaled_one_hot(anchor_rows),
        predictive_std=np.asarray(predictive_std, dtype=float),
        training_residuals=np.asarray(training_residuals, dtype=float),
        domain=domain,
    )
