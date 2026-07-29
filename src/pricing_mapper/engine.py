"""Strict offline inference engine reconstructed from safe v1 artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator

from pricing_mapper.domain import CarQuoteInput, DomainSpec
from pricing_mapper.encoding import FeatureEncoder
from pricing_mapper.evaluation import conformal_bounds
from pricing_mapper.models import ModelKind, predict_estimator
from pricing_mapper.types import Prediction


class PricingEngine:
    """Supported v1 inference entry point with calibrated 90% intervals."""

    def __init__(
        self,
        *,
        estimator: BaseEstimator,
        model_kind: ModelKind,
        domain: DomainSpec,
        encoder: FeatureEncoder,
        conformal_radius: float,
        conformal_coverage: float,
        model_version: str,
        warnings: Sequence[str] = (),
    ) -> None:
        if not math.isfinite(conformal_radius) or conformal_radius < 0:
            raise ValueError("conformal radius must be finite and non-negative")
        if not 0.5 < conformal_coverage < 1.0:
            raise ValueError("conformal coverage must be between 0.5 and 1.0")
        if not model_version.strip():
            raise ValueError("model_version cannot be empty")
        self.estimator = estimator
        self.model_kind = model_kind
        self.domain = domain
        self.encoder = encoder
        self.conformal_radius = conformal_radius
        self.conformal_coverage = conformal_coverage
        self.model_version = model_version
        self.warnings = tuple(warnings)

    @classmethod
    def load(cls, artifact_dir: str | Path) -> Self:
        """Load only a hash-verified, version-compatible skops v1 artifact."""

        from pricing_mapper.artifact import load_artifact_components

        components = load_artifact_components(artifact_dir)
        return cls(**components)

    def _validated_rows(
        self,
        rows: Sequence[Mapping[str, Any] | CarQuoteInput],
    ) -> list[dict[str, Any]]:
        return [CarQuoteInput.from_mapping(row, domain=self.domain).as_dict() for row in rows]

    def predict_batch(
        self,
        rows: Sequence[Mapping[str, Any] | CarQuoteInput],
    ) -> list[Prediction]:
        """Validate and predict a batch; out-of-domain values are never clipped."""

        if not rows:
            return []
        validated = self._validated_rows(rows)
        premiums = predict_estimator(
            self.estimator,
            self.model_kind,
            self.encoder,
            validated,
        )
        lower, upper = conformal_bounds(premiums, self.conformal_radius)
        return [
            Prediction(
                premium=float(premium),
                lower=float(low),
                upper=float(high),
                model_version=self.model_version,
                warnings=self.warnings,
            )
            for premium, low, high in zip(premiums, lower, upper, strict=True)
        ]

    def predict(self, row: Mapping[str, Any] | CarQuoteInput) -> Prediction:
        """Return one premium and calibrated interval."""

        return self.predict_batch([row])[0]

    def predict_row(self, row: Mapping[str, Any] | CarQuoteInput) -> Prediction:
        """Alias retained for discoverability in single-row workflows."""

        return self.predict(row)

    def predict_premiums(
        self,
        rows: Sequence[Mapping[str, Any] | CarQuoteInput],
    ) -> np.ndarray:
        """Return only point estimates for evaluation code."""

        predictions = self.predict_batch(rows)
        return np.asarray([item.premium for item in predictions], dtype=float)

    def model_info(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": 1,
            "model_version": self.model_version,
            "model_kind": self.model_kind,
            "conformal_coverage": self.conformal_coverage,
            "conformal_radius": self.conformal_radius,
            "warnings": list(self.warnings),
            "domain": self.domain.to_dict(),
            "encoding": self.encoder.metadata(),
        }
