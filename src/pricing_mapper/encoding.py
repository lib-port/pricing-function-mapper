"""Versioned, code-native feature reconstruction for sklearn estimators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pricing_mapper.domain import (
    CATEGORICAL_FIELDS,
    FIELD_ORDER,
    DomainSpec,
)


@dataclass(frozen=True)
class FeatureEncoder:
    """Deterministic encoder reconstructed from artifact JSON metadata."""

    domain: DomainSpec

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FIELD_ORDER

    @property
    def categorical_mask(self) -> tuple[bool, ...]:
        categorical = set(CATEGORICAL_FIELDS)
        return tuple(name in categorical for name in FIELD_ORDER)

    def ordinal(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Encode categoricals as native HGB category indices."""

        matrix = np.empty((len(rows), len(FIELD_ORDER)), dtype=np.float64)
        category_maps = {
            name: {level: index for index, level in enumerate(self.domain.categorical[name])}
            for name in CATEGORICAL_FIELDS
        }
        for row_index, row in enumerate(rows):
            for column_index, name in enumerate(FIELD_ORDER):
                if name in category_maps:
                    try:
                        matrix[row_index, column_index] = category_maps[name][row[name]]
                    except KeyError as exc:
                        raise ValueError(f"unknown categorical value for {name}") from exc
                else:
                    matrix[row_index, column_index] = float(row[name])
        return matrix

    def scaled_one_hot(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Encode numeric values to [0, 1] and categoricals as one-hot blocks."""

        width = (
            len(FIELD_ORDER)
            - len(CATEGORICAL_FIELDS)
            + sum(len(self.domain.categorical[name]) for name in CATEGORICAL_FIELDS)
        )
        matrix = np.zeros((len(rows), width), dtype=np.float64)
        numeric_names = [name for name in FIELD_ORDER if name not in CATEGORICAL_FIELDS]
        category_maps = {
            name: {level: index for index, level in enumerate(self.domain.categorical[name])}
            for name in CATEGORICAL_FIELDS
        }
        for row_index, row in enumerate(rows):
            column = 0
            for name in numeric_names:
                bounds = self.domain.numeric[name]
                matrix[row_index, column] = (float(row[name]) - bounds.low) / (
                    bounds.high - bounds.low
                )
                column += 1
            for name in CATEGORICAL_FIELDS:
                try:
                    category_index = category_maps[name][row[name]]
                except KeyError as exc:
                    raise ValueError(f"unknown categorical value for {name}") from exc
                matrix[row_index, column + category_index] = 1.0
                column += len(category_maps[name])
        return matrix

    def frame(self, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """Create the stable raw DataFrame consumed by the ExtraTrees pipeline."""

        return pd.DataFrame.from_records(rows, columns=list(FIELD_ORDER))

    def metadata(self) -> dict[str, Any]:
        return {
            "encoding_version": 1,
            "field_order": list(FIELD_ORDER),
            "categorical_fields": list(CATEGORICAL_FIELDS),
            "categorical_levels": {
                name: list(self.domain.categorical[name]) for name in CATEGORICAL_FIELDS
            },
            "hgb_encoding": "ordinal_native_categorical",
            "committee_encoding": "scaled_numeric_plus_one_hot",
            "extra_trees_encoding": "sklearn_one_hot_pipeline",
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        domain: DomainSpec,
    ) -> FeatureEncoder:
        expected = cls(domain).metadata()
        if dict(metadata) != expected:
            raise ValueError("artifact feature encoding metadata does not match its domain")
        return cls(domain)


def rows_to_mappings(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalize validated model objects or mappings for internal estimators."""

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "as_dict"):
            normalized.append(row.as_dict())
        else:
            normalized.append(dict(row))
    return normalized
