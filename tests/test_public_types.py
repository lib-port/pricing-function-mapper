from __future__ import annotations

import pytest
from pydantic import ValidationError

from pricing_mapper.types import Prediction


def test_prediction_is_strict_and_interval_ordered() -> None:
    prediction = Prediction(
        premium=100.0,
        lower=80.0,
        upper=120.0,
        model_version="v1-test",
    )
    assert prediction.warnings == ()
    with pytest.raises(ValidationError, match="lower <= premium <= upper"):
        Prediction(
            premium=100.0,
            lower=110.0,
            upper=120.0,
            model_version="v1-test",
        )
    with pytest.raises(ValidationError, match="model_version"):
        Prediction(
            premium=100.0,
            lower=80.0,
            upper=120.0,
            model_version=" ",
        )
    with pytest.raises(ValidationError):
        Prediction.model_validate(
            {
                "premium": "100",
                "lower": 80.0,
                "upper": 120.0,
                "model_version": "v1-test",
            }
        )
