from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pricing_mapper.config import (
    ArtifactConfig,
    EarlyStoppingConfig,
    EvaluationConfig,
    MapperConfig,
    ModelConfig,
    ProviderConfig,
    SamplingConfig,
)


@pytest.fixture
def valid_row() -> dict[str, Any]:
    return {
        "driver_age": 40.0,
        "years_licensed": 20,
        "vehicle_year": 2022,
        "vehicle_value": 35_000.0,
        "annual_km": 10_000,
        "claims_5y": 0,
        "convictions_5y": 0,
        "postcode_risk": 0.2,
        "theft_risk": 0.2,
        "excess": 700,
        "usage": "private",
        "parking": "garage",
        "hire_car": "none",
        "windscreen": "no",
        "rating": "market",
    }


def tiny_config(
    output_dir: Path,
    **sampling_updates: Any,
) -> MapperConfig:
    sampling_values: dict[str, Any] = {
        "mapping_budget": 10,
        "initial_size": 6,
        "batch_size": 2,
        "candidate_pool_size": 30,
        "seed": 17,
        "strategy": "active",
    }
    sampling_values.update(sampling_updates)
    return MapperConfig(
        sampling=SamplingConfig(**sampling_values),
        model=ModelConfig(
            search_iterations=1,
            hgb_max_iter=20,
            extra_trees_estimators=20,
            committee_size=2,
            committee_estimators=10,
            n_jobs=1,
            max_p95_latency_ms=1_000.0,
            latency_repetitions=5,
            early_stopping=EarlyStoppingConfig(
                patience_batches=0,
                minimum_batches=2,
                minimum_relative_improvement=0.01,
                confidence_level=0.90,
            ),
        ),
        evaluation=EvaluationConfig(
            evaluation_budget=9,
            bootstrap_iterations=20,
        ),
        provider=ProviderConfig(
            max_retries=0,
            initial_backoff_seconds=0.0,
            maximum_backoff_seconds=0.0,
        ),
        artifact=ArtifactConfig(output_dir=str(output_dir)),
    )


@pytest.fixture
def completed_run(tmp_path: Path) -> tuple[MapperConfig, Any]:
    from pricing_mapper.orchestration import MappingRun

    config = tiny_config(tmp_path)
    result = MappingRun(config, run_id="test-run").run()
    return config, result
