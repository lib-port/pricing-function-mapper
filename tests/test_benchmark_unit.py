from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import tiny_config

from pricing_mapper.benchmark import run_benchmark, run_hybrid_benchmark
from pricing_mapper.config import OllamaConfig, OptimizerConfig


def test_five_seed_benchmark_summary_and_baseline(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    class FakeRun:
        def __init__(self, config: Any, **_: Any) -> None:
            self.config = config

        def run(self) -> Any:
            strategy = self.config.sampling.strategy
            mae = {"active": 80.0, "lhs": 100.0, "random": 110.0}[strategy]
            latency = {"active": 2.0, "lhs": 1.8, "random": 1.7}[strategy]
            return SimpleNamespace(
                evaluation_report={
                    "audit": {
                        "metrics": {
                            "mae": mae,
                            "rmse": mae * 1.2,
                            "p95_absolute_error": mae * 2,
                        }
                    },
                    "audit_interval": {"coverage": 0.9},
                    "latency": {"warm_single_row_p95_ms": latency},
                    "promotion_gates": {"eligible": True},
                },
                artifact_dir=Path("fake-artifact"),
                mapping_samples=self.config.sampling.mapping_budget,
            )

    monkeypatch.setattr("pricing_mapper.benchmark.MappingRun", FakeRun)
    monkeypatch.setattr(
        "pricing_mapper.benchmark.validate_artifact",
        lambda _: {"valid": True},
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"summary": {"active": {"median_p95_latency_ms": 2.0}}}),
        encoding="utf-8",
    )
    output = tmp_path / "benchmark.json"
    payload = run_benchmark(
        tiny_config(tmp_path / "runs"),
        output,
        baseline=baseline,
    )
    assert len(payload["records"]) == 15
    assert payload["gates"]["passed"]
    assert payload["gates"]["active_median_mae_improvement_over_lhs"] == 0.2
    assert payload["gates"]["coverage_passed"]
    assert output.is_file()
    assert output.with_suffix(".csv").is_file()


def test_hybrid_five_seed_ablation_and_production_gate(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    class FakeRun:
        def __init__(self, config: Any, **_: Any) -> None:
            self.config = config

        def run(self) -> Any:
            if self.config.sampling.strategy == "active":
                mae = 100.0
            elif self.config.optimizer.ollama is None:
                mae = 90.0
            else:
                mae = 80.0
            enabled = self.config.optimizer.ollama is not None
            return SimpleNamespace(
                evaluation_report={
                    "audit": {
                        "metrics": {
                            "mae": mae,
                            "rmse": mae * 1.2,
                            "p95_absolute_error": mae * 2,
                        }
                    },
                    "audit_interval": {"coverage": 0.9},
                    "latency": {"warm_single_row_p95_ms": 2.0},
                    "advisor": {
                        "maximum_response_latency_ms": 10.0 if enabled else 0.0,
                        "installed_model_bytes": 2_099_501_664 if enabled else 0,
                        "maximum_resident_model_bytes": 2_350_000_000 if enabled else 0,
                        "memory_passed": True,
                    },
                    "promotion_gates": {"eligible": True},
                },
                artifact_dir=Path("fake-hybrid-artifact"),
                mapping_samples=self.config.sampling.mapping_budget,
            )

    monkeypatch.setattr("pricing_mapper.benchmark.MappingRun", FakeRun)
    monkeypatch.setattr(
        "pricing_mapper.benchmark.validate_artifact",
        lambda _: {"valid": True},
    )
    base = tiny_config(tmp_path / "runs")
    config = base.model_copy(
        update={
            "optimizer": OptimizerConfig(ollama=OllamaConfig(required_digest="sha256:" + "a" * 64))
        }
    )
    output = tmp_path / "hybrid.json"
    payload = run_hybrid_benchmark(config, output)
    assert len(payload["records"]) == 15
    assert payload["summary"]["current_active"]["mapping_budget"] == 260
    assert payload["summary"]["bayesian"]["mapping_budget"] == 208
    assert payload["gates"]["mapping_quote_reduction"] == pytest.approx(0.2)
    assert payload["gates"]["ablation_passed"]
    assert payload["gates"]["individual_seed_passed"]
    assert payload["gates"]["advisor_enabled_for_production"]
    assert payload["gates"]["production_mapping_strategy"] == "hybrid"
    assert output.is_file()
    assert output.with_suffix(".csv").is_file()

    class FailedAblationRun(FakeRun):
        def run(self) -> Any:
            result = super().run()
            if self.config.optimizer.ollama is not None:
                metrics = result.evaluation_report["audit"]["metrics"]
                metrics.update(
                    {
                        "mae": 89.0,
                        "rmse": 89.0 * 1.2,
                        "p95_absolute_error": 89.0 * 2,
                    }
                )
            return result

    monkeypatch.setattr("pricing_mapper.benchmark.MappingRun", FailedAblationRun)
    failed = run_hybrid_benchmark(config, tmp_path / "failed-hybrid.json")
    assert not failed["gates"]["passed"]
    assert not failed["gates"]["advisor_enabled_for_production"]
    assert failed["gates"]["production_mapping_strategy"] == "bayesian"
