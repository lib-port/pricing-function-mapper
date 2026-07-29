from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from conftest import tiny_config

from pricing_mapper.benchmark import run_benchmark


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
                mapping_samples=self.config.sampling.mapping_budget,
            )

    monkeypatch.setattr("pricing_mapper.benchmark.MappingRun", FakeRun)
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
