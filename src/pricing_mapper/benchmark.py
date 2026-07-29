"""Five-seed query-efficiency and latency release benchmark."""

from __future__ import annotations

import csv
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any

from pricing_mapper.config import MapperConfig
from pricing_mapper.orchestration import MappingRun
from pricing_mapper.types import QuoteProvider

FIXED_BENCHMARK_SEEDS: tuple[int, ...] = (11, 23, 37, 53, 71)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_benchmark(
    config: MapperConfig,
    output: str | Path,
    *,
    provider: QuoteProvider | None = None,
    baseline: str | Path | None = None,
) -> dict[str, Any]:
    """Compare active acquisition with equal-budget LHS and random sampling."""

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pricing-mapper-benchmark-") as temporary:
        for strategy in ("active", "lhs", "random"):
            for seed in FIXED_BENCHMARK_SEEDS:
                sampling = config.sampling.model_copy(update={"strategy": strategy, "seed": seed})
                artifact = config.artifact.model_copy(update={"output_dir": temporary})
                benchmark_config = config.model_copy(
                    update={"sampling": sampling, "artifact": artifact}
                )
                result = MappingRun(
                    benchmark_config,
                    provider=provider,
                    run_id=f"benchmark-{strategy}-{seed}",
                ).run()
                audit_metrics = result.evaluation_report["audit"]["metrics"]
                audit_interval = result.evaluation_report["audit_interval"]
                latency = result.evaluation_report["latency"]
                promotion = result.evaluation_report["promotion_gates"]
                records.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
                        "audit_mae": float(audit_metrics["mae"]),
                        "audit_rmse": float(audit_metrics["rmse"]),
                        "audit_p95_absolute_error": float(audit_metrics["p95_absolute_error"]),
                        "audit_coverage": float(audit_interval["coverage"]),
                        "p95_latency_ms": float(latency["warm_single_row_p95_ms"]),
                        "promotion_eligible": bool(promotion["eligible"]),
                        "mapping_samples": result.mapping_samples,
                    }
                )

    summary: dict[str, Any] = {}
    for strategy in ("active", "lhs", "random"):
        selected = [record for record in records if record["strategy"] == strategy]
        summary[strategy] = {
            "median_audit_mae": statistics.median(record["audit_mae"] for record in selected),
            "median_p95_latency_ms": statistics.median(
                record["p95_latency_ms"] for record in selected
            ),
            "maximum_p95_latency_ms": max(record["p95_latency_ms"] for record in selected),
            "mean_audit_coverage": statistics.mean(record["audit_coverage"] for record in selected),
            "promotion_eligible_runs": sum(record["promotion_eligible"] for record in selected),
            "seeds": list(FIXED_BENCHMARK_SEEDS),
        }
    lhs_mae = float(summary["lhs"]["median_audit_mae"])
    active_mae = float(summary["active"]["median_audit_mae"])
    improvement = 0.0 if lhs_mae == 0 else (lhs_mae - active_mae) / lhs_mae
    accuracy_passed = improvement >= 0.10
    active_latency_records = [
        float(record["p95_latency_ms"]) for record in records if record["strategy"] == "active"
    ]
    latency_passed = all(
        latency <= config.model.max_p95_latency_ms for latency in active_latency_records
    )
    active_coverage = float(summary["active"]["mean_audit_coverage"])
    coverage_passed = (
        config.evaluation.minimum_audit_coverage
        <= active_coverage
        <= config.evaluation.maximum_audit_coverage
    )

    latency_regression: float | None = None
    latency_regression_passed = True
    if baseline is not None:
        baseline_raw = json.loads(Path(baseline).read_text(encoding="utf-8"))
        previous = float(baseline_raw["summary"]["active"]["median_p95_latency_ms"])
        current = float(summary["active"]["median_p95_latency_ms"])
        if not math.isfinite(previous) or previous <= 0:
            raise ValueError("benchmark baseline active median latency must be positive")
        latency_regression = (current - previous) / previous
        latency_regression_passed = latency_regression <= 0.20

    payload = {
        "schema_version": 1,
        "fixed_seeds": list(FIXED_BENCHMARK_SEEDS),
        "mapping_budget": config.sampling.mapping_budget,
        "evaluation_budget": config.evaluation.evaluation_budget,
        "records": records,
        "summary": summary,
        "gates": {
            "active_median_mae_improvement_over_lhs": improvement,
            "required_improvement": 0.10,
            "accuracy_passed": accuracy_passed,
            "active_mean_audit_coverage": active_coverage,
            "minimum_audit_coverage": config.evaluation.minimum_audit_coverage,
            "maximum_audit_coverage": config.evaluation.maximum_audit_coverage,
            "coverage_passed": coverage_passed,
            "latency_passed": latency_passed,
            "latency_regression": latency_regression,
            "maximum_latency_regression": 0.20,
            "latency_regression_passed": latency_regression_passed,
            "passed": (
                accuracy_passed and coverage_passed and latency_passed and latency_regression_passed
            ),
        },
    }
    output_path = Path(output)
    _atomic_json(output_path, payload)
    csv_path = output_path.with_suffix(".csv")
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary_csv.replace(csv_path)
    return payload
