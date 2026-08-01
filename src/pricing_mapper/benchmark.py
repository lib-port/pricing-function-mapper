"""Five-seed query-efficiency and latency release benchmark."""

from __future__ import annotations

import csv
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any

from pricing_mapper.advisor import PolicyAdvisor
from pricing_mapper.artifact import validate_artifact
from pricing_mapper.config import MapperConfig, OptimizerConfig
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

    if config.optimizer.ollama is not None:
        return run_hybrid_benchmark(
            config,
            output,
            provider=provider,
        )

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


def run_hybrid_benchmark(
    config: MapperConfig,
    output: str | Path,
    *,
    provider: QuoteProvider | None = None,
    advisor: PolicyAdvisor | None = None,
) -> dict[str, Any]:
    """Run the five-seed active/Bayesian/hybrid production ablation gate."""

    if config.optimizer.ollama is None:
        raise ValueError("the hybrid benchmark requires optimizer.ollama configuration")
    active_budget = 260
    bayesian_budget = 208
    if bayesian_budget < config.sampling.initial_size:
        raise ValueError(
            "the 20%-reduced Bayesian benchmark budget cannot be below sampling.initial_size"
        )
    variants = (
        ("current_active", "active", active_budget, OptimizerConfig(), None),
        ("bayesian", "bayesian", bayesian_budget, OptimizerConfig(), None),
        ("hybrid", "bayesian", bayesian_budget, config.optimizer, advisor),
    )
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pricing-mapper-hybrid-benchmark-") as temporary:
        for label, strategy, budget, optimizer, selected_advisor in variants:
            for seed in FIXED_BENCHMARK_SEEDS:
                sampling = config.sampling.model_copy(
                    update={"strategy": strategy, "seed": seed, "mapping_budget": budget}
                )
                artifact = config.artifact.model_copy(update={"output_dir": temporary})
                benchmark_config = config.model_copy(
                    update={
                        "sampling": sampling,
                        "artifact": artifact,
                        "optimizer": optimizer,
                    }
                )
                result = MappingRun(
                    benchmark_config,
                    provider=provider,
                    advisor=selected_advisor,
                    run_id=f"benchmark-{label}-{seed}",
                ).run()
                inspection = validate_artifact(result.artifact_dir)
                audit_metrics = result.evaluation_report["audit"]["metrics"]
                audit_interval = result.evaluation_report["audit_interval"]
                latency = result.evaluation_report["latency"]
                advisor_report = result.evaluation_report["advisor"]
                promotion = result.evaluation_report["promotion_gates"]
                records.append(
                    {
                        "strategy": label,
                        "seed": seed,
                        "audit_mae": float(audit_metrics["mae"]),
                        "audit_rmse": float(audit_metrics["rmse"]),
                        "audit_p95_absolute_error": float(audit_metrics["p95_absolute_error"]),
                        "audit_coverage": float(audit_interval["coverage"]),
                        "prediction_p95_latency_ms": float(latency["warm_single_row_p95_ms"]),
                        "advisor_maximum_latency_ms": float(
                            advisor_report["maximum_response_latency_ms"]
                        ),
                        "advisor_installed_model_bytes": int(
                            advisor_report["installed_model_bytes"]
                        ),
                        "advisor_maximum_resident_model_bytes": int(
                            advisor_report["maximum_resident_model_bytes"]
                        ),
                        "advisor_memory_passed": bool(advisor_report["memory_passed"]),
                        "promotion_eligible": bool(promotion["eligible"]),
                        "artifact_integrity": bool(inspection["valid"]),
                        "mapping_samples": result.mapping_samples,
                    }
                )

    summary: dict[str, Any] = {}
    for label, _, budget, _, _ in variants:
        selected = [record for record in records if record["strategy"] == label]
        summary[label] = {
            "mapping_budget": budget,
            "median_audit_mae": statistics.median(record["audit_mae"] for record in selected),
            "median_prediction_p95_latency_ms": statistics.median(
                record["prediction_p95_latency_ms"] for record in selected
            ),
            "maximum_advisor_latency_ms": max(
                record["advisor_maximum_latency_ms"] for record in selected
            ),
            "maximum_advisor_installed_model_bytes": max(
                record["advisor_installed_model_bytes"] for record in selected
            ),
            "maximum_advisor_resident_model_bytes": max(
                record["advisor_maximum_resident_model_bytes"] for record in selected
            ),
            "mean_audit_coverage": statistics.mean(record["audit_coverage"] for record in selected),
            "promotion_eligible_runs": sum(record["promotion_eligible"] for record in selected),
            "seeds": list(FIXED_BENCHMARK_SEEDS),
        }

    active_mae = float(summary["current_active"]["median_audit_mae"])
    bayesian_mae = float(summary["bayesian"]["median_audit_mae"])
    hybrid_mae = float(summary["hybrid"]["median_audit_mae"])
    quote_reduction = 0.0 if active_budget == 0 else 1.0 - bayesian_budget / active_budget
    hybrid_vs_active = 0.0 if active_mae == 0.0 else (active_mae - hybrid_mae) / active_mae
    hybrid_vs_bayesian = 0.0 if bayesian_mae == 0.0 else (bayesian_mae - hybrid_mae) / bayesian_mae
    per_seed_regressions: dict[str, float] = {}
    for seed in FIXED_BENCHMARK_SEEDS:
        bayesian_seed = next(
            float(item["audit_mae"])
            for item in records
            if item["strategy"] == "bayesian" and item["seed"] == seed
        )
        hybrid_seed = next(
            float(item["audit_mae"])
            for item in records
            if item["strategy"] == "hybrid" and item["seed"] == seed
        )
        per_seed_regressions[str(seed)] = (
            0.0 if bayesian_seed == 0.0 else (hybrid_seed - bayesian_seed) / bayesian_seed
        )
    query_efficiency_passed = quote_reduction >= 0.20 - 1e-12 and hybrid_mae <= active_mae
    budget_compliance_passed = all(
        (
            int(item["mapping_samples"]) == 260
            if item["strategy"] == "current_active"
            else int(item["mapping_samples"]) <= 208
        )
        for item in records
    )
    query_efficiency_passed = query_efficiency_passed and budget_compliance_passed
    ablation_passed = hybrid_vs_bayesian >= 0.05
    individual_seed_passed = all(value <= 0.10 for value in per_seed_regressions.values())
    hybrid_records = [item for item in records if item["strategy"] == "hybrid"]
    promotion_passed = all(bool(item["promotion_eligible"]) for item in hybrid_records)
    integrity_passed = all(bool(item["artifact_integrity"]) for item in records)
    advisor_latency_passed = all(
        float(item["advisor_maximum_latency_ms"]) <= 60_000.0 for item in hybrid_records
    )
    advisor_memory_passed = all(bool(item["advisor_memory_passed"]) for item in hybrid_records)
    passed = (
        query_efficiency_passed
        and ablation_passed
        and individual_seed_passed
        and promotion_passed
        and integrity_passed
        and advisor_latency_passed
        and advisor_memory_passed
    )
    payload = {
        "schema_version": 2,
        "fixed_seeds": list(FIXED_BENCHMARK_SEEDS),
        "records": records,
        "summary": summary,
        "gates": {
            "mapping_quote_reduction": quote_reduction,
            "required_mapping_quote_reduction": 0.20,
            "budget_compliance_passed": budget_compliance_passed,
            "hybrid_median_mae_improvement_over_current_active": hybrid_vs_active,
            "query_efficiency_passed": query_efficiency_passed,
            "hybrid_median_mae_improvement_over_bayesian": hybrid_vs_bayesian,
            "required_bayesian_ablation_improvement": 0.05,
            "ablation_passed": ablation_passed,
            "per_seed_mae_regression_over_bayesian": per_seed_regressions,
            "maximum_individual_seed_regression": 0.10,
            "individual_seed_passed": individual_seed_passed,
            "promotion_passed": promotion_passed,
            "artifact_integrity_passed": integrity_passed,
            "advisor_latency_ceiling_ms": 60_000.0,
            "advisor_latency_passed": advisor_latency_passed,
            "advisor_memory_ceiling_bytes": 8 * 1024**3,
            "advisor_memory_passed": advisor_memory_passed,
            "passed": passed,
            "production_mapping_strategy": "hybrid" if passed else "bayesian",
            "advisor_enabled_for_production": passed,
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
