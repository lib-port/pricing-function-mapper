from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from conftest import tiny_config

from pricing_mapper.domain import FIELD_ORDER, CarQuoteInput
from pricing_mapper.engine import PricingEngine
from pricing_mapper.exceptions import PersistenceError, ProviderRejected
from pricing_mapper.orchestration import MappingRun
from pricing_mapper.provider import reference_car_quote


class CrashOnceProvider:
    provider_id = "test.crash-once"
    thread_safe = False
    max_concurrency = 1

    def __init__(self, fail_at: int | None) -> None:
        self.fail_at = fail_at
        self.calls = 0
        self.failed = False

    def __call__(self, quote: CarQuoteInput) -> float:
        self.calls += 1
        if self.fail_at == self.calls and not self.failed:
            self.failed = True
            raise ProviderRejected("simulated process boundary")
        return reference_car_quote(quote)


def _mapping_dataset(path: Path) -> list[dict[str, str]]:
    with (path / "dataset.csv").open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == "mapping"]


def test_crash_resume_matches_uninterrupted_mapping_exactly(tmp_path: Path) -> None:
    resumed_config = tiny_config(tmp_path / "resumed")
    crashing_provider = CrashOnceProvider(fail_at=12)
    resumed_run = MappingRun(
        resumed_config,
        provider=crashing_provider,
        run_id="deterministic",
    )
    with pytest.raises(ProviderRejected):
        resumed_run.run()
    state = resumed_run.state_database
    assert state.is_file()
    connection = sqlite3.connect(state)
    try:
        before = connection.execute(
            "SELECT COUNT(*) FROM samples WHERE status = 'complete'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert before >= resumed_config.evaluation.evaluation_budget

    resumed_result = resumed_run.resume()
    full_config = tiny_config(tmp_path / "full")
    full_result = MappingRun(
        full_config,
        provider=CrashOnceProvider(fail_at=None),
        run_id="uninterrupted",
    ).run()
    assert _mapping_dataset(resumed_result.artifact_dir) == _mapping_dataset(
        full_result.artifact_dir
    )
    resumed_engine = PricingEngine.load(resumed_result.artifact_dir)
    full_engine = PricingEngine.load(full_result.artifact_dir)
    assert resumed_engine.model_version == full_engine.model_version
    audit_rows = [
        {
            name: (
                float(row[name])
                if name in {"driver_age", "vehicle_value", "postcode_risk", "theft_risk"}
                else (
                    int(row[name])
                    if name
                    in {
                        "years_licensed",
                        "vehicle_year",
                        "annual_km",
                        "claims_5y",
                        "convictions_5y",
                        "excess",
                    }
                    else row[name]
                )
            )
            for name in FIELD_ORDER
        }
        for row in _mapping_dataset(resumed_result.artifact_dir)[:3]
    ]
    assert resumed_engine.predict_premiums(audit_rows) == pytest.approx(
        full_engine.predict_premiums(audit_rows),
        rel=0.0,
        abs=0.0,
    )


def test_resume_rejects_missing_rng_after_mapping_started(tmp_path: Path) -> None:
    config = tiny_config(tmp_path / "rng-corrupt")
    provider = CrashOnceProvider(fail_at=12)
    run = MappingRun(config, provider=provider, run_id="rng-corrupt")
    with pytest.raises(ProviderRejected):
        run.run()
    connection = sqlite3.connect(run.state_database)
    try:
        connection.execute("DELETE FROM rng_states WHERE name = 'mapping'")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(PersistenceError, match="RNG state"):
        run.resume()


def test_holdout_rows_are_generated_first_and_have_disjoint_hashes(
    completed_run: tuple[Any, Any],
) -> None:
    _, result = completed_run
    connection = sqlite3.connect(result.state_database)
    try:
        rows = connection.execute(
            "SELECT ordinal, split, row_hash FROM samples ORDER BY ordinal"
        ).fetchall()
    finally:
        connection.close()
    evaluation_count = result.evaluation_samples
    assert all(split != "mapping" for _, split, _ in rows[:evaluation_count])
    assert all(split == "mapping" for _, split, _ in rows[evaluation_count:])
    assert len({row_hash for _, _, row_hash in rows}) == len(rows)


def test_report_records_leakage_guards(completed_run: tuple[Any, Any]) -> None:
    _, result = completed_run
    design = result.evaluation_report["evaluation_design"]
    assert design["samples_generated_before_adaptive_mapping"]
    assert not design["calibration_or_audit_used_for_acquisition"]
    assert not design["calibration_or_audit_used_for_tuning_or_early_stopping"]
    assert design["final_unbiased_data"] == ["audit"]


def test_fixed_seed_learning_curve_regression(
    completed_run: tuple[Any, Any],
) -> None:
    """Catch acquisition/training drift while allowing small numerical variation."""

    _, result = completed_run
    history = result.evaluation_report["batch_history"]
    assert [batch["metrics"]["mapping_samples"] for batch in history] == [6, 8, 10]
    observed = [batch["metrics"]["validation_mae"] for batch in history]
    assert observed == pytest.approx(
        [7_343.13625, 6_514.988139, 10_442.323028],
        rel=0.05,
        abs=1.0,
    )


def test_minimum_mapping_budget_completes_with_explicit_warnings(tmp_path: Path) -> None:
    config = tiny_config(
        tmp_path / "minimum",
        mapping_budget=1,
        initial_size=1,
        batch_size=1,
    )
    result = MappingRun(config, run_id="minimum").run()
    assert result.mapping_samples == 1
    assert result.evaluation_samples == config.evaluation.evaluation_budget
    assert result.evaluation_report["warnings"]
    assert PricingEngine.load(result.artifact_dir).model_version.startswith("v1-")


def test_seed_data_is_validated_and_reused(tmp_path: Path, valid_row: dict[str, Any]) -> None:
    seed = tmp_path / "seed.csv"
    with seed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*FIELD_ORDER, "premium"])
        writer.writeheader()
        writer.writerow({**valid_row, "premium": 999.0})
    config = tiny_config(
        tmp_path / "outputs",
        mapping_budget=6,
        initial_size=3,
        batch_size=2,
    )
    result = MappingRun(config, run_id="seeded").run(seed_data=seed)
    rows = _mapping_dataset(result.artifact_dir)
    assert any(row["source"] == "seed_data" and float(row["premium"]) == 999.0 for row in rows)
    different_seed = tmp_path / "different-seed.csv"
    with different_seed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*FIELD_ORDER, "premium"])
        writer.writeheader()
        writer.writerow({**valid_row, "premium": 1_000.0})
    with pytest.raises(PersistenceError, match="completed run import"):
        MappingRun(config, run_id="seeded").run(
            resume=True,
            seed_data=different_seed,
        )

    bad = tmp_path / "bad.csv"
    bad.write_text("premium\n100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        MappingRun(config, run_id="bad-seed").run(seed_data=bad)

    extra = tmp_path / "extra.csv"
    extra.write_text(
        ",".join([*FIELD_ORDER, "premium"])
        + "\n"
        + ",".join([str(valid_row[name]) for name in FIELD_ORDER])
        + ",999,unexpected\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="extra columns"):
        MappingRun(config, run_id="extra-seed").run(seed_data=extra)
