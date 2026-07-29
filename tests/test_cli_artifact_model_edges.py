from __future__ import annotations

import argparse
import csv
import json
import runpy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from conftest import tiny_config
from sklearn.ensemble import ExtraTreesRegressor

import pricing_mapper.cli as cli
from pricing_mapper.acquisition import AcquisitionContext, AcquisitionStrategy, normalize01
from pricing_mapper.artifact import artifact_config, validate_artifact
from pricing_mapper.config import AcquisitionConfig, config_toml
from pricing_mapper.domain import FIELD_ORDER, CarQuoteInput, DomainSpec
from pricing_mapper.encoding import FeatureEncoder, rows_to_mappings
from pricing_mapper.engine import PricingEngine
from pricing_mapper.evaluation import EarlyStopTracker, conformal_bounds, interval_report
from pricing_mapper.exceptions import ArtifactError, PersistenceError
from pricing_mapper.models import (
    AcquisitionCommittee,
    CandidateSpec,
    build_estimator,
    predict_estimator,
    validate_loaded_estimator,
    validation_mae,
)


def test_cli_map_resume_evaluate_and_json_input(tmp_path: Path, valid_row: dict[str, Any]) -> None:
    config = tiny_config(tmp_path / "outputs")
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_toml(config), encoding="utf-8")
    schema_path = tmp_path / "config.schema.json"
    assert (
        cli.run_cli(
            [
                "config",
                "validate",
                "--config",
                str(config_path),
                "--schema-output",
                str(schema_path),
            ]
        )
        == cli.EXIT_OK
    )
    assert schema_path.is_file()
    assert (
        cli.run_cli(
            [
                "map",
                "run",
                "--config",
                str(config_path),
                "--run-id",
                "cli-run",
            ]
        )
        == cli.EXIT_OK
    )
    artifact = tmp_path / "outputs" / "cli-run"
    assert (
        cli.run_cli(
            [
                "map",
                "resume",
                "--config",
                str(config_path),
                "--run-id",
                "cli-run",
            ]
        )
        == cli.EXIT_OK
    )
    report_path = tmp_path / "audit.json"
    code = cli.run_cli(
        [
            "model",
            "evaluate",
            "--artifact",
            str(artifact),
            "--output",
            str(report_path),
            "--require-gates",
        ]
    )
    assert code in {cli.EXIT_OK, cli.EXIT_GATE}
    assert report_path.is_file()

    row_path = tmp_path / "row.json"
    row_path.write_text(json.dumps(valid_row), encoding="utf-8")
    assert (
        cli.run_cli(
            [
                "predict",
                "row",
                "--artifact",
                str(artifact),
                "--input",
                str(row_path),
            ]
        )
        == cli.EXIT_OK
    )
    row_path.write_text("[]", encoding="utf-8")
    assert (
        cli.run_cli(
            [
                "predict",
                "row",
                "--artifact",
                str(artifact),
                "--input",
                str(row_path),
            ]
        )
        == cli.EXIT_INPUT
    )


def test_cli_benchmark_dispatch_and_run_failure_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_config(tmp_path)
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_toml(config), encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "run_benchmark",
        lambda *_args, **_kwargs: {"gates": {"passed": False}},
    )
    assert (
        cli.run_cli(
            [
                "benchmark",
                "--config",
                str(config_path),
                "--output",
                str(tmp_path / "benchmark.json"),
                "--enforce",
            ]
        )
        == cli.EXIT_GATE
    )

    def raise_persistence(_: argparse.Namespace) -> int:
        raise PersistenceError("broken")

    monkeypatch.setattr(cli, "_dispatch", raise_persistence)
    assert cli.run_cli(["config", "validate", "--config", str(config_path)]) == cli.EXIT_RUN


def test_cli_csv_and_audit_read_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("wrong\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        cli._read_prediction_csv(bad)
    with pytest.raises(ArtifactError, match="columns"):
        cli._read_audit_dataset(bad)

    extra = tmp_path / "extra.csv"
    extra.write_text(
        ",".join(FIELD_ORDER) + "\n" + ",".join(["1"] * (len(FIELD_ORDER) + 1)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="extra columns"):
        cli._read_prediction_csv(extra)

    no_audit = tmp_path / "no-audit.csv"
    with no_audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*FIELD_ORDER, "premium", "split", "source"],
        )
        writer.writeheader()
    with pytest.raises(ArtifactError, match="no audit"):
        cli._read_audit_dataset(no_audit)

    with pytest.raises(AssertionError):
        cli._dispatch(argparse.Namespace(command="impossible"))


def test_engine_encoding_acquisition_and_model_error_edges(
    valid_row: dict[str, Any],
) -> None:
    domain = DomainSpec.default()
    encoder = FeatureEncoder(domain)
    assert encoder.feature_names == FIELD_ORDER
    assert sum(encoder.categorical_mask) == 5
    assert encoder.ordinal([]).shape == (0, 15)
    assert encoder.scaled_one_hot([]).shape[0] == 0
    quote = CarQuoteInput.from_mapping(valid_row)
    assert rows_to_mappings([quote, valid_row]) == [quote.as_dict(), valid_row]
    with pytest.raises(ValueError, match="metadata"):
        FeatureEncoder.from_metadata({}, domain)
    with pytest.raises(ValueError, match="categorical"):
        encoder.ordinal([dict(valid_row, usage="fleet")])
    with pytest.raises(ValueError, match="categorical"):
        encoder.scaled_one_hot([dict(valid_row, usage="fleet")])

    with pytest.raises(ValueError, match="finite"):
        normalize01(np.asarray([1.0, np.nan]))
    strategy = AcquisitionStrategy(AcquisitionConfig())
    empty_context = AcquisitionContext(
        candidate_rows=[],
        training_rows=[],
        candidate_features=np.empty((0, 1)),
        training_features=np.empty((0, 1)),
        residual_anchor_features=np.empty((0, 1)),
        predictive_std=np.empty(0),
        training_residuals=np.empty(0),
        domain=domain,
    )
    assert strategy.select(empty_context, 0) == []
    with pytest.raises(ValueError, match="negative"):
        strategy.select(empty_context, -1)
    with pytest.raises(ValueError, match="missing"):
        AcquisitionStrategy(AcquisitionConfig(), components=[])

    model_config = tiny_config(Path("unused-test-output")).model
    committee = AcquisitionCommittee(model_config, domain, 1)
    with pytest.raises(ValueError, match="non-empty"):
        committee.fit([], np.empty(0))
    with pytest.raises(RuntimeError, match="fitted"):
        committee.predict([valid_row])
    with pytest.raises(ValueError, match="unsupported"):
        build_estimator(
            CandidateSpec("bad", cast(Any, "bad"), {}, False),
            config=model_config,
            domain=domain,
            seed=1,
            training_size=1,
        )
    assert (
        predict_estimator(
            ExtraTreesRegressor(),
            "extra_trees",
            encoder,
            [],
        ).size
        == 0
    )
    with pytest.raises(ValueError, match="unsupported"):
        predict_estimator(
            ExtraTreesRegressor(),
            cast(Any, "bad"),
            encoder,
            [valid_row],
        )
    with pytest.raises(ValueError, match="not a sklearn Pipeline"):
        validate_loaded_estimator(ExtraTreesRegressor(), "extra_trees")
    with pytest.raises(ValueError, match="HistGradient"):
        validate_loaded_estimator(ExtraTreesRegressor(), "hist_gradient_boosting")
    with pytest.raises(ValueError, match="not fitted"):
        validate_loaded_estimator(
            build_estimator(
                CandidateSpec(
                    "unfitted",
                    "hist_gradient_boosting",
                    {
                        "learning_rate": 0.1,
                        "max_leaf_nodes": 15,
                        "min_samples_leaf": 2,
                        "l2_regularization": 0.0,
                        "max_iter": 20,
                    },
                    False,
                ),
                config=model_config,
                domain=domain,
                seed=1,
                training_size=10,
            ),
            "hist_gradient_boosting",
        )
    with pytest.raises(ValueError, match="aligned"):
        validation_mae(np.asarray([1.0]), np.asarray([]))


def test_engine_constructor_evaluation_and_artifact_config_edges(
    completed_run: tuple[Any, Any],
) -> None:
    config, result = completed_run
    engine = PricingEngine.load(result.artifact_dir)
    assert engine.model_info()["model_version"] == engine.model_version
    assert engine.predict_premiums([]).size == 0
    assert artifact_config(result.artifact_dir) == config
    with pytest.raises(ValueError, match="radius"):
        PricingEngine(
            estimator=engine.estimator,
            model_kind=engine.model_kind,
            domain=engine.domain,
            encoder=engine.encoder,
            conformal_radius=-1.0,
            conformal_coverage=0.9,
            model_version="v1",
        )
    with pytest.raises(ValueError, match="coverage"):
        PricingEngine(
            estimator=engine.estimator,
            model_kind=engine.model_kind,
            domain=engine.domain,
            encoder=engine.encoder,
            conformal_radius=1.0,
            conformal_coverage=0.1,
            model_version="v1",
        )
    with pytest.raises(ValueError, match="model_version"):
        PricingEngine(
            estimator=engine.estimator,
            model_kind=engine.model_kind,
            domain=engine.domain,
            encoder=engine.encoder,
            conformal_radius=1.0,
            conformal_coverage=0.9,
            model_version=" ",
        )

    tracker = EarlyStopTracker.from_dict({"batches_seen": 2, "stopped": True})
    assert tracker.to_dict()["stopped"]
    with pytest.raises(ValueError, match="non-negative"):
        tracker.update(
            mae=-1.0,
            lower_bound=0.0,
            upper_bound=1.0,
            config=config.model.early_stopping,
        )
    with pytest.raises(ValueError, match="inverted"):
        tracker.update(
            mae=1.0,
            lower_bound=2.0,
            upper_bound=1.0,
            config=config.model.early_stopping,
        )
    with pytest.raises(ValueError, match="radius"):
        conformal_bounds([1.0], -1.0)
    with pytest.raises(ValueError, match="invalid interval"):
        interval_report([1.0], [2.0], [1.0])
    assert validate_artifact(result.artifact_dir, check_runtime=False)["valid"]


def test_module_entrypoint_invokes_main(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(cli, "main", lambda: called.append(True))
    runpy.run_module("pricing_mapper.__main__", run_name="__main__")
    assert called == [True]
