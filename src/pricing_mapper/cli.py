"""Subcommand-only offline CLI for pricing-mapper v1."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pricing_mapper.artifact import validate_artifact
from pricing_mapper.benchmark import run_benchmark
from pricing_mapper.config import config_json_schema, load_config
from pricing_mapper.domain import (
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
    FIELD_ORDER,
    INTEGER_FIELDS,
)
from pricing_mapper.engine import PricingEngine
from pricing_mapper.evaluation import interval_report, regression_report
from pricing_mapper.exceptions import (
    ArtifactError,
    ConfigurationError,
    DomainValidationError,
    LegacyArtifactError,
    PersistenceError,
    ProviderError,
)
from pricing_mapper.models import measure_single_row_latency
from pricing_mapper.orchestration import MappingRun, _parse_seed_number
from pricing_mapper.provider import ProviderExecutor, resolve_provider

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_RUN = 3
EXIT_ARTIFACT = 4
EXIT_GATE = 5
LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pricing-mapper",
        description="Offline comprehensive-car-insurance pricing mapper v1",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        type=str.upper,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("config", help="configuration operations")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_validate = config_commands.add_parser("validate", help="validate v1 TOML")
    config_validate.add_argument("--config", required=True)
    config_validate.add_argument("--schema-output")

    mapping = commands.add_parser("map", help="mapping run operations")
    mapping_commands = mapping.add_subparsers(dest="map_command", required=True)
    map_run = mapping_commands.add_parser("run", help="start a new mapping run")
    map_run.add_argument("--config", required=True)
    map_run.add_argument("--run-id")
    map_run.add_argument("--seed-data")
    map_resume = mapping_commands.add_parser("resume", help="resume a v1 SQLite run")
    map_resume.add_argument("--config", required=True)
    map_resume.add_argument("--run-id", required=True)
    map_resume.add_argument("--seed-data")

    model = commands.add_parser("model", help="model diagnostics")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    evaluate = model_commands.add_parser(
        "evaluate",
        help="recompute independent audit metrics from an artifact",
    )
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--output")
    evaluate.add_argument("--require-gates", action="store_true")

    predict = commands.add_parser("predict", help="offline prediction")
    predict_commands = predict.add_subparsers(dest="predict_command", required=True)
    predict_row = predict_commands.add_parser("row", help="predict one JSON row")
    predict_row.add_argument("--artifact", required=True)
    row_source = predict_row.add_mutually_exclusive_group(required=True)
    row_source.add_argument("--json", dest="row_json")
    row_source.add_argument("--input")
    predict_batch = predict_commands.add_parser("batch", help="predict a CSV batch")
    predict_batch.add_argument("--artifact", required=True)
    predict_batch.add_argument("--input", required=True)
    predict_batch.add_argument("--output", required=True)

    artifact = commands.add_parser("artifact", help="artifact operations")
    artifact_commands = artifact.add_subparsers(dest="artifact_command", required=True)
    inspect = artifact_commands.add_parser("inspect", help="verify a v1 artifact")
    inspect.add_argument("--artifact", required=True)

    benchmark = commands.add_parser("benchmark", help="run five-seed release benchmark")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--baseline")
    benchmark.add_argument("--enforce", action="store_true")
    return parser


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _mapping_from_csv(raw: dict[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in CONTINUOUS_FIELDS:
        values[name] = _parse_seed_number(raw[name], name, integer=False)
    for name in INTEGER_FIELDS:
        values[name] = _parse_seed_number(raw[name], name, integer=True)
    for name in CATEGORICAL_FIELDS:
        values[name] = raw[name]
    return values


def _read_prediction_csv(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELD_ORDER):
            raise ValueError(f"prediction CSV columns must exactly match {list(FIELD_ORDER)!r}")
        rows: list[dict[str, Any]] = []
        for line_number, raw in enumerate(reader, start=2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"prediction CSV row {line_number} has missing or extra columns")
            rows.append(_mapping_from_csv(raw))
        return rows


def _read_audit_dataset(path: Path) -> tuple[list[dict[str, Any]], list[float]]:
    rows: list[dict[str, Any]] = []
    targets: list[float] = []
    expected = [*FIELD_ORDER, "premium", "split", "source"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected:
            raise ArtifactError("artifact dataset columns do not match dataset.schema.json")
        for line_number, raw in enumerate(reader, start=2):
            if None in raw or any(value is None for value in raw.values()):
                raise ArtifactError(
                    f"artifact dataset row {line_number} has missing or extra columns"
                )
            if raw["split"] != "audit":
                continue
            rows.append(_mapping_from_csv(raw))
            targets.append(float(_parse_seed_number(raw["premium"], "premium", integer=False)))
    if not rows:
        raise ArtifactError("artifact dataset contains no audit rows")
    return rows, targets


def _config_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    descriptor = resolve_provider(config.provider.callable)
    ProviderExecutor(descriptor, config.provider)
    schema = config_json_schema()
    if args.schema_output:
        _atomic_json(args.schema_output, schema)
    _print_json(
        {
            "valid": True,
            "config_version": config.config_version,
            "fingerprint": config.fingerprint,
            "provider_identity": descriptor.identity,
            "evaluation_split_counts": config.evaluation.split_counts(),
            "resolved_domain": config.resolved_domain.to_dict(),
        }
    )
    return EXIT_OK


def _map(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run = MappingRun(config, run_id=args.run_id)
    result = run.run(
        resume=args.map_command == "resume",
        seed_data=args.seed_data,
    )
    _print_json(
        {
            "run_id": result.run_id,
            "artifact_dir": str(result.artifact_dir),
            "state_database": str(result.state_database),
            "mapping_samples": result.mapping_samples,
            "evaluation_samples": result.evaluation_samples,
            "early_stopped": result.early_stopped,
            "promotion_eligible": result.evaluation_report["promotion_gates"]["eligible"],
        }
    )
    return EXIT_OK


def _evaluate(args: argparse.Namespace) -> int:
    validate_artifact(args.artifact)
    engine = PricingEngine.load(args.artifact)
    config = load_config(Path(args.artifact) / "config.toml")
    rows, targets = _read_audit_dataset(Path(args.artifact) / "dataset.csv")
    predictions = engine.predict_batch(rows)
    premiums = [prediction.premium for prediction in predictions]
    report = regression_report(
        rows,
        targets,
        premiums,
        evaluation=config.evaluation,
        seed=config.sampling.seed + 4_003,
    )
    intervals = interval_report(
        targets,
        [prediction.lower for prediction in predictions],
        [prediction.upper for prediction in predictions],
    )
    coverage = float(intervals["coverage"])
    coverage_passed = (
        config.evaluation.minimum_audit_coverage
        <= coverage
        <= config.evaluation.maximum_audit_coverage
    )
    latency = measure_single_row_latency(
        engine.estimator,
        engine.model_kind,
        engine.encoder,
        rows[0],
        config.model.latency_repetitions,
    )
    latency_passed = latency <= config.model.max_p95_latency_ms
    gates_passed = coverage_passed and latency_passed
    payload = {
        "artifact": str(args.artifact),
        "model_version": engine.model_version,
        "audit": report,
        "audit_interval": intervals,
        "latency": {
            "warm_single_row_p95_ms": latency,
            "ceiling_ms": config.model.max_p95_latency_ms,
            "passed": latency_passed,
        },
        "gates": {
            "audit_coverage_passed": coverage_passed,
            "latency_passed": latency_passed,
            "passed": gates_passed,
        },
    }
    if args.output:
        _atomic_json(args.output, payload)
    _print_json(payload)
    if args.require_gates and not gates_passed:
        return EXIT_GATE
    return EXIT_OK


def _predict_row(args: argparse.Namespace) -> int:
    engine = PricingEngine.load(args.artifact)
    if args.input:
        raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        raw = json.loads(args.row_json)
    if not isinstance(raw, dict):
        raise ValueError("single-row prediction input must be a JSON object")
    prediction = engine.predict(raw)
    _print_json(prediction.model_dump(mode="json"))
    return EXIT_OK


def _predict_batch(args: argparse.Namespace) -> int:
    rows = _read_prediction_csv(args.input)
    engine = PricingEngine.load(args.artifact)
    predictions = engine.predict_batch(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    columns = [*FIELD_ORDER, "premium", "lower", "upper", "model_version", "warnings"]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for raw, prediction in zip(rows, predictions, strict=True):
            row = dict(raw)
            row.update(
                {
                    "premium": prediction.premium,
                    "lower": prediction.lower,
                    "upper": prediction.upper,
                    "model_version": prediction.model_version,
                    "warnings": json.dumps(prediction.warnings),
                }
            )
            writer.writerow(row)
    temporary.replace(output)
    _print_json({"rows": len(rows), "output": str(output)})
    return EXIT_OK


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "config" and args.config_command == "validate":
        return _config_validate(args)
    if args.command == "map":
        return _map(args)
    if args.command == "model" and args.model_command == "evaluate":
        return _evaluate(args)
    if args.command == "predict" and args.predict_command == "row":
        return _predict_row(args)
    if args.command == "predict" and args.predict_command == "batch":
        return _predict_batch(args)
    if args.command == "artifact" and args.artifact_command == "inspect":
        _print_json(validate_artifact(args.artifact))
        return EXIT_OK
    if args.command == "benchmark":
        config = load_config(args.config)
        payload = run_benchmark(
            config,
            args.output,
            baseline=args.baseline,
        )
        _print_json(payload["gates"])
        if args.enforce and not payload["gates"]["passed"]:
            return EXIT_GATE
        return EXIT_OK
    raise AssertionError("argparse accepted an unhandled command")


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Parse and execute the CLI, returning a documented process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)
    try:
        return _dispatch(args)
    except (
        ConfigurationError,
        DomainValidationError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        logging.getLogger("pricing_mapper").error("%s", exc)
        return EXIT_INPUT
    except LegacyArtifactError as exc:
        logging.getLogger("pricing_mapper").error("%s", exc)
        return EXIT_ARTIFACT
    except ArtifactError as exc:
        logging.getLogger("pricing_mapper").error("%s", exc)
        return EXIT_ARTIFACT
    except (PersistenceError, ProviderError, RuntimeError) as exc:
        logging.getLogger("pricing_mapper").error("%s", exc)
        return EXIT_RUN


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
