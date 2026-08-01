"""Atomic, hash-verified, pickle-free v1 artifact directories."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import pydantic
import sklearn
import skops
import skops.io as sio
from sklearn.base import BaseEstimator

from pricing_mapper.advisor import (
    AdvisorDecisionRecord,
    allowed_diagnostic_bin_ids,
    derived_advisor_seed,
    system_prompt,
)
from pricing_mapper.config import MapperConfig, config_toml, load_config
from pricing_mapper.domain import FIELD_ORDER, CarQuoteInput, DomainSpec, row_key, rows_json_schema
from pricing_mapper.encoding import FeatureEncoder
from pricing_mapper.exceptions import (
    ArtifactError,
    ConfigurationError,
    DomainValidationError,
    LegacyArtifactError,
)
from pricing_mapper.models import ModelKind, validate_loaded_estimator
from pricing_mapper.persistence import SampleRecord, validate_run_id

ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_FORMAT = "pricing-function-mapper/v1"
MANIFEST_NAME = "manifest.json"
REQUIRED_FILES = {
    "config.toml",
    "conformal.json",
    "dataset.csv",
    "dataset.schema.json",
    "dependencies.json",
    "domain.json",
    "evaluation.json",
    "model-card.md",
    "model.json",
    "model.skops",
    "provenance.json",
    "schema.json",
}
# Current sklearn pipelines encode two standard-library/sklearn callables that
# skops conservatively reports as unknown. This fixed allow-list is intentionally
# narrow; artifact metadata cannot expand it.
SAFE_SKOPS_TYPES = {
    "functools.partial",
    "sklearn.utils.validation.check_array",
}
DATASET_COLUMNS = [*FIELD_ORDER, "premium", "split", "source"]
DATASET_SPLITS = ("mapping", "validation", "calibration", "audit")
DEPENDENCY_NAMES = {
    "numpy",
    "pandas",
    "pydantic",
    "scikit-learn",
    "skops",
    "threadpoolctl",
    "tomli-w",
    "pricing-function-mapper",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EngineComponents(TypedDict):
    estimator: BaseEstimator
    model_kind: ModelKind
    domain: DomainSpec
    encoder: FeatureEncoder
    conformal_radius: float
    conformal_coverage: float
    model_version: str
    warnings: Sequence[str]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependencies() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in sorted(DEPENDENCY_NAMES):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name == "pricing-function-mapper":
                result[name] = "1.0.0"
            else:
                raise ArtifactError(f"runtime dependency {name!r} is not installed") from None
    return result


def _git_revision() -> str | None:
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603
            [git_executable, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if len(revision) == 40 else None


def _dataset_schema(row_count: int) -> dict[str, Any]:
    numeric_float = {"driver_age", "vehicle_value", "postcode_risk", "theft_risk"}
    numeric_integer = {
        "years_licensed",
        "vehicle_year",
        "annual_km",
        "claims_5y",
        "convictions_5y",
        "excess",
    }
    field_types: dict[str, str] = {}
    for name in FIELD_ORDER:
        if name in numeric_float:
            field_types[name] = "float64"
        elif name in numeric_integer:
            field_types[name] = "int64"
        else:
            field_types[name] = "string"
    return {
        "schema_version": 1,
        "format": "text/csv; charset=utf-8; header=present",
        "row_count": row_count,
        "columns": [
            *[{"name": name, "type": field_types[name], "nullable": False} for name in FIELD_ORDER],
            {"name": "premium", "type": "float64", "nullable": False},
            {
                "name": "split",
                "type": "enum",
                "levels": ["mapping", "validation", "calibration", "audit"],
                "nullable": False,
            },
            {"name": "source", "type": "string", "nullable": False},
        ],
        "target": "premium",
        "split_column": "split",
    }


def _write_dataset(path: Path, samples: Sequence[SampleRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for sample in samples:
            if sample.premium is None:
                raise ArtifactError("cannot export an incomplete sample")
            row = {name: sample.row[name] for name in FIELD_ORDER}
            row.update(
                {
                    "premium": sample.premium,
                    "split": sample.split,
                    "source": sample.source,
                }
            )
            writer.writerow(row)


def _model_card(
    *,
    title: str,
    model_version: str,
    model_kind: str,
    evaluation: Mapping[str, Any],
    warnings: Sequence[str],
) -> str:
    audit = evaluation.get("audit", {})
    metrics = audit.get("metrics", {}) if isinstance(audit, Mapping) else {}
    interval = evaluation.get("audit_interval", {})
    warning_text = "\n".join(f"- {item}" for item in warnings) or "- None recorded."
    return f"""# {title}

Model version: `{model_version}`

## Intended use

Offline approximation of a trusted local comprehensive-car-insurance quote
function within the exact domain recorded in `domain.json`. It is not an
underwriting decision system and must not be used outside that domain.

## Model

- Selected family: `{model_kind}`
- Point-model selection: lowest held-out validation MAE within the configured
  warm single-row p95 latency ceiling.
- Uncertainty: split-conformal interval at the coverage configured in
  `conformal.json`; RF committee spread is not exported as uncertainty.

## Independent audit

- MAE: `{metrics.get("mae", "unavailable")}`
- RMSE: `{metrics.get("rmse", "unavailable")}`
- WAPE: `{metrics.get("wape", "unavailable")}`
- R²: `{metrics.get("r2", "unavailable")}`
- Interval coverage: `{interval.get("coverage", "unavailable")}`
- Mean interval width: `{interval.get("mean_width", "unavailable")}`

Full confidence bounds and risk-slice results are in `evaluation.json`.

## Warnings

{warning_text}

## Limitations and governance

The synthetic reference provider is illustrative. A production insurer must
perform legal, actuarial, fairness, privacy, explainability, change-control,
and human-oversight reviews. This artifact does not establish regulatory
compliance, rate adequacy, or permission to use any listed feature.
"""


def export_artifact(
    *,
    output_dir: str | Path,
    run_id: str,
    config: MapperConfig,
    domain: DomainSpec,
    samples: Sequence[SampleRecord],
    estimator: BaseEstimator,
    model_kind: ModelKind,
    model_version: str,
    conformal_radius: float,
    evaluation_report: Mapping[str, Any],
    selection_report: Mapping[str, Any],
    provider_summary: Mapping[str, Any],
    optimizer_summary: Mapping[str, Any],
    warnings: Sequence[str],
    run_started_at_utc: str,
) -> Path:
    """Write a complete staging directory and atomically publish it."""

    safe_run_id = validate_run_id(run_id)
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    final_dir = base / safe_run_id
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {final_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{safe_run_id}.staging-", dir=base))
    created_at = datetime.now(UTC).isoformat()
    try:
        (staging / "config.toml").write_text(config_toml(config), encoding="utf-8")
        _write_json(staging / "domain.json", domain.to_dict())
        _write_json(staging / "schema.json", rows_json_schema())
        _write_dataset(staging / "dataset.csv", samples)
        _write_json(staging / "dataset.schema.json", _dataset_schema(len(samples)))
        _write_json(
            staging / "model.json",
            {
                "schema_version": 1,
                "model_version": model_version,
                "kind": model_kind,
                "encoding": FeatureEncoder(domain).metadata(),
                "selection": dict(selection_report),
            },
        )
        sio.dump(estimator, staging / "model.skops")
        _write_json(
            staging / "conformal.json",
            {
                "schema_version": 1,
                "method": "split_conformal_absolute_residual",
                "nominal_coverage": config.evaluation.conformal_coverage,
                "radius": conformal_radius,
                "calibration_count": config.evaluation.split_counts()["calibration"],
            },
        )
        _write_json(staging / "evaluation.json", dict(evaluation_report))
        (staging / "model-card.md").write_text(
            _model_card(
                title=config.artifact.model_card_title,
                model_version=model_version,
                model_kind=model_kind,
                evaluation=evaluation_report,
                warnings=warnings,
            ),
            encoding="utf-8",
        )
        dependencies = _dependencies()
        _write_json(staging / "dependencies.json", dependencies)
        _write_json(
            staging / "provenance.json",
            {
                "schema_version": 1,
                "run_id": safe_run_id,
                "run_started_at_utc": run_started_at_utc,
                "artifact_created_at_utc": created_at,
                "config_fingerprint": config.fingerprint,
                "provider": dict(provider_summary),
                "optimizer": dict(optimizer_summary),
                "git_revision": _git_revision(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "library_versions": {
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "pydantic": pydantic.__version__,
                    "scikit_learn": sklearn.__version__,
                    "skops": skops.__version__,
                    "threadpoolctl": importlib.metadata.version("threadpoolctl"),
                },
                "quote_payloads_logged": False,
            },
        )

        actual_files = {path.name for path in staging.iterdir() if path.is_file()}
        if actual_files != REQUIRED_FILES:
            raise ArtifactError(
                f"artifact staging set mismatch; missing={sorted(REQUIRED_FILES - actual_files)}, "
                f"unexpected={sorted(actual_files - REQUIRED_FILES)}"
            )
        hashes = {
            name: {
                "sha256": _sha256(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in sorted(REQUIRED_FILES)
        }
        _write_json(
            staging / MANIFEST_NAME,
            {
                "artifact_format": ARTIFACT_FORMAT,
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": safe_run_id,
                "model_version": model_version,
                "created_at_utc": created_at,
                "files": hashes,
            },
        )
        for path in staging.iterdir():
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, final_dir)
        parent_fd = os.open(base, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return final_dir
    except Exception:
        if (
            staging.exists()
            and staging.parent == base
            and staging.name.startswith(f".{safe_run_id}.staging-")
        ):
            shutil.rmtree(staging)
        raise


def _legacy_error(path: Path) -> LegacyArtifactError:
    return LegacyArtifactError(
        f"{path} is not a v1 artifact directory. v0 pickle/state artifacts are "
        "intentionally rejected because pickle can execute code. Re-run mapping or "
        "import validated observations with 'map run --seed-data'."
    )


def _read_json(path: Path) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactError(f"cannot read artifact JSON {path.name}: {exc}") from exc


def _validate_dataset(
    path: Path,
    *,
    metadata: Any,
    domain: DomainSpec,
    config: MapperConfig,
) -> dict[str, int]:
    """Validate the explicit CSV schema and every exported observation."""

    counts = dict.fromkeys(DATASET_SPLITS, 0)
    seen_hashes: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != DATASET_COLUMNS:
                raise ArtifactError(f"dataset.csv columns must exactly match {DATASET_COLUMNS!r}")
            for line_number, raw in enumerate(reader, start=2):
                try:
                    if None in raw or any(value is None for value in raw.values()):
                        raise ValueError("row has missing or extra columns")
                    values: dict[str, Any] = {}
                    for name in ("driver_age", "vehicle_value", "postcode_risk", "theft_risk"):
                        values[name] = float(raw[name])
                    for name in (
                        "years_licensed",
                        "vehicle_year",
                        "annual_km",
                        "claims_5y",
                        "convictions_5y",
                        "excess",
                    ):
                        values[name] = int(raw[name])
                    for name in (
                        "usage",
                        "parking",
                        "hire_car",
                        "windscreen",
                        "rating",
                    ):
                        values[name] = raw[name]
                    quote = CarQuoteInput.from_mapping(values, domain=domain)
                    premium = float(raw["premium"])
                    if not math.isfinite(premium) or premium < 0:
                        raise ValueError("premium must be finite and non-negative")
                    split = raw["split"]
                    if split not in counts:
                        raise ValueError(f"unknown split {split!r}")
                    if not raw["source"]:
                        raise ValueError("source cannot be empty")
                except (KeyError, TypeError, ValueError) as exc:
                    raise ArtifactError(f"dataset.csv row {line_number} is invalid: {exc}") from exc
                hashed = row_key(quote)
                if hashed in seen_hashes:
                    raise ArtifactError(
                        f"dataset.csv row {line_number} duplicates an earlier observation"
                    )
                seen_hashes.add(hashed)
                counts[split] += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ArtifactError(f"cannot read artifact dataset.csv: {exc}") from exc

    row_count = sum(counts.values())
    if metadata != _dataset_schema(row_count):
        raise ArtifactError("dataset.schema.json does not match dataset.csv")
    expected_evaluation = config.evaluation.split_counts()
    for split in ("validation", "calibration", "audit"):
        if counts[split] != expected_evaluation[split]:
            raise ArtifactError(
                f"dataset.csv {split} count {counts[split]} does not match "
                f"configuration count {expected_evaluation[split]}"
            )
    minimum_mapping = min(
        config.sampling.initial_size,
        config.sampling.mapping_budget,
    )
    if not minimum_mapping <= counts["mapping"] <= config.sampling.mapping_budget:
        raise ArtifactError(
            "dataset.csv mapping count is inconsistent with the configured mapping budget"
        )
    return counts


def _validate_conformal(
    raw: Any,
    *,
    config: MapperConfig,
    split_counts: Mapping[str, int],
) -> tuple[float, float]:
    expected_keys = {
        "schema_version",
        "method",
        "nominal_coverage",
        "radius",
        "calibration_count",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise ArtifactError("conformal.json has missing or unknown fields")
    try:
        schema_version = raw["schema_version"]
        if isinstance(raw["nominal_coverage"], bool) or isinstance(raw["radius"], bool):
            raise TypeError("coverage and radius must be numbers")
        coverage = float(raw["nominal_coverage"])
        radius = float(raw["radius"])
        calibration_count = raw["calibration_count"]
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"conformal.json has invalid values: {exc}") from exc
    if type(schema_version) is not int or schema_version != 1:
        raise ArtifactError("conformal.json has an unsupported schema")
    if raw["method"] != "split_conformal_absolute_residual":
        raise ArtifactError("conformal.json has an unsupported method")
    if (
        not math.isfinite(coverage)
        or coverage != config.evaluation.conformal_coverage
        or not 0.5 < coverage < 1.0
    ):
        raise ArtifactError("conformal.json coverage does not match configuration")
    if not math.isfinite(radius) or radius < 0:
        raise ArtifactError("conformal.json radius must be finite and non-negative")
    if type(calibration_count) is not int or calibration_count != split_counts["calibration"]:
        raise ArtifactError("conformal.json calibration count does not match dataset.csv")
    return radius, coverage


def validate_artifact(
    artifact_dir: str | Path,
    *,
    check_runtime: bool = True,
) -> dict[str, Any]:
    """Validate layout, hashes, schemas, safe model types, and runtime version."""

    target = Path(artifact_dir)
    if target.suffix.lower() in {".pkl", ".pickle"} or target.is_file():
        raise _legacy_error(target)
    if not target.is_dir():
        raise ArtifactError(f"artifact directory does not exist: {target}")
    manifest_path = target / MANIFEST_NAME
    if not manifest_path.is_file():
        raise _legacy_error(target)
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ArtifactError("manifest.json must contain an object")
    expected_manifest_keys = {
        "artifact_format",
        "schema_version",
        "run_id",
        "model_version",
        "created_at_utc",
        "files",
    }
    if set(manifest) != expected_manifest_keys:
        raise ArtifactError("manifest.json has missing or unknown fields")
    if (
        manifest["artifact_format"] != ARTIFACT_FORMAT
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != ARTIFACT_SCHEMA_VERSION
    ):
        raise ArtifactError(
            f"unsupported artifact format/version: {manifest.get('artifact_format')!r} "
            f"schema {manifest.get('schema_version')!r}"
        )
    if not isinstance(manifest["model_version"], str) or not manifest["model_version"].strip():
        raise ArtifactError("manifest model_version must be a non-empty string")
    if not isinstance(manifest["created_at_utc"], str) or not manifest["created_at_utc"].strip():
        raise ArtifactError("manifest created_at_utc must be a non-empty string")
    try:
        validate_run_id(manifest["run_id"])
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"manifest run_id is invalid: {exc}") from exc
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != REQUIRED_FILES:
        raise ArtifactError("manifest file set does not match the v1 artifact contract")
    actual = {path.name for path in target.iterdir()}
    if actual != REQUIRED_FILES | {MANIFEST_NAME}:
        raise ArtifactError(
            f"artifact directory has missing or unlisted files: "
            f"missing={sorted((REQUIRED_FILES | {MANIFEST_NAME}) - actual)}, "
            f"unexpected={sorted(actual - (REQUIRED_FILES | {MANIFEST_NAME}))}"
        )
    for name, expected in files.items():
        path = target / name
        if path.is_symlink() or not path.is_file():
            raise ArtifactError(f"artifact entry {name!r} must be a regular file")
        if not isinstance(expected, dict) or set(expected) != {"sha256", "size_bytes"}:
            raise ArtifactError(f"manifest hash entry for {name!r} is invalid")
        digest = expected["sha256"]
        size = expected["size_bytes"]
        if (
            not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or type(size) is not int
            or size < 0
        ):
            raise ArtifactError(f"manifest hash entry for {name!r} is invalid")
        if path.stat().st_size != size or _sha256(path) != digest:
            raise ArtifactError(f"artifact integrity check failed for {name}")

    try:
        config = load_config(target / "config.toml")
    except (ConfigurationError, OSError, ValueError) as exc:
        raise ArtifactError(f"artifact config.toml is invalid: {exc}") from exc
    model_metadata = _read_json(target / "model.json")
    conformal = _read_json(target / "conformal.json")
    domain_raw = _read_json(target / "domain.json")
    row_schema = _read_json(target / "schema.json")
    dataset_metadata = _read_json(target / "dataset.schema.json")
    evaluation = _read_json(target / "evaluation.json")
    provenance = _read_json(target / "provenance.json")
    dependencies = _read_json(target / "dependencies.json")

    try:
        domain = DomainSpec.from_dict(domain_raw)
    except (DomainValidationError, TypeError, ValueError) as exc:
        raise ArtifactError(str(exc)) from exc
    if domain != config.resolved_domain:
        raise ArtifactError("domain.json does not match the validated configuration snapshot")
    if row_schema != rows_json_schema():
        raise ArtifactError("schema.json does not match the v1 CarQuoteInput schema")
    split_counts = _validate_dataset(
        target / "dataset.csv",
        metadata=dataset_metadata,
        domain=domain,
        config=config,
    )
    radius, coverage = _validate_conformal(
        conformal,
        config=config,
        split_counts=split_counts,
    )

    expected_model_keys = {
        "schema_version",
        "model_version",
        "kind",
        "encoding",
        "selection",
    }
    if not isinstance(model_metadata, dict) or set(model_metadata) != expected_model_keys:
        raise ArtifactError("model.json has missing or unknown fields")
    if type(model_metadata["schema_version"]) is not int or model_metadata["schema_version"] != 1:
        raise ArtifactError("model.json has an unsupported schema")
    if model_metadata.get("model_version") != manifest["model_version"]:
        raise ArtifactError("model version differs between manifest.json and model.json")
    if model_metadata.get("kind") not in {"hist_gradient_boosting", "extra_trees"}:
        raise ArtifactError("model.json has an unsupported estimator kind")
    try:
        FeatureEncoder.from_metadata(model_metadata["encoding"], domain)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"model.json encoding metadata is invalid: {exc}") from exc
    if not isinstance(model_metadata["selection"], dict):
        raise ArtifactError("model.json selection must be an object")

    if not isinstance(dependencies, dict) or set(dependencies) != DEPENDENCY_NAMES:
        raise ArtifactError("dependencies.json has missing or unknown dependencies")
    if any(not isinstance(value, str) or not value for value in dependencies.values()):
        raise ArtifactError("dependencies.json versions must be non-empty strings")
    if check_runtime and dependencies["scikit-learn"] != sklearn.__version__:
        raise ArtifactError(
            "artifact scikit-learn version mismatch: saved with "
            f"{dependencies['scikit-learn']}, running {sklearn.__version__}; "
            "cross-version sklearn model loading is unsupported"
        )

    expected_provenance_keys = {
        "schema_version",
        "run_id",
        "run_started_at_utc",
        "artifact_created_at_utc",
        "config_fingerprint",
        "provider",
        "optimizer",
        "git_revision",
        "python",
        "platform",
        "library_versions",
        "quote_payloads_logged",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance_keys:
        raise ArtifactError("provenance.json has missing or unknown fields")
    if type(provenance["schema_version"]) is not int or provenance["schema_version"] != 1:
        raise ArtifactError("provenance.json has an unsupported schema")
    if (
        provenance["run_id"] != manifest["run_id"]
        or provenance["artifact_created_at_utc"] != manifest["created_at_utc"]
    ):
        raise ArtifactError("provenance identifiers differ from manifest.json")
    if provenance["config_fingerprint"] != config.fingerprint:
        raise ArtifactError("provenance config fingerprint does not match config.toml")
    if provenance["quote_payloads_logged"] is not False:
        raise ArtifactError("provenance must attest that quote payloads were not logged")
    provider = provenance["provider"]
    expected_provider_keys = {
        "identity",
        "attempts",
        "successes",
        "retryable_failures",
        "permanent_failures",
        "total_latency_ms",
    }
    if not isinstance(provider, dict) or set(provider) != expected_provider_keys:
        raise ArtifactError("provenance provider summary is invalid")
    outcome_counts = (
        provider["successes"],
        provider["retryable_failures"],
        provider["permanent_failures"],
    )
    if (
        not isinstance(provider["identity"], str)
        or not provider["identity"]
        or any(type(value) is not int or value < 0 for value in outcome_counts)
        or type(provider["attempts"]) is not int
        or provider["attempts"] != sum(outcome_counts)
    ):
        raise ArtifactError("provenance provider counts or identity are invalid")
    try:
        if isinstance(provider["total_latency_ms"], bool):
            raise TypeError("latency must be numeric")
        provider_latency = float(provider["total_latency_ms"])
    except (TypeError, ValueError) as exc:
        raise ArtifactError("provenance provider latency is invalid") from exc
    if not math.isfinite(provider_latency) or provider_latency < 0:
        raise ArtifactError("provenance provider latency is invalid")

    expected_evaluation_keys = {
        "schema_version",
        "evaluation_design",
        "validation_selection",
        "calibration",
        "audit",
        "audit_interval",
        "conformal",
        "latency",
        "advisor",
        "promotion_gates",
        "early_stopped",
        "stop_reason",
        "batch_history",
        "warnings",
    }
    if not isinstance(evaluation, dict) or set(evaluation) != expected_evaluation_keys:
        raise ArtifactError("evaluation.json has missing or unknown fields")
    optimizer = provenance["optimizer"]
    if optimizer != evaluation["advisor"]:
        raise ArtifactError("provenance optimizer summary differs from evaluation.json")
    if type(evaluation["schema_version"]) is not int or evaluation["schema_version"] != 1:
        raise ArtifactError("evaluation.json has an unsupported schema")
    design = evaluation["evaluation_design"]
    design_counts = design.get("split_counts") if isinstance(design, dict) else None
    if (
        not isinstance(design, dict)
        or not isinstance(design_counts, dict)
        or any(type(value) is not int for value in design_counts.values())
        or design_counts != config.evaluation.split_counts()
        or design.get("final_unbiased_data") != ["audit"]
        or design.get("conformal_data") != ["calibration"]
        or design.get("calibration_or_audit_used_for_acquisition") is not False
        or design.get("calibration_or_audit_used_for_tuning_or_early_stopping") is not False
    ):
        raise ArtifactError("evaluation.json holdout design is inconsistent")
    for section, split in (
        ("validation_selection", "validation"),
        ("calibration", "calibration"),
        ("audit", "audit"),
    ):
        report = evaluation[section]
        if (
            not isinstance(report, dict)
            or type(report.get("count")) is not int
            or report.get("count") != split_counts[split]
        ):
            raise ArtifactError(f"evaluation.json {section} count does not match dataset.csv")
    audit_interval = evaluation["audit_interval"]
    if (
        not isinstance(audit_interval, dict)
        or type(audit_interval.get("count")) is not int
        or audit_interval.get("count") != split_counts["audit"]
    ):
        raise ArtifactError("evaluation.json audit interval count does not match dataset.csv")
    try:
        if isinstance(audit_interval["coverage"], bool):
            raise TypeError("coverage must be numeric")
        audit_coverage = float(audit_interval["coverage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("evaluation.json audit coverage is invalid") from exc
    if not math.isfinite(audit_coverage) or not 0.0 <= audit_coverage <= 1.0:
        raise ArtifactError("evaluation.json audit coverage is invalid")
    evaluation_conformal = evaluation["conformal"]
    if evaluation_conformal != {
        "nominal_coverage": coverage,
        "radius": radius,
    }:
        raise ArtifactError("evaluation.json conformal state differs from conformal.json")
    latency = evaluation["latency"]
    if not isinstance(latency, dict) or set(latency) != {
        "warm_single_row_p95_ms",
        "ceiling_ms",
        "passed",
    }:
        raise ArtifactError("evaluation.json latency report is invalid")
    try:
        if isinstance(latency["warm_single_row_p95_ms"], bool) or isinstance(
            latency["ceiling_ms"], bool
        ):
            raise TypeError("latency must be numeric")
        measured_latency = float(latency["warm_single_row_p95_ms"])
        latency_ceiling = float(latency["ceiling_ms"])
    except (TypeError, ValueError) as exc:
        raise ArtifactError("evaluation.json latency values are invalid") from exc
    if (
        not math.isfinite(measured_latency)
        or measured_latency < 0
        or not math.isfinite(latency_ceiling)
        or latency_ceiling != config.model.max_p95_latency_ms
    ):
        raise ArtifactError("evaluation.json latency values are invalid")
    latency_passed = measured_latency <= latency_ceiling
    if type(latency["passed"]) is not bool or latency["passed"] != latency_passed:
        raise ArtifactError("evaluation.json latency gate is inconsistent")
    coverage_passed = (
        config.evaluation.minimum_audit_coverage
        <= audit_coverage
        <= config.evaluation.maximum_audit_coverage
    )
    advisor = evaluation["advisor"]
    if not isinstance(advisor, dict):
        raise ArtifactError("evaluation.json advisor report is invalid")
    expected_advisor_keys = {
        "enabled",
        "prompt_version",
        "runtime",
        "resource_mode",
        "decision_count",
        "total_latency_ms",
        "maximum_response_latency_ms",
        "latency_ceiling_ms",
        "latency_passed",
        "installed_model_bytes",
        "maximum_resident_model_bytes",
        "maximum_vram_bytes",
        "memory_ceiling_bytes",
        "memory_passed",
        "data_shared",
    }
    if set(advisor) != expected_advisor_keys:
        raise ArtifactError("evaluation.json advisor report is invalid")
    enabled = config.optimizer.ollama is not None
    if type(advisor["enabled"]) is not bool or advisor["enabled"] != enabled:
        raise ArtifactError("evaluation.json advisor enablement differs from config.toml")
    numeric_advisor_values = (
        advisor["total_latency_ms"],
        advisor["maximum_response_latency_ms"],
        advisor["latency_ceiling_ms"],
    )
    try:
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in numeric_advisor_values
        ):
            raise TypeError("advisor timing must be numeric")
        parsed_advisor_values = [float(value) for value in numeric_advisor_values]
    except (TypeError, ValueError) as exc:
        raise ArtifactError("evaluation.json advisor timing is invalid") from exc
    if (
        any(not math.isfinite(value) or value < 0.0 for value in parsed_advisor_values)
        or parsed_advisor_values[0] < parsed_advisor_values[1]
        or parsed_advisor_values[2] != 60_000.0
        or type(advisor["latency_passed"]) is not bool
        or advisor["latency_passed"] != (parsed_advisor_values[1] <= 60_000.0)
        or type(advisor["memory_passed"]) is not bool
        or type(advisor["decision_count"]) is not int
        or advisor["decision_count"] < 0
        or type(advisor["installed_model_bytes"]) is not int
        or advisor["installed_model_bytes"] < 0
        or type(advisor["maximum_resident_model_bytes"]) is not int
        or advisor["maximum_resident_model_bytes"] < 0
        or type(advisor["maximum_vram_bytes"]) is not int
        or advisor["maximum_vram_bytes"] < 0
        or type(advisor["memory_ceiling_bytes"]) is not int
        or advisor["memory_ceiling_bytes"] != 8 * 1024**3
        or advisor["memory_passed"]
        != (advisor["maximum_resident_model_bytes"] <= advisor["memory_ceiling_bytes"])
    ):
        raise ArtifactError("evaluation.json advisor resource report is invalid")
    expected_data_shared = {
        "aggregate_diagnostics_only": True,
        "raw_premiums": False,
        "individual_quote_rows": False,
        "calibration_data": False,
        "audit_data": False,
    }
    if advisor["data_shared"] != expected_data_shared:
        raise ArtifactError("evaluation.json advisor data-sharing attestation is invalid")
    if enabled:
        ollama = config.optimizer.ollama
        if ollama is None:
            raise AssertionError("enabled advisor lacks validated config")
        runtime = advisor["runtime"]
        expected_runtime_keys = {
            "ollama_version",
            "model",
            "digest",
            "quantization_level",
            "model_size_bytes",
            "resource_mode",
        }
        if (
            advisor["prompt_version"] != ollama.prompt_version
            or advisor["resource_mode"] != ollama.resource_mode
            or not isinstance(runtime, dict)
            or set(runtime) != expected_runtime_keys
            or not isinstance(runtime.get("ollama_version"), str)
            or not runtime.get("ollama_version")
            or runtime.get("model") != ollama.model
            or runtime.get("digest") != ollama.required_digest
            or runtime.get("quantization_level") != "Q4_K_M"
            or runtime.get("resource_mode") != ollama.resource_mode
            or type(runtime.get("model_size_bytes")) is not int
            or runtime.get("model_size_bytes", 0) <= 0
            or advisor["installed_model_bytes"] != runtime.get("model_size_bytes")
        ):
            raise ArtifactError("evaluation.json advisor runtime differs from config.toml")
    elif (
        advisor["prompt_version"] is not None
        or advisor["runtime"] is not None
        or advisor["resource_mode"] is not None
        or advisor["decision_count"] != 0
        or advisor["installed_model_bytes"] != 0
        or advisor["maximum_resident_model_bytes"] != 0
        or advisor["maximum_vram_bytes"] != 0
        or float(advisor["total_latency_ms"]) != 0.0
        or float(advisor["maximum_response_latency_ms"]) != 0.0
        or advisor["latency_passed"] is not True
        or advisor["memory_passed"] is not True
    ):
        raise ArtifactError("evaluation.json contains advisor state while Ollama is disabled")
    gates = evaluation["promotion_gates"]
    expected_gates = {
        "audit_coverage_passed": coverage_passed,
        "latency_passed": latency_passed,
        "advisor_resources_passed": advisor["latency_passed"] and advisor["memory_passed"],
        "eligible": (
            coverage_passed
            and latency_passed
            and advisor["latency_passed"]
            and advisor["memory_passed"]
        ),
    }
    if (
        not isinstance(gates, dict)
        or any(type(value) is not bool for value in gates.values())
        or gates != expected_gates
    ):
        raise ArtifactError("evaluation.json promotion gates are inconsistent")
    if (
        type(evaluation["early_stopped"]) is not bool
        or not isinstance(evaluation["batch_history"], list)
        or not isinstance(evaluation["warnings"], list)
        or any(not isinstance(item, str) for item in evaluation["warnings"])
    ):
        raise ArtifactError("evaluation.json run state or warnings are invalid")
    advisor_records: list[AdvisorDecisionRecord] = []
    allowed_bins = allowed_diagnostic_bin_ids(domain)
    for item in evaluation["batch_history"]:
        if not isinstance(item, dict) or set(item) != {
            "batch_id",
            "status",
            "advisor",
            "metrics",
        }:
            raise ArtifactError("evaluation.json batch history is invalid")
        if type(item["batch_id"]) is not int or item["batch_id"] < 0:
            raise ArtifactError("evaluation.json batch history is invalid")
        raw_decision = item["advisor"]
        if raw_decision is None:
            continue
        try:
            decision_record = AdvisorDecisionRecord.model_validate(raw_decision, strict=True)
        except ValueError as exc:
            raise ArtifactError("evaluation.json advisor decision is invalid") from exc
        expected_prompt_hash = (
            "sha256:"
            + hashlib.sha256(system_prompt(decision_record.prompt_version).encode()).hexdigest()
        )
        if (
            decision_record.batch_id != item["batch_id"]
            or decision_record.generation.seed
            != derived_advisor_seed(config.sampling.seed, item["batch_id"])
            or decision_record.prompt_hash != expected_prompt_hash
            or any(
                boost.bin_id not in allowed_bins for boost in decision_record.response.bin_boosts
            )
        ):
            raise ArtifactError("evaluation.json advisor decision is invalid")
        advisor_records.append(decision_record)
    if len(advisor_records) != advisor["decision_count"]:
        raise ArtifactError("evaluation.json advisor decision count is inconsistent")
    if enabled:
        decision_total = sum(item.total_latency_ms for item in advisor_records)
        decision_maximum = max(
            (latency for item in advisor_records for latency in item.attempt_latencies_ms),
            default=0.0,
        )
        decision_resident = max(
            (item.memory.resident_size_bytes for item in advisor_records),
            default=0,
        )
        decision_vram = max(
            (item.memory.vram_size_bytes for item in advisor_records),
            default=0,
        )
        if (
            any(
                item.runtime.model_dump(mode="json") != advisor["runtime"]
                for item in advisor_records
            )
            or not math.isclose(
                decision_total,
                float(advisor["total_latency_ms"]),
                rel_tol=0.0,
                abs_tol=0.00001,
            )
            or not math.isclose(
                decision_maximum,
                float(advisor["maximum_response_latency_ms"]),
                rel_tol=0.0,
                abs_tol=0.00001,
            )
            or decision_resident != advisor["maximum_resident_model_bytes"]
            or decision_vram != advisor["maximum_vram_bytes"]
        ):
            raise ArtifactError("evaluation.json advisor provenance is inconsistent")

    try:
        untrusted = sio.get_untrusted_types(file=target / "model.skops")
    except Exception as exc:
        raise ArtifactError(f"model.skops cannot be inspected safely: {exc}") from exc
    unexpected_types = set(untrusted) - SAFE_SKOPS_TYPES
    if unexpected_types:
        raise ArtifactError(
            "model.skops contains untrusted or non-sklearn types and will not be loaded: "
            f"{sorted(unexpected_types)}"
        )
    if check_runtime:
        trusted_types = sorted(set(untrusted) & SAFE_SKOPS_TYPES)
        try:
            inspected_estimator = sio.load(
                target / "model.skops",
                trusted=trusted_types,
            )
            validate_loaded_estimator(inspected_estimator, model_metadata["kind"])
        except Exception as exc:
            raise ArtifactError(f"model.skops estimator contract is invalid: {exc}") from exc
    return {
        "valid": True,
        "path": str(target),
        "run_id": manifest["run_id"],
        "model_version": manifest["model_version"],
        "model_kind": model_metadata["kind"],
        "files_verified": len(REQUIRED_FILES),
        "dataset_rows": sum(split_counts.values()),
        "dataset_split_counts": split_counts,
        "config_fingerprint": config.fingerprint,
        "provider_identity": provider["identity"],
        "scikit_learn_saved": dependencies["scikit-learn"],
        "scikit_learn_runtime": sklearn.__version__,
        "runtime_compatible": dependencies["scikit-learn"] == sklearn.__version__,
        "untrusted_model_types": sorted(unexpected_types),
        "explicitly_allowed_skops_types": sorted(set(untrusted) & SAFE_SKOPS_TYPES),
    }


def load_artifact_components(artifact_dir: str | Path) -> EngineComponents:
    target = Path(artifact_dir)
    validate_artifact(target, check_runtime=True)
    domain_raw = _read_json(target / "domain.json")
    model_metadata = _read_json(target / "model.json")
    conformal = _read_json(target / "conformal.json")
    evaluation = _read_json(target / "evaluation.json")
    if not all(
        isinstance(value, dict) for value in (domain_raw, model_metadata, conformal, evaluation)
    ):
        raise ArtifactError("artifact component JSON must contain objects")
    domain = DomainSpec.from_dict(domain_raw)
    try:
        encoder = FeatureEncoder.from_metadata(model_metadata["encoding"], domain)
        kind: ModelKind = model_metadata["kind"]
        reported_types = sio.get_untrusted_types(file=target / "model.skops")
        trusted_types = sorted(set(reported_types) & SAFE_SKOPS_TYPES)
        estimator = sio.load(target / "model.skops", trusted=trusted_types)
        fitted = validate_loaded_estimator(estimator, kind)
        radius = float(conformal["radius"])
        coverage = float(conformal["nominal_coverage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"artifact model metadata is invalid: {exc}") from exc
    warnings_raw = evaluation.get("warnings", [])
    if not isinstance(warnings_raw, list) or any(
        not isinstance(item, str) for item in warnings_raw
    ):
        raise ArtifactError("evaluation warnings must be a list of strings")
    return {
        "estimator": fitted,
        "model_kind": kind,
        "domain": domain,
        "encoder": encoder,
        "conformal_radius": radius,
        "conformal_coverage": coverage,
        "model_version": str(model_metadata["model_version"]),
        "warnings": tuple(warnings_raw),
    }


def artifact_config(artifact_dir: str | Path) -> MapperConfig:
    target = Path(artifact_dir)
    validate_artifact(target, check_runtime=False)
    return load_config(target / "config.toml")
