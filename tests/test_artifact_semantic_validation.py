from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from pricing_mapper.artifact import (
    DATASET_COLUMNS,
    _dataset_schema,
    validate_artifact,
)
from pricing_mapper.exceptions import ArtifactError


def _rewrite_manifest_hash(artifact: Path, name: str) -> None:
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = (artifact / name).read_bytes()
    manifest["files"][name] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _set_path(raw: Any, path: tuple[str, ...], value: Any) -> Any:
    if not path:
        return value
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return raw


def _delete_path(raw: Any, path: tuple[str, ...]) -> Any:
    target = raw
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    return raw


def _clone(base: Path, root: Path, label: str) -> Path:
    target = root / label
    shutil.copytree(base, target)
    return target


def _assert_invalid(artifact: Path, expected: str, label: str) -> None:
    try:
        validate_artifact(artifact)
    except ArtifactError as exc:
        assert expected in str(exc), f"{label}: {exc}"
    else:
        pytest.fail(f"{label}: invalid artifact was accepted")


def test_manifest_layout_and_hash_contract_failures(
    completed_run: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    _, result = completed_run
    base = result.artifact_dir

    replacements: list[tuple[str, Any, str]] = [
        ("not-object", [], "object"),
        ("unknown-field", {"unknown": True}, "missing or unknown"),
    ]
    for label, replacement, expected in replacements:
        artifact = _clone(base, tmp_path, label)
        if label == "unknown-field":
            raw = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
            raw["unknown"] = True
            replacement = raw
        (artifact / "manifest.json").write_text(
            json.dumps(replacement) + "\n",
            encoding="utf-8",
        )
        _assert_invalid(artifact, expected, label)

    mutations: list[tuple[str, tuple[str, ...], Any, str]] = [
        ("version", ("schema_version",), 2, "unsupported"),
        ("empty-model-version", ("model_version",), "", "model_version"),
        ("empty-created", ("created_at_utc",), "", "created_at_utc"),
        ("bad-run-id", ("run_id",), "../bad", "run_id"),
        ("bad-file-set", ("files",), [], "file set"),
        (
            "bad-hash-object",
            ("files", "domain.json"),
            {},
            "hash entry",
        ),
        (
            "bad-digest",
            ("files", "domain.json", "sha256"),
            "not-a-digest",
            "hash entry",
        ),
        (
            "bad-size",
            ("files", "domain.json", "size_bytes"),
            True,
            "hash entry",
        ),
    ]
    for label, path, value, expected in mutations:
        artifact = _clone(base, tmp_path, label)
        manifest_path = artifact / "manifest.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        _set_path(raw, path, value)
        manifest_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
        _assert_invalid(artifact, expected, label)

    extra = _clone(base, tmp_path, "extra-file")
    (extra / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    _assert_invalid(extra, "unlisted", "extra-file")

    symlink = _clone(base, tmp_path, "symlink")
    domain = symlink / "domain.json"
    domain.unlink()
    domain.symlink_to(base / "domain.json")
    _assert_invalid(symlink, "regular file", "symlink")


def test_semantically_inconsistent_json_is_rejected(
    completed_run: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    _, result = completed_run
    base = result.artifact_dir
    cases: list[tuple[str, str, tuple[str, ...], Any, str, bool]] = [
        ("domain-version", "domain.json", ("schema_version",), 2, "domain schema", False),
        (
            "domain-mismatch",
            "domain.json",
            ("numeric", "vehicle_value", "high"),
            199_000.0,
            "configuration snapshot",
            False,
        ),
        ("row-schema", "schema.json", (), {}, "CarQuoteInput schema", False),
        (
            "dataset-metadata",
            "dataset.schema.json",
            ("row_count",),
            0,
            "does not match",
            False,
        ),
        ("conformal-fields", "conformal.json", ("method",), None, "missing or unknown", True),
        ("conformal-values", "conformal.json", ("radius",), [], "invalid values", False),
        ("conformal-version", "conformal.json", ("schema_version",), 2, "schema", False),
        ("conformal-method", "conformal.json", ("method",), "other", "method", False),
        ("conformal-coverage", "conformal.json", ("nominal_coverage",), 0.8, "coverage", False),
        ("conformal-radius", "conformal.json", ("radius",), -1.0, "radius", False),
        (
            "conformal-radius-boolean",
            "conformal.json",
            ("radius",),
            False,
            "invalid values",
            False,
        ),
        (
            "conformal-count",
            "conformal.json",
            ("calibration_count",),
            999,
            "calibration count",
            False,
        ),
        ("model-fields", "model.json", ("selection",), None, "missing or unknown", True),
        ("model-version-schema", "model.json", ("schema_version",), 2, "schema", False),
        ("model-version", "model.json", ("model_version",), "other", "model version", False),
        ("model-kind", "model.json", ("kind",), "unknown", "estimator kind", False),
        ("model-encoding", "model.json", ("encoding",), {}, "encoding metadata", False),
        ("model-selection", "model.json", ("selection",), [], "selection", False),
        (
            "dependency-fields",
            "dependencies.json",
            ("numpy",),
            None,
            "dependencies",
            True,
        ),
        (
            "dependency-version",
            "dependencies.json",
            ("numpy",),
            "",
            "non-empty",
            False,
        ),
        (
            "provenance-fields",
            "provenance.json",
            ("platform",),
            None,
            "missing or unknown",
            True,
        ),
        (
            "provenance-version",
            "provenance.json",
            ("schema_version",),
            2,
            "schema",
            False,
        ),
        (
            "provenance-id",
            "provenance.json",
            ("run_id",),
            "other",
            "identifiers",
            False,
        ),
        (
            "provenance-fingerprint",
            "provenance.json",
            ("config_fingerprint",),
            "0" * 64,
            "fingerprint",
            False,
        ),
        (
            "provenance-payload-log",
            "provenance.json",
            ("quote_payloads_logged",),
            True,
            "payloads",
            False,
        ),
        (
            "provider-fields",
            "provenance.json",
            ("provider", "identity"),
            None,
            "provider summary",
            True,
        ),
        (
            "provider-counts",
            "provenance.json",
            ("provider", "attempts"),
            999,
            "counts",
            False,
        ),
        (
            "provider-latency-type",
            "provenance.json",
            ("provider", "total_latency_ms"),
            [],
            "latency",
            False,
        ),
        (
            "provider-latency-negative",
            "provenance.json",
            ("provider", "total_latency_ms"),
            -1.0,
            "latency",
            False,
        ),
        (
            "evaluation-fields",
            "evaluation.json",
            ("warnings",),
            None,
            "missing or unknown",
            True,
        ),
        (
            "evaluation-version",
            "evaluation.json",
            ("schema_version",),
            2,
            "schema",
            False,
        ),
        (
            "evaluation-design",
            "evaluation.json",
            ("evaluation_design", "final_unbiased_data"),
            ["validation"],
            "holdout design",
            False,
        ),
        (
            "evaluation-count",
            "evaluation.json",
            ("audit", "count"),
            0,
            "count",
            False,
        ),
        (
            "interval-count",
            "evaluation.json",
            ("audit_interval", "count"),
            0,
            "interval count",
            False,
        ),
        (
            "coverage-type",
            "evaluation.json",
            ("audit_interval", "coverage"),
            [],
            "coverage",
            False,
        ),
        (
            "coverage-range",
            "evaluation.json",
            ("audit_interval", "coverage"),
            2.0,
            "coverage",
            False,
        ),
        (
            "evaluation-conformal",
            "evaluation.json",
            ("conformal", "radius"),
            -1.0,
            "conformal state",
            False,
        ),
        (
            "latency-shape",
            "evaluation.json",
            ("latency", "passed"),
            None,
            "latency report",
            True,
        ),
        (
            "latency-type",
            "evaluation.json",
            ("latency", "warm_single_row_p95_ms"),
            [],
            "latency values",
            False,
        ),
        (
            "latency-ceiling",
            "evaluation.json",
            ("latency", "ceiling_ms"),
            99.0,
            "latency values",
            False,
        ),
        (
            "latency-gate",
            "evaluation.json",
            ("latency", "passed"),
            "yes",
            "latency gate",
            False,
        ),
        (
            "promotion-gates",
            "evaluation.json",
            ("promotion_gates", "eligible"),
            "yes",
            "promotion gates",
            False,
        ),
        (
            "evaluation-state",
            "evaluation.json",
            ("early_stopped",),
            "no",
            "run state",
            False,
        ),
    ]

    for label, filename, path, value, expected, delete in cases:
        artifact = _clone(base, tmp_path, label)
        target = artifact / filename
        raw = json.loads(target.read_text(encoding="utf-8"))
        if delete:
            _delete_path(raw, path)
        else:
            raw = _set_path(raw, path, value)
        target.write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _rewrite_manifest_hash(artifact, filename)
        _assert_invalid(artifact, expected, label)


def test_strict_json_and_invalid_config_are_rejected(
    completed_run: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    _, result = completed_run
    base = result.artifact_dir

    duplicate = _clone(base, tmp_path, "duplicate-json")
    (duplicate / "conformal.json").write_text(
        '{"schema_version":1,"schema_version":1}\n',
        encoding="utf-8",
    )
    _rewrite_manifest_hash(duplicate, "conformal.json")
    _assert_invalid(duplicate, "duplicate", "duplicate-json")

    nonfinite = _clone(base, tmp_path, "nonfinite-json")
    (nonfinite / "conformal.json").write_text('{"radius":NaN}\n', encoding="utf-8")
    _rewrite_manifest_hash(nonfinite, "conformal.json")
    _assert_invalid(nonfinite, "non-finite", "nonfinite-json")

    bad_config = _clone(base, tmp_path, "bad-config")
    with (bad_config / "config.toml").open("a", encoding="utf-8") as handle:
        handle.write("\nunknown = true\n")
    _rewrite_manifest_hash(bad_config, "config.toml")
    _assert_invalid(bad_config, "config.toml", "bad-config")


def _read_dataset(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_dataset(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_schema_rows_splits_and_uniqueness_are_enforced(
    completed_run: tuple[Any, Any],
    tmp_path: Path,
) -> None:
    _, result = completed_run
    base = result.artifact_dir

    wrong_header = _clone(base, tmp_path, "wrong-header")
    dataset = wrong_header / "dataset.csv"
    lines = dataset.read_text(encoding="utf-8").splitlines()
    lines[0] = "wrong"
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rewrite_manifest_hash(wrong_header, "dataset.csv")
    _assert_invalid(wrong_header, "columns", "wrong-header")

    extra_column = _clone(base, tmp_path, "extra-column")
    dataset = extra_column / "dataset.csv"
    lines = dataset.read_text(encoding="utf-8").splitlines()
    lines[1] += ",extra"
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rewrite_manifest_hash(extra_column, "dataset.csv")
    _assert_invalid(extra_column, "extra columns", "extra-column")

    mutations = [
        ("negative-premium", "premium", "-1", "premium"),
        ("unknown-split", "split", "training", "unknown split"),
        ("empty-source", "source", "", "source"),
    ]
    for label, field, value, expected in mutations:
        artifact = _clone(base, tmp_path, label)
        rows = _read_dataset(artifact / "dataset.csv")
        rows[0][field] = value
        _write_dataset(artifact / "dataset.csv", rows)
        _rewrite_manifest_hash(artifact, "dataset.csv")
        _assert_invalid(artifact, expected, label)

    duplicate = _clone(base, tmp_path, "duplicate-row")
    rows = _read_dataset(duplicate / "dataset.csv")
    rows.append(dict(rows[0]))
    _write_dataset(duplicate / "dataset.csv", rows)
    _rewrite_manifest_hash(duplicate, "dataset.csv")
    _assert_invalid(duplicate, "duplicates", "duplicate-row")

    split_count = _clone(base, tmp_path, "split-count")
    rows = _read_dataset(split_count / "dataset.csv")
    audit = next(row for row in rows if row["split"] == "audit")
    audit["split"] = "mapping"
    _write_dataset(split_count / "dataset.csv", rows)
    _rewrite_manifest_hash(split_count, "dataset.csv")
    _assert_invalid(split_count, "audit count", "split-count")

    mapping_count = _clone(base, tmp_path, "mapping-count")
    rows = _read_dataset(mapping_count / "dataset.csv")
    removed = 0
    retained: list[dict[str, str]] = []
    for row in rows:
        if row["split"] == "mapping" and removed < 5:
            removed += 1
        else:
            retained.append(row)
    _write_dataset(mapping_count / "dataset.csv", retained)
    (mapping_count / "dataset.schema.json").write_text(
        json.dumps(_dataset_schema(len(retained)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest_hash(mapping_count, "dataset.csv")
    _rewrite_manifest_hash(mapping_count, "dataset.schema.json")
    _assert_invalid(mapping_count, "mapping count", "mapping-count")


def test_unexpected_skops_type_is_never_trusted(
    completed_run: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = completed_run
    monkeypatch.setattr(
        "pricing_mapper.artifact.sio.get_untrusted_types",
        lambda **_: ["builtins.eval"],
    )
    with pytest.raises(ArtifactError, match="untrusted"):
        validate_artifact(result.artifact_dir)


def test_unreadable_skops_container_is_an_artifact_error(
    completed_run: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = completed_run

    def fail_inspection(**_: Any) -> list[str]:
        raise RuntimeError("malformed container")

    monkeypatch.setattr(
        "pricing_mapper.artifact.sio.get_untrusted_types",
        fail_inspection,
    )
    with pytest.raises(ArtifactError, match="inspected safely"):
        validate_artifact(result.artifact_dir)
