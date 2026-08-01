from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pricing_mapper.artifact import validate_artifact
from pricing_mapper.cli import EXIT_ARTIFACT, EXIT_INPUT, EXIT_OK, run_cli
from pricing_mapper.domain import FIELD_ORDER
from pricing_mapper.engine import PricingEngine
from pricing_mapper.exceptions import ArtifactError, DomainValidationError, LegacyArtifactError
from pricing_mapper.orchestration import MappingRun
from pricing_mapper.provider import reference_car_quote


def _rewrite_manifest_hash(artifact: Path, name: str) -> None:
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = (artifact / name).read_bytes()
    manifest["files"][name] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_artifact_load_prediction_and_strict_domain(
    completed_run: tuple[Any, Any],
    valid_row: dict[str, Any],
) -> None:
    _, result = completed_run
    inspection = validate_artifact(result.artifact_dir)
    assert inspection["valid"]
    assert inspection["files_verified"] == 12
    engine = PricingEngine.load(result.artifact_dir)
    prediction = engine.predict(valid_row)
    assert 0 <= prediction.lower <= prediction.premium <= prediction.upper
    assert prediction.model_version.startswith("v1-")
    assert engine.predict_batch([]) == []

    outside = dict(valid_row, driver_age=5.0)
    with pytest.raises(DomainValidationError):
        engine.predict(outside)


def test_extra_trees_artifact_round_trip(tmp_path: Path) -> None:
    from conftest import tiny_config

    config = tiny_config(tmp_path, seed=5)
    result = MappingRun(config, run_id="extra-trees").run()
    inspection = validate_artifact(result.artifact_dir)
    assert inspection["model_kind"] == "extra_trees"
    assert PricingEngine.load(result.artifact_dir).model_kind == "extra_trees"


def test_model_card_contains_complete_immutable_run_evidence(
    completed_run: tuple[Any, Any],
) -> None:
    _, result = completed_run
    artifact = result.artifact_dir
    card = (artifact / "model-card.md").read_text(encoding="utf-8")
    evaluation = json.loads((artifact / "evaluation.json").read_text(encoding="utf-8"))
    selection = json.loads((artifact / "model.json").read_text(encoding="utf-8"))["selection"]
    dependencies = json.loads((artifact / "dependencies.json").read_text(encoding="utf-8"))

    for heading in (
        "## Model selection",
        "## Conformal interval",
        "## Independent audit",
        "### Predefined risk slices",
        "## Promotion gates",
        "## Provider provenance",
        "## Runtime dependencies",
        "## Warnings",
        "## Limitations and governance",
    ):
        assert heading in card
    assert str(selection["selected"]["validation_mae"]) in card
    assert str(evaluation["audit"]["metrics"]["p95_absolute_error"]) in card
    assert str(evaluation["audit_interval"]["median_width"]) in card
    assert "young_driver_under_25" in card
    assert "pricing_mapper.reference_car_quote.v1" in card
    for name, version in dependencies.items():
        assert f"`{name}`: `{version}`" in card
    assert "Do not edit this hash-protected artifact" in card
    assert validate_artifact(artifact)["valid"]


def test_model_card_records_custom_provider_identity(tmp_path: Path) -> None:
    from conftest import tiny_config

    class CustomProvider:
        provider_id = "test.custom-provider.v2"

        def __call__(self, quote: Any) -> float:
            return reference_car_quote(quote)

    result = MappingRun(
        tiny_config(tmp_path, seed=31),
        provider=CustomProvider(),
        run_id="custom-provider-card",
    ).run()
    card = (result.artifact_dir / "model-card.md").read_text(encoding="utf-8")
    assert "Identity: `test.custom-provider.v2`" in card


def test_tampered_and_unlisted_artifacts_are_rejected(
    completed_run: tuple[Any, Any],
) -> None:
    _, result = completed_run
    dataset = result.artifact_dir / "dataset.csv"
    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="integrity"):
        validate_artifact(result.artifact_dir)


def test_runtime_version_mismatch_is_rejected(
    completed_run: tuple[Any, Any],
) -> None:
    _, result = completed_run
    dependencies = result.artifact_dir / "dependencies.json"
    raw = json.loads(dependencies.read_text(encoding="utf-8"))
    raw["scikit-learn"] = "0.0.0"
    dependencies.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_manifest_hash(result.artifact_dir, "dependencies.json")
    with pytest.raises(ArtifactError, match="version mismatch"):
        PricingEngine.load(result.artifact_dir)


def test_v0_pickle_is_rejected_without_loading(tmp_path: Path) -> None:
    legacy = tmp_path / "engine.pkl"
    legacy.write_bytes(b"cos\nsystem\n(S'echo unsafe'\ntR.")
    with pytest.raises(LegacyArtifactError, match="pickle"):
        PricingEngine.load(legacy)


def test_cli_validate_inspect_predict_and_batch(
    completed_run: tuple[Any, Any],
    valid_row: dict[str, Any],
    tmp_path: Path,
) -> None:
    config, result = completed_run
    config_path = result.artifact_dir / "config.toml"
    assert run_cli(["config", "validate", "--config", str(config_path)]) == EXIT_OK
    assert run_cli(["artifact", "inspect", "--artifact", str(result.artifact_dir)]) == EXIT_OK
    assert (
        run_cli(
            [
                "predict",
                "row",
                "--artifact",
                str(result.artifact_dir),
                "--json",
                json.dumps(valid_row),
            ]
        )
        == EXIT_OK
    )

    input_csv = tmp_path / "rows.csv"
    output_csv = tmp_path / "predictions.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELD_ORDER))
        writer.writeheader()
        writer.writerow(valid_row)
    assert (
        run_cli(
            [
                "predict",
                "batch",
                "--artifact",
                str(result.artifact_dir),
                "--input",
                str(input_csv),
                "--output",
                str(output_csv),
            ]
        )
        == EXIT_OK
    )
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["lower"]) <= float(rows[0]["premium"]) <= float(rows[0]["upper"])
    assert config.config_version == 1


def test_cli_exit_codes_for_bad_config_and_legacy_artifact(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("unknown = true\n", encoding="utf-8")
    assert run_cli(["config", "validate", "--config", str(bad)]) == EXIT_INPUT
    legacy = tmp_path / "legacy.pkl"
    legacy.write_bytes(b"not loaded")
    assert run_cli(["artifact", "inspect", "--artifact", str(legacy)]) == EXIT_ARTIFACT
