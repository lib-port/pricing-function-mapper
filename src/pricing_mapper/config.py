"""Strict nested TOML configuration for pricing-mapper v1."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pricing_mapper.domain import (
    CONTINUOUS_FIELDS,
    INTEGER_FIELDS,
    DomainSpec,
    NumericDomain,
    default_numeric_bounds,
)
from pricing_mapper.exceptions import ConfigurationError, DomainValidationError


class StrictConfigModel(BaseModel):
    """Shared behavior for every nested configuration model."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class BoundsConfig(StrictConfigModel):
    low: float
    high: float

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if not math.isfinite(self.low) or not math.isfinite(self.high) or self.low >= self.high:
            raise ValueError("bounds must be finite and satisfy low < high")
        return self


def _default_bounds_config() -> dict[str, BoundsConfig]:
    return {
        name: BoundsConfig(low=float(low), high=float(high))
        for name, (low, high) in default_numeric_bounds().items()
    }


class DomainConfig(StrictConfigModel):
    """Optional narrowing of the supported v1 numeric domain."""

    bounds: dict[str, BoundsConfig] = Field(default_factory=_default_bounds_config)

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        defaults = default_numeric_bounds()
        unknown = sorted(set(self.bounds) - set(defaults))
        missing = sorted(set(defaults) - set(self.bounds))
        if unknown or missing:
            raise ValueError(f"domain bounds mismatch; missing={missing}, unknown={unknown}")
        for name, configured in self.bounds.items():
            outer_low, outer_high = defaults[name]
            if configured.low < outer_low or configured.high > outer_high:
                raise ValueError(
                    f"domain.bounds.{name} must stay within [{outer_low:g}, {outer_high:g}]"
                )
            if name in INTEGER_FIELDS and (
                not configured.low.is_integer() or not configured.high.is_integer()
            ):
                raise ValueError(f"domain.bounds.{name} must use integer limits")
        self.to_domain()
        return self

    def to_domain(self) -> DomainSpec:
        numeric = {
            name: NumericDomain(
                low=value.low,
                high=value.high,
                integer=name in INTEGER_FIELDS,
            )
            for name, value in self.bounds.items()
        }
        default = DomainSpec.default()
        try:
            return DomainSpec(numeric=numeric, categorical=default.categorical)
        except DomainValidationError as exc:
            raise ValueError(str(exc)) from exc


class AcquisitionConfig(StrictConfigModel):
    """Weights for pluggable active-acquisition score components."""

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "uncertainty": 0.10,
            "residual": 0.05,
            "breakpoint": 0.20,
            "diversity": 0.65,
        }
    )
    greedy_diversity_weight: float = Field(default=0.50, ge=0.0, le=1.0)
    focused_fraction: float = Field(default=0.30, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        supported = {"uncertainty", "residual", "breakpoint", "diversity"}
        unknown = sorted(set(self.weights) - supported)
        if unknown:
            raise ValueError(f"unknown acquisition score components: {unknown}")
        if not self.weights or any(
            value < 0 or not math.isfinite(value) for value in self.weights.values()
        ):
            raise ValueError("acquisition weights must be finite and non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("at least one acquisition weight must be positive")
        return self


class SamplingConfig(StrictConfigModel):
    mapping_budget: int = Field(default=260, ge=1)
    initial_size: int = Field(default=80, ge=1)
    batch_size: int = Field(default=20, ge=1)
    candidate_pool_size: int = Field(default=4_000, ge=10)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    strategy: Literal["active", "bayesian", "lhs", "random"] = "active"
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.initial_size > self.mapping_budget:
            raise ValueError("sampling.initial_size cannot exceed sampling.mapping_budget")
        return self


class EarlyStoppingConfig(StrictConfigModel):
    patience_batches: int = Field(default=0, ge=0)
    minimum_batches: int = Field(default=3, ge=1)
    minimum_relative_improvement: float = Field(default=0.005, ge=0.0, lt=1.0)
    confidence_level: float = Field(default=0.90, gt=0.5, lt=1.0)


def _default_monotonic_constraints() -> dict[str, int]:
    return {
        "vehicle_value": 1,
        "postcode_risk": 1,
        "theft_risk": 1,
        "claims_5y": 1,
        "convictions_5y": 1,
        "excess": -1,
        "annual_km": 1,
    }


class ModelConfig(StrictConfigModel):
    search_iterations: int = Field(default=4, ge=1, le=50)
    hgb_max_iter: int = Field(default=180, ge=20, le=2_000)
    extra_trees_estimators: int = Field(default=180, ge=20, le=5_000)
    committee_size: int = Field(default=5, ge=2, le=30)
    committee_estimators: int = Field(default=60, ge=10, le=1_000)
    n_jobs: int = Field(default=1, ge=1)
    max_p95_latency_ms: float = Field(default=25.0, gt=0.0)
    latency_repetitions: int = Field(default=100, ge=5, le=10_000)
    monotonic_constraints: dict[str, int] = Field(default_factory=_default_monotonic_constraints)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)

    @model_validator(mode="after")
    def validate_constraints(self) -> Self:
        numeric = set(CONTINUOUS_FIELDS) | set(INTEGER_FIELDS)
        unknown = sorted(set(self.monotonic_constraints) - numeric)
        if unknown:
            raise ValueError(f"unknown monotonic constraint fields: {unknown}")
        invalid = {
            name: value
            for name, value in self.monotonic_constraints.items()
            if value not in {-1, 0, 1}
        }
        if invalid:
            raise ValueError(f"monotonic constraints must be -1, 0, or 1: {invalid}")
        return self


class EvaluationConfig(StrictConfigModel):
    evaluation_budget: int = Field(default=120, ge=3)
    validation_fraction: float = Field(default=0.40, gt=0.0, lt=1.0)
    calibration_fraction: float = Field(default=0.30, gt=0.0, lt=1.0)
    audit_fraction: float = Field(default=0.30, gt=0.0, lt=1.0)
    conformal_coverage: float = Field(default=0.90, gt=0.5, lt=1.0)
    minimum_audit_coverage: float = Field(default=0.85, ge=0.0, le=1.0)
    maximum_audit_coverage: float = Field(default=0.95, ge=0.0, le=1.0)
    bootstrap_iterations: int = Field(default=300, ge=0, le=100_000)
    bootstrap_confidence: float = Field(default=0.95, gt=0.5, lt=1.0)

    @model_validator(mode="after")
    def validate_allocation(self) -> Self:
        total = self.validation_fraction + self.calibration_fraction + self.audit_fraction
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("evaluation split fractions must sum to 1.0")
        if self.minimum_audit_coverage > self.maximum_audit_coverage:
            raise ValueError(
                "evaluation.minimum_audit_coverage cannot exceed maximum_audit_coverage"
            )
        counts = self.split_counts()
        if min(counts.values()) < 1:
            raise ValueError("evaluation_budget must allocate at least one row to every split")
        return self

    def split_counts(self) -> dict[str, int]:
        labels = ("validation", "calibration", "audit")
        weights = (
            self.validation_fraction,
            self.calibration_fraction,
            self.audit_fraction,
        )
        raw = [self.evaluation_budget * weight for weight in weights]
        counts = [math.floor(value) for value in raw]
        for index in sorted(
            range(len(labels)),
            key=lambda item: (raw[item] - counts[item], -item),
            reverse=True,
        )[: self.evaluation_budget - sum(counts)]:
            counts[index] += 1
        return dict(zip(labels, counts, strict=True))


class ProviderConfig(StrictConfigModel):
    callable: str | None = None
    max_retries: int = Field(default=2, ge=0, le=20)
    initial_backoff_seconds: float = Field(default=0.05, ge=0.0, le=60.0)
    maximum_backoff_seconds: float = Field(default=1.0, ge=0.0, le=600.0)
    concurrency: int = Field(default=1, ge=1, le=256)

    @model_validator(mode="after")
    def validate_provider(self) -> Self:
        if self.callable is not None:
            target = self.callable.strip()
            if target.count(":") != 1 or any(not part for part in target.split(":")):
                raise ValueError("provider.callable must use non-empty 'module:function' syntax")
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "provider.maximum_backoff_seconds cannot be below initial_backoff_seconds"
            )
        return self


class ArtifactConfig(StrictConfigModel):
    output_dir: str = "outputs"
    state_dir_name: str = ".runs"
    model_card_title: str = "Comprehensive Car Insurance Surrogate"

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        for label, value in (
            ("artifact.output_dir", self.output_dir),
            ("artifact.state_dir_name", self.state_dir_name),
        ):
            if not value.strip() or "\x00" in value:
                raise ValueError(f"{label} must be a non-empty safe path")
        state_path = Path(self.state_dir_name)
        if state_path == Path(".") or state_path.is_absolute() or ".." in state_path.parts:
            raise ValueError("artifact.state_dir_name must be a relative child path")
        if not self.model_card_title.strip():
            raise ValueError("artifact.model_card_title cannot be empty")
        return self


_FULL_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}:[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class OllamaConfig(StrictConfigModel):
    """Pinned, opt-in local acquisition-policy advisor configuration."""

    endpoint: str = "http://127.0.0.1:11434"
    model: str = "granite4.1:3b"
    required_digest: str
    prompt_version: Literal["policy-advisor-v1"] = "policy-advisor-v1"
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=60.0)
    retry_count: int = Field(default=2, ge=0, le=2)
    resource_mode: Literal["cpu-only-2cpu-8gb"] = "cpu-only-2cpu-8gb"

    @model_validator(mode="after")
    def validate_ollama(self) -> Self:
        endpoint = urlsplit(self.endpoint)
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.hostname
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
            or endpoint.path not in {"", "/"}
        ):
            raise ValueError(
                "optimizer.ollama.endpoint must be an HTTP(S) origin without credentials, "
                "a path, query, or fragment"
            )
        if not _MODEL_NAME.fullmatch(self.model):
            raise ValueError("optimizer.ollama.model must be a fully tagged Ollama model name")
        if not _FULL_DIGEST.fullmatch(self.required_digest):
            raise ValueError(
                "optimizer.ollama.required_digest must be a full lowercase sha256 digest"
            )
        return self


class OptimizerConfig(StrictConfigModel):
    """Numerical optimizer extensions; all external advising is opt-in."""

    ollama: OllamaConfig | None = None


class MapperConfig(StrictConfigModel):
    """Stable v1 configuration consumed by :class:`MappingRun`."""

    config_version: Literal[1] = 1
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    artifact: ArtifactConfig = Field(default_factory=ArtifactConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)

    @model_validator(mode="after")
    def validate_optimizer(self) -> Self:
        if self.optimizer.ollama is not None and self.sampling.strategy != "bayesian":
            raise ValueError(
                "optimizer.ollama requires sampling.strategy = 'bayesian' so the advisor "
                "can only tune the local numerical optimizer"
            )
        return self

    @property
    def resolved_domain(self) -> DomainSpec:
        return self.domain.to_domain()

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> MapperConfig:
    """Load a v1 TOML file, rejecting legacy JSON and every unknown setting."""

    target = Path(path)
    if target.suffix.lower() != ".toml":
        raise ConfigurationError(
            "v1 configuration must be TOML; legacy JSON configuration is not supported"
        )
    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Cannot read v1 TOML config {target}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationError("TOML configuration must contain a top-level table")
    try:
        return MapperConfig.model_validate(dict(raw), strict=True)
    except ValueError as exc:
        raise ConfigurationError(f"Invalid v1 configuration: {exc}") from exc


def config_toml(config: MapperConfig) -> str:
    """Serialize a validated configuration snapshot as deterministic TOML."""

    return tomli_w.dumps(config.model_dump(mode="python", exclude_none=True))


def write_config(path: str | Path, config: MapperConfig) -> None:
    target = Path(path)
    target.write_text(config_toml(config), encoding="utf-8")


def config_json_schema() -> dict[str, Any]:
    schema = MapperConfig.model_json_schema()
    schema["$id"] = "https://pricing-function-mapper.local/schema/mapper-config-v1.json"
    schema["title"] = "Pricing Function Mapper v1 configuration"
    return schema
