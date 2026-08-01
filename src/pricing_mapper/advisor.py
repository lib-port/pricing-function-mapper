"""Untrusted Ollama policy advice for the local Bayesian acquisition scorer."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pricing_mapper.acquisition import (
    AcquisitionContext,
    ResidualScore,
    UncertaintyScore,
    normalize01,
    select_from_base_scores,
)
from pricing_mapper.config import OllamaConfig
from pricing_mapper.domain import CATEGORICAL_FIELDS, FIELD_ORDER, DomainSpec
from pricing_mapper.exceptions import (
    AdvisorError,
    AdvisorModelError,
    AdvisorUnavailable,
    AdvisorValidationError,
)

PolicyName = Literal["balanced", "uncertainty", "residual", "explore", "exploit"]
BoostValue = float
MAX_RESPONSE_BYTES = 16_384
MAX_POLICY_CONTENT_BYTES = 4_096
MAX_DIAGNOSTIC_BYTES = 64_000
MAX_OUTPUT_TOKENS = 128


@dataclass(frozen=True)
class PolicyDefinition:
    uncertainty_weight: float
    residual_weight: float
    exploration_fraction: float


POLICY_CATALOGUE: dict[PolicyName, PolicyDefinition] = {
    "balanced": PolicyDefinition(0.70, 0.30, 0.20),
    "uncertainty": PolicyDefinition(0.85, 0.15, 0.20),
    "residual": PolicyDefinition(0.55, 0.45, 0.20),
    "explore": PolicyDefinition(0.60, 0.20, 0.40),
    "exploit": PolicyDefinition(0.80, 0.20, 0.10),
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class ScoreDistribution(_StrictModel):
    minimum: float = Field(ge=0.0, le=1.0)
    q10: float = Field(ge=0.0, le=1.0)
    median: float = Field(ge=0.0, le=1.0)
    q90: float = Field(ge=0.0, le=1.0)
    maximum: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        values = (self.minimum, self.q10, self.median, self.q90, self.maximum)
        if any(left > right for left, right in itertools.pairwise(values)):
            raise ValueError("score distribution quantiles must be ordered")
        return self


class DiagnosticBin(_StrictModel):
    bin_id: str = Field(min_length=1, max_length=96)
    field_name: str = Field(min_length=1, max_length=64)
    field_kind: Literal["numeric", "categorical"]
    candidate_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    normalized_uncertainty: float = Field(ge=0.0, le=1.0)
    normalized_absolute_residual: float = Field(ge=0.0, le=1.0)


class PreviousPolicy(_StrictModel):
    policy: PolicyName
    relative_validation_mae_improvement: float | None = None


class DiagnosticSummary(_StrictModel):
    schema_version: Literal[1] = 1
    mapping_sample_count: int = Field(ge=1)
    validation_mae_trend: list[float] = Field(max_length=8)
    bins: list[DiagnosticBin] = Field(min_length=1, max_length=96)
    acquisition_score_distributions: dict[str, ScoreDistribution]
    previous_policy: PreviousPolicy | None = None

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if any(value < 0.0 or not math.isfinite(value) for value in self.validation_mae_trend):
            raise ValueError("validation MAE trend must be finite and non-negative")
        ids = [item.bin_id for item in self.bins]
        if len(ids) != len(set(ids)):
            raise ValueError("diagnostic bin IDs must be unique")
        if set(self.acquisition_score_distributions) != {
            "posterior_uncertainty",
            "residual_targeting",
            "balanced_combined",
        }:
            raise ValueError("diagnostic acquisition distributions have an invalid set")
        return self


class BinBoost(_StrictModel):
    bin_id: str = Field(min_length=1, max_length=96)
    boost: BoostValue = Field(json_schema_extra={"enum": [1.1, 1.25]})

    @field_validator("boost")
    @classmethod
    def validate_boost(cls, value: float) -> float:
        if value not in {1.1, 1.25}:
            raise ValueError("bin boost must be exactly 1.10 or 1.25")
        return value


class PolicyResponse(_StrictModel):
    policy: PolicyName
    bin_boosts: list[BinBoost] = Field(max_length=3)

    @model_validator(mode="after")
    def validate_unique_bins(self) -> Self:
        ids = [item.bin_id for item in self.bin_boosts]
        if len(ids) != len(set(ids)):
            raise ValueError("a diagnostic bin may be nominated only once")
        return self


class RuntimeRecord(_StrictModel):
    ollama_version: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=192)
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    quantization_level: Literal["Q4_K_M"]
    model_size_bytes: int = Field(gt=0)
    resource_mode: Literal["cpu-only-2cpu-8gb"]


class GenerationRecord(_StrictModel):
    temperature: Literal[0]
    seed: int = Field(ge=0, le=2**31 - 1)
    num_predict: Literal[128]
    stream: Literal[False]


class AdvisorMemoryRecord(_StrictModel):
    model: str = Field(min_length=1, max_length=192)
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    resident_size_bytes: int = Field(gt=0)
    vram_size_bytes: int = Field(ge=0)


class AdvisorDecisionRecord(_StrictModel):
    schema_version: Literal[1]
    batch_id: int = Field(ge=0)
    response: PolicyResponse
    prompt_version: Literal["policy-advisor-v1"]
    prompt_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    request_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    response_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runtime: RuntimeRecord
    generation: GenerationRecord
    memory: AdvisorMemoryRecord
    attempts: int = Field(ge=1, le=3)
    attempt_latencies_ms: list[float] = Field(min_length=1, max_length=3)
    total_latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        if self.attempts != len(self.attempt_latencies_ms):
            raise ValueError("advisor attempt count and timings must be aligned")
        if any(value < 0.0 or not math.isfinite(value) for value in self.attempt_latencies_ms):
            raise ValueError("advisor attempt timings must be finite and non-negative")
        if not math.isclose(
            self.total_latency_ms,
            sum(self.attempt_latencies_ms),
            rel_tol=0.0,
            abs_tol=0.00001,
        ):
            raise ValueError("advisor total latency must equal its attempt timings")
        return self


@dataclass(frozen=True)
class DiagnosticBundle:
    """Safe outbound aggregates plus local-only candidate memberships."""

    summary: DiagnosticSummary
    candidate_bin_memberships: Mapping[str, np.ndarray]

    @property
    def allowed_bin_ids(self) -> frozenset[str]:
        return frozenset(item.bin_id for item in self.summary.bins)


@dataclass(frozen=True)
class OllamaRuntime:
    ollama_version: str
    model: str
    digest: str
    quantization_level: str
    model_size_bytes: int
    resource_mode: str

    def to_record(self) -> dict[str, Any]:
        return {
            "ollama_version": self.ollama_version,
            "model": self.model,
            "digest": self.digest,
            "quantization_level": self.quantization_level,
            "model_size_bytes": self.model_size_bytes,
            "resource_mode": self.resource_mode,
        }


@dataclass(frozen=True)
class AdvisorDecision:
    response: PolicyResponse
    record: Mapping[str, Any]


@dataclass(frozen=True)
class PolicyApplication:
    selected_indices: tuple[int, ...]
    exploration_count: int


def _round(value: float) -> float:
    return float(round(float(value), 6))


def _distribution(values: np.ndarray) -> ScoreDistribution:
    array = normalize01(np.asarray(values, dtype=float))
    if array.size == 0:
        points = np.zeros(5, dtype=float)
    else:
        points = np.quantile(array, (0.0, 0.10, 0.50, 0.90, 1.0))
    return ScoreDistribution(
        minimum=_round(points[0]),
        q10=_round(points[1]),
        median=_round(points[2]),
        q90=_round(points[3]),
        maximum=_round(points[4]),
    )


def _numeric_bin(values: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = np.clip((values.astype(float) - low) / (high - low), 0.0, 1.0)
    return np.asarray(np.minimum((scaled * 4.0).astype(int), 3), dtype=int)


def allowed_diagnostic_bin_ids(domain: DomainSpec) -> frozenset[str]:
    """Return every code-owned diagnostic bin ID for a resolved domain."""

    result: set[str] = set()
    for field_name in FIELD_ORDER:
        if field_name in domain.numeric:
            result.update(f"v1:num:{field_name}:{index}" for index in range(4))
        elif field_name in CATEGORICAL_FIELDS:
            result.update(
                f"v1:cat:{field_name}:{index}"
                for index, _ in enumerate(domain.categorical[field_name])
            )
    return frozenset(result)


def _previous_policy(history: Sequence[Mapping[str, Any]]) -> PreviousPolicy | None:
    maes: list[tuple[int, float]] = []
    for index, batch in enumerate(history):
        metrics = batch.get("metrics")
        if isinstance(metrics, Mapping) and isinstance(metrics.get("validation_mae"), (int, float)):
            value = float(metrics["validation_mae"])
            if math.isfinite(value) and value >= 0.0:
                maes.append((index, value))
    for index in range(len(history) - 1, -1, -1):
        raw = history[index].get("advisor")
        if not isinstance(raw, Mapping):
            continue
        response = raw.get("response")
        if not isinstance(response, Mapping) or response.get("policy") not in POLICY_CATALOGUE:
            continue
        current = next((mae for item, mae in reversed(maes) if item == index), None)
        prior = next((mae for item, mae in reversed(maes) if item < index), None)
        improvement = None
        if current is not None and prior is not None:
            improvement = 0.0 if prior == 0.0 else _round((prior - current) / prior)
        return PreviousPolicy(
            policy=response["policy"],
            relative_validation_mae_improvement=improvement,
        )
    return None


def build_diagnostic_summary(
    *,
    context: AcquisitionContext,
    validation_rows: Sequence[Mapping[str, Any]],
    batch_history: Sequence[Mapping[str, Any]],
    domain: DomainSpec,
) -> DiagnosticBundle:
    """Aggregate advisor diagnostics without premiums or individual quote rows."""

    if len(validation_rows) != context.training_residuals.size:
        raise ValueError("validation diagnostic rows and residuals must be aligned")
    uncertainty = UncertaintyScore().score(context)
    residual_targeting = ResidualScore().score(context)
    balanced = normalize01(0.70 * uncertainty + 0.30 * residual_targeting)
    validation_residual = normalize01(np.abs(context.training_residuals))
    candidate_bins: dict[str, np.ndarray] = {}
    bins: list[DiagnosticBin] = []

    candidate_rows = context.candidate_rows
    for field_name in FIELD_ORDER:
        if field_name in domain.numeric:
            bounds = domain.numeric[field_name]
            candidate_values = np.asarray(
                [float(row[field_name]) for row in candidate_rows], dtype=float
            )
            validation_values = np.asarray(
                [float(row[field_name]) for row in validation_rows], dtype=float
            )
            candidate_indices = _numeric_bin(candidate_values, bounds.low, bounds.high)
            validation_indices = _numeric_bin(validation_values, bounds.low, bounds.high)
            for index in range(4):
                bin_id = f"v1:num:{field_name}:{index}"
                candidate_mask = candidate_indices == index
                validation_mask = validation_indices == index
                candidate_bins[bin_id] = candidate_mask
                bins.append(
                    DiagnosticBin(
                        bin_id=bin_id,
                        field_name=field_name,
                        field_kind="numeric",
                        candidate_count=int(np.sum(candidate_mask)),
                        validation_count=int(np.sum(validation_mask)),
                        normalized_uncertainty=(
                            _round(float(np.mean(uncertainty[candidate_mask])))
                            if np.any(candidate_mask)
                            else 0.0
                        ),
                        normalized_absolute_residual=(
                            _round(float(np.mean(validation_residual[validation_mask])))
                            if np.any(validation_mask)
                            else 0.0
                        ),
                    )
                )
        elif field_name in CATEGORICAL_FIELDS:
            categories = domain.categorical[field_name]
            for index, category in enumerate(categories):
                bin_id = f"v1:cat:{field_name}:{index}"
                candidate_mask = np.asarray(
                    [row[field_name] == category for row in candidate_rows], dtype=bool
                )
                validation_mask = np.asarray(
                    [row[field_name] == category for row in validation_rows], dtype=bool
                )
                candidate_bins[bin_id] = candidate_mask
                bins.append(
                    DiagnosticBin(
                        bin_id=bin_id,
                        field_name=field_name,
                        field_kind="categorical",
                        candidate_count=int(np.sum(candidate_mask)),
                        validation_count=int(np.sum(validation_mask)),
                        normalized_uncertainty=(
                            _round(float(np.mean(uncertainty[candidate_mask])))
                            if np.any(candidate_mask)
                            else 0.0
                        ),
                        normalized_absolute_residual=(
                            _round(float(np.mean(validation_residual[validation_mask])))
                            if np.any(validation_mask)
                            else 0.0
                        ),
                    )
                )

    mae_trend: list[float] = []
    for batch in batch_history:
        metrics = batch.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        value = metrics.get("validation_mae")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        parsed = float(value)
        if math.isfinite(parsed) and parsed >= 0.0:
            mae_trend.append(_round(parsed))

    summary = DiagnosticSummary(
        mapping_sample_count=len(context.training_rows),
        validation_mae_trend=mae_trend[-8:],
        bins=bins,
        acquisition_score_distributions={
            "posterior_uncertainty": _distribution(uncertainty),
            "residual_targeting": _distribution(residual_targeting),
            "balanced_combined": _distribution(balanced),
        },
        previous_policy=_previous_policy(batch_history),
    )
    if frozenset(candidate_bins) != allowed_diagnostic_bin_ids(domain):
        raise AssertionError("diagnostic construction omitted a code-owned bin")
    assert_diagnostic_payload_safe(summary.model_dump(mode="json"))
    return DiagnosticBundle(summary=summary, candidate_bin_memberships=candidate_bins)


_FORBIDDEN_DIAGNOSTIC_KEYS = {
    "premium",
    "premiums",
    "quote",
    "quotes",
    "row",
    "rows",
    "target",
    "targets",
    "calibration",
    "audit",
}


def assert_diagnostic_payload_safe(payload: Mapping[str, Any]) -> None:
    """Reject accidental raw pricing or holdout data before network transport."""

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str) or key.casefold() in _FORBIDDEN_DIAGNOSTIC_KEYS:
                    raise AdvisorValidationError("advisor diagnostic contains a forbidden field")
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise AdvisorValidationError("advisor diagnostic contains a non-finite value")

    visit(payload)
    encoded = _canonical_json(payload)
    if len(encoded) > MAX_DIAGNOSTIC_BYTES:
        raise AdvisorValidationError("advisor diagnostic exceeds its size limit")


def _reject_constant(value: str) -> None:
    raise AdvisorValidationError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdvisorValidationError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _parse_json_object(raw: bytes | str, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise AdvisorValidationError(f"{context} is not UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise AdvisorValidationError(f"{context} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise AdvisorValidationError(f"{context} must be a JSON object")
    return value


def validate_policy_response(
    raw: bytes | str, allowed_bin_ids: set[str] | frozenset[str]
) -> PolicyResponse:
    """Strictly validate one structured response and its diagnostic-bin allow-list."""

    if len(raw.encode("utf-8") if isinstance(raw, str) else raw) > MAX_POLICY_CONTENT_BYTES:
        raise AdvisorValidationError("advisor policy content exceeds its size limit")
    parsed = _parse_json_object(raw, "advisor policy content")
    try:
        response = PolicyResponse.model_validate(parsed, strict=True)
    except ValueError as exc:
        raise AdvisorValidationError("advisor policy content violates its JSON schema") from exc
    unknown = sorted({boost.bin_id for boost in response.bin_boosts} - set(allowed_bin_ids))
    if unknown:
        raise AdvisorValidationError("advisor nominated a diagnostic bin outside the allow-list")
    return response


def validate_advisor_decision(
    decision: AdvisorDecision,
    *,
    allowed_bin_ids: frozenset[str],
    batch_id: int,
    run_seed: int,
    runtime: OllamaRuntime,
) -> dict[str, Any]:
    """Validate even injected advisor implementations before journal persistence."""

    try:
        parsed = AdvisorDecisionRecord.model_validate(dict(decision.record), strict=True)
    except (TypeError, ValueError) as exc:
        raise AdvisorValidationError("advisor decision provenance record is invalid") from exc
    if parsed.batch_id != batch_id:
        raise AdvisorValidationError("advisor decision batch differs from the proposed batch")
    if parsed.response != decision.response:
        raise AdvisorValidationError("advisor decision response differs from its provenance")
    if any(boost.bin_id not in allowed_bin_ids for boost in parsed.response.bin_boosts):
        raise AdvisorValidationError("advisor decision contains a bin outside the allow-list")
    if parsed.runtime.model_dump(mode="json") != runtime.to_record():
        raise AdvisorValidationError("advisor decision runtime differs from verified runtime")
    if parsed.memory.model != runtime.model or parsed.memory.digest != runtime.digest:
        raise AdvisorValidationError("advisor decision memory differs from verified model")
    if parsed.generation.seed != derived_advisor_seed(run_seed, batch_id):
        raise AdvisorValidationError("advisor decision generation seed is invalid")
    return parsed.model_dump(mode="json")


def policy_response_schema() -> dict[str, Any]:
    """Return the exact JSON Schema supplied to Ollama's ``format`` field."""

    return PolicyResponse.model_json_schema()


def system_prompt(version: str) -> str:
    if version != "policy-advisor-v1":
        raise AdvisorValidationError(f"unsupported advisor prompt version {version!r}")
    return (
        "Policy Advisor protocol policy-advisor-v1. Choose one acquisition policy from the "
        "fixed catalogue using only the aggregate diagnostics in the user message. Catalogue: "
        "balanced=(0.70 uncertainty,0.30 residual,0.20 exploration); "
        "uncertainty=(0.85,0.15,0.20); residual=(0.55,0.45,0.20); "
        "explore=(0.60,0.20,0.40); exploit=(0.80,0.20,0.10). You may nominate zero to "
        "three bin_id values already present in the diagnostics, each with boost 1.10 or 1.25. "
        "Do not invent fields, bins, breakpoints, weights, expressions, or inputs. Return only "
        "JSON conforming exactly to the supplied schema."
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def derived_advisor_seed(run_seed: int, batch_id: int) -> int:
    if run_seed < 0 or batch_id < 0:
        raise ValueError("advisor seed inputs must be non-negative")
    raw = hashlib.sha256(f"{run_seed}:{batch_id}:policy-advisor-v1".encode()).digest()
    return int.from_bytes(raw[:4], "big") & (2**31 - 1)


class AdvisorTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        payload: bytes | None,
        *,
        timeout: float,
        max_bytes: int,
    ) -> bytes:
        """Return one bounded JSON response body or raise ``AdvisorError``."""


class UrllibAdvisorTransport:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        payload: bytes | None,
        *,
        timeout: float,
        max_bytes: int,
    ) -> bytes:
        request = urllib.request.Request(  # noqa: S310 - endpoint is validated HTTP(S)
            f"{self.endpoint}{path}",
            data=payload,
            method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                content_type = response.headers.get_content_type()
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > max_bytes:
                    raise AdvisorValidationError("advisor response exceeds its size limit")
                if content_type != "application/json":
                    raise AdvisorValidationError("advisor response is not application/json")
                body = bytes(response.read(max_bytes + 1))
        except AdvisorError:
            raise
        except TimeoutError as exc:
            raise AdvisorUnavailable("advisor request timed out") from exc
        except urllib.error.HTTPError as exc:
            raise AdvisorUnavailable(f"advisor HTTP request failed with status {exc.code}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise AdvisorUnavailable("advisor endpoint is unavailable") from exc
        if len(body) > max_bytes:
            raise AdvisorValidationError("advisor response exceeds its size limit")
        return body


class PolicyAdvisor(Protocol):
    def verify(self) -> OllamaRuntime:
        """Verify the configured runtime and full installed-model digest."""

    def advise(
        self,
        diagnostics: DiagnosticSummary,
        *,
        allowed_bin_ids: frozenset[str],
        run_seed: int,
        batch_id: int,
        runtime: OllamaRuntime,
    ) -> AdvisorDecision:
        """Return one strictly validated, provenance-ready decision."""


class OllamaPolicyAdvisor:
    def __init__(
        self,
        config: OllamaConfig,
        *,
        transport: AdvisorTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibAdvisorTransport(config.endpoint)

    def _get_object(self, path: str, context: str) -> dict[str, Any]:
        body = self.transport.request(
            "GET",
            path,
            None,
            timeout=self.config.timeout_seconds,
            max_bytes=MAX_RESPONSE_BYTES,
        )
        if len(body) > MAX_RESPONSE_BYTES:
            raise AdvisorValidationError("advisor response exceeds its size limit")
        return _parse_json_object(body, context)

    def verify(self) -> OllamaRuntime:
        try:
            version_raw = self._get_object("/api/version", "Ollama version response")
            tags = self._get_object("/api/tags", "Ollama model-list response")
        except AdvisorModelError:
            raise
        except AdvisorError as exc:
            raise AdvisorModelError("cannot verify the pinned Ollama runtime and model") from exc
        version = version_raw.get("version")
        if not isinstance(version, str) or not version.strip() or len(version) > 64:
            raise AdvisorModelError("Ollama version response is invalid")
        models = tags.get("models")
        if not isinstance(models, list):
            raise AdvisorModelError("Ollama model-list response is invalid")
        installed: Mapping[str, Any] | None = None
        for candidate in models:
            if not isinstance(candidate, Mapping):
                raise AdvisorModelError("Ollama model-list contains an invalid entry")
            if (
                candidate.get("name") == self.config.model
                or candidate.get("model") == self.config.model
            ):
                if installed is not None:
                    raise AdvisorModelError("Ollama model-list contains duplicate model entries")
                installed = candidate
        if installed is None:
            raise AdvisorModelError("the configured Ollama model is not installed")
        digest = installed.get("digest")
        if isinstance(digest, str) and not digest.startswith("sha256:"):
            digest = f"sha256:{digest}"
        if digest != self.config.required_digest:
            raise AdvisorModelError("the installed Ollama model digest does not match the pin")
        details = installed.get("details")
        quantization = details.get("quantization_level") if isinstance(details, Mapping) else None
        if quantization != "Q4_K_M":
            raise AdvisorModelError("the installed Ollama model is not Q4_K_M")
        size = installed.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise AdvisorModelError("the installed Ollama model size is invalid")
        return OllamaRuntime(
            ollama_version=version,
            model=self.config.model,
            digest=digest,
            quantization_level=quantization,
            model_size_bytes=size,
            resource_mode=self.config.resource_mode,
        )

    def _model_memory(self) -> dict[str, Any]:
        running = self._get_object("/api/ps", "Ollama running-model response")
        models = running.get("models")
        if not isinstance(models, list):
            raise AdvisorValidationError("Ollama running-model response is invalid")
        matches = [
            item
            for item in models
            if isinstance(item, Mapping)
            and (item.get("name") == self.config.model or item.get("model") == self.config.model)
        ]
        if len(matches) != 1:
            raise AdvisorValidationError("the advised Ollama model is not resident exactly once")
        selected = matches[0]
        digest = selected.get("digest")
        if isinstance(digest, str) and not digest.startswith("sha256:"):
            digest = f"sha256:{digest}"
        resident = selected.get("size")
        vram = selected.get("size_vram")
        if (
            digest != self.config.required_digest
            or isinstance(resident, bool)
            or not isinstance(resident, int)
            or resident <= 0
            or isinstance(vram, bool)
            or not isinstance(vram, int)
            or vram < 0
        ):
            raise AdvisorValidationError("Ollama running-model memory metadata is invalid")
        return {
            "model": self.config.model,
            "digest": digest,
            "resident_size_bytes": resident,
            "vram_size_bytes": vram,
        }

    def advise(
        self,
        diagnostics: DiagnosticSummary,
        *,
        allowed_bin_ids: frozenset[str],
        run_seed: int,
        batch_id: int,
        runtime: OllamaRuntime,
    ) -> AdvisorDecision:
        if runtime.model != self.config.model or runtime.digest != self.config.required_digest:
            raise AdvisorModelError("advisor runtime metadata differs from configuration")
        if allowed_bin_ids != frozenset(item.bin_id for item in diagnostics.bins):
            raise AdvisorValidationError("advisor bin allow-list differs from diagnostics")
        diagnostic_payload = diagnostics.model_dump(mode="json")
        assert_diagnostic_payload_safe(diagnostic_payload)
        prompt = system_prompt(self.config.prompt_version)
        seed = derived_advisor_seed(run_seed, batch_id)
        generation = {
            "temperature": 0,
            "seed": seed,
            "num_predict": MAX_OUTPUT_TOKENS,
        }
        request_body = {
            "model": self.config.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": _canonical_json(diagnostic_payload).decode("utf-8"),
                },
            ],
            "format": policy_response_schema(),
            "options": generation,
        }
        encoded_request = _canonical_json(request_body)
        attempt_latencies: list[float] = []
        last_error: AdvisorError | None = None
        for _ in range(self.config.retry_count + 1):
            started = time.perf_counter()
            try:
                raw_response = self.transport.request(
                    "POST",
                    "/api/chat",
                    encoded_request,
                    timeout=self.config.timeout_seconds,
                    max_bytes=MAX_RESPONSE_BYTES,
                )
                if len(raw_response) > MAX_RESPONSE_BYTES:
                    raise AdvisorValidationError("advisor response exceeds its size limit")
                response_latency = (time.perf_counter() - started) * 1_000.0
                attempt_latencies.append(response_latency)
                if response_latency > self.config.timeout_seconds * 1_000.0:
                    raise AdvisorUnavailable("advisor response exceeded its wall-clock deadline")
                outer = _parse_json_object(raw_response, "Ollama chat response")
                if outer.get("model") != self.config.model:
                    raise AdvisorModelError("Ollama chat response model differs from the pin")
                if outer.get("done") is not True:
                    raise AdvisorValidationError("Ollama non-streaming response is incomplete")
                message = outer.get("message")
                if not isinstance(message, Mapping) or message.get("role") != "assistant":
                    raise AdvisorValidationError("Ollama chat response message is invalid")
                content = message.get("content")
                if not isinstance(content, str):
                    raise AdvisorValidationError("Ollama chat response content is invalid")
                policy = validate_policy_response(content, allowed_bin_ids)
                memory = self._model_memory()
                record = AdvisorDecisionRecord.model_validate(
                    {
                        "schema_version": 1,
                        "batch_id": batch_id,
                        "response": policy.model_dump(mode="json"),
                        "prompt_version": self.config.prompt_version,
                        "prompt_hash": _digest(prompt.encode("utf-8")),
                        "request_digest": _digest(encoded_request),
                        "response_digest": _digest(raw_response),
                        "runtime": runtime.to_record(),
                        "generation": {**generation, "stream": False},
                        "memory": memory,
                        "attempts": len(attempt_latencies),
                        "attempt_latencies_ms": [_round(item) for item in attempt_latencies],
                        "total_latency_ms": _round(sum(attempt_latencies)),
                    },
                    strict=True,
                )
                return AdvisorDecision(response=policy, record=record.model_dump(mode="json"))
            except AdvisorError as exc:
                if len(attempt_latencies) < (_ + 1):
                    attempt_latencies.append((time.perf_counter() - started) * 1_000.0)
                last_error = exc
        if last_error is None:
            raise AssertionError("advisor retry loop completed without a result or error")
        raise AdvisorError(
            f"advisor failed closed after {self.config.retry_count + 1} attempts"
        ) from last_error


def apply_policy(
    context: AcquisitionContext,
    diagnostics: DiagnosticBundle,
    response: PolicyResponse,
    *,
    count: int,
    greedy_diversity_weight: float,
) -> PolicyApplication:
    """Apply a finite catalogue policy to local numerical scores only."""

    if count < 0:
        raise ValueError("policy selection count cannot be negative")
    if response.policy not in POLICY_CATALOGUE:
        raise AdvisorValidationError("advisor selected an unknown policy")
    if not 0.0 <= greedy_diversity_weight <= 1.0:
        raise ValueError("greedy diversity weight must be within [0, 1]")
    policy = POLICY_CATALOGUE[response.policy]
    uncertainty = UncertaintyScore().score(context)
    residual = ResidualScore().score(context)
    base = normalize01(policy.uncertainty_weight * uncertainty + policy.residual_weight * residual)
    factors = np.ones(len(base), dtype=float)
    for nomination in response.bin_boosts:
        membership = diagnostics.candidate_bin_memberships.get(nomination.bin_id)
        if membership is None:
            raise AdvisorValidationError(
                "advisor nominated a diagnostic bin outside the allow-list"
            )
        mask = np.asarray(membership, dtype=bool)
        if mask.shape != base.shape:
            raise AdvisorValidationError("advisor diagnostic bin membership is corrupt")
        # Multiple nominated dimensions never compound beyond the largest fixed boost.
        factors[mask] = np.maximum(factors[mask], float(nomination.boost))
    boosted = normalize01(base * factors)
    bounded_count = min(count, len(context.candidate_rows))
    exploration_count = min(
        bounded_count,
        max(0, round(bounded_count * policy.exploration_fraction)),
    )
    active_count = bounded_count - exploration_count
    indices = select_from_base_scores(
        context,
        boosted,
        active_count,
        diversity_weight=greedy_diversity_weight,
    )
    if len(indices) != active_count:
        raise AdvisorValidationError("local policy application selected too few candidates")
    return PolicyApplication(tuple(indices), exploration_count)


def balanced_policy() -> PolicyResponse:
    """Deterministic Bayesian-only policy used when no advisor is configured."""

    return PolicyResponse(policy="balanced", bin_boosts=[])
