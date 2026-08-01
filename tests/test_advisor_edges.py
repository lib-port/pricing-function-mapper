from __future__ import annotations

import copy
import json
import urllib.error
import urllib.request
from email.message import Message
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from test_advisor import (
    FakeTransport,
    advisor_config,
    diagnostic_bundle,
)

from pricing_mapper.advisor import (
    AdvisorDecision,
    DiagnosticBundle,
    DiagnosticSummary,
    OllamaPolicyAdvisor,
    PolicyResponse,
    ScoreDistribution,
    UrllibAdvisorTransport,
    apply_policy,
    assert_diagnostic_payload_safe,
    derived_advisor_seed,
    system_prompt,
    validate_advisor_decision,
    validate_policy_response,
)
from pricing_mapper.exceptions import AdvisorError, AdvisorModelError, AdvisorValidationError


class FakeHTTPResponse:
    def __init__(
        self,
        body: bytes,
        *,
        content_type: str = "application/json",
        content_length: str | None = None,
    ) -> None:
        self.body = body
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def read(self, amount: int) -> bytes:
        return self.body[:amount]


def test_urllib_transport_bounds_content_type_and_failures(monkeypatch: Any) -> None:
    transport = UrllibAdvisorTransport("http://127.0.0.1:11434")

    monkeypatch.setattr(
        transport._opener,
        "open",
        lambda *_args, **_kwargs: FakeHTTPResponse(b'{"ok":true}'),
    )
    assert (
        transport.request("GET", "/api/version", None, timeout=1.0, max_bytes=100) == b'{"ok":true}'
    )

    responses = (
        FakeHTTPResponse(b"{}", content_length="101"),
        FakeHTTPResponse(b"{}", content_type="text/plain"),
        FakeHTTPResponse(b"x" * 101),
    )
    for response in responses:
        monkeypatch.setattr(
            transport._opener,
            "open",
            lambda *_args, _response=response, **_kwargs: _response,
        )
        with pytest.raises(AdvisorValidationError):
            transport.request("GET", "/api/version", None, timeout=1.0, max_bytes=100)

    errors = (
        TimeoutError(),
        urllib.error.HTTPError("http://local", 503, "down", {}, None),
        urllib.error.URLError("down"),
    )
    for error in errors:

        def fail(*_args: Any, _error: Exception = error, **_kwargs: Any) -> Any:
            raise _error

        monkeypatch.setattr(transport._opener, "open", fail)
        with pytest.raises(AdvisorError):
            transport.request("GET", "/api/version", None, timeout=1.0, max_bytes=100)


def test_urllib_transport_disables_redirects(monkeypatch: Any) -> None:
    transport = UrllibAdvisorTransport("http://127.0.0.1:11434")
    assert transport._proxy_handler.proxies == {}
    redirects = [
        handler
        for handler in transport._opener.handlers
        if isinstance(handler, urllib.request.HTTPRedirectHandler)
    ]
    assert len(redirects) == 1
    assert redirects[0].redirect_request(None, None, None, None, None, None) is None

    calls: list[str] = []

    def redirect_error(request: Any, **_kwargs: Any) -> Any:
        calls.append(request.full_url)
        raise urllib.error.HTTPError(
            request.full_url,
            302,
            "redirect forbidden",
            {"Location": "https://example.com/collect"},
            None,
        )

    monkeypatch.setattr(transport._opener, "open", redirect_error)
    with pytest.raises(AdvisorError):
        transport.request("GET", "/api/version", None, timeout=1.0, max_bytes=100)
    assert calls == ["http://127.0.0.1:11434/api/version"]


class ObjectTransport(FakeTransport):
    def __init__(self, objects: dict[str, Any]) -> None:
        super().__init__()
        self.objects = objects

    def request(self, method: str, path: str, payload: bytes | None, **kwargs: Any) -> bytes:
        if path in self.objects:
            value = self.objects[path]
            return value if isinstance(value, bytes) else json.dumps(value).encode()
        return super().request(method, path, payload, **kwargs)


@pytest.mark.parametrize(
    ("objects", "expected"),
    [
        ({"/api/version": {"version": ""}}, "version"),
        ({"/api/tags": {"models": {}}}, "model-list"),
        ({"/api/tags": {"models": ["bad"]}}, "invalid entry"),
        (
            {
                "/api/tags": {
                    "models": [
                        {
                            "name": "granite4.1:3b",
                            "digest": "a" * 64,
                            "size": 1,
                            "details": {"quantization_level": "Q4_K_M"},
                        },
                        {
                            "name": "granite4.1:3b",
                            "digest": "a" * 64,
                            "size": 1,
                            "details": {"quantization_level": "Q4_K_M"},
                        },
                    ]
                }
            },
            "duplicate",
        ),
        (
            {
                "/api/tags": {
                    "models": [
                        {
                            "name": "granite4.1:3b",
                            "digest": "a" * 64,
                            "size": 1,
                            "details": {"quantization_level": "Q8_0"},
                        }
                    ]
                }
            },
            "Q4_K_M",
        ),
        (
            {
                "/api/tags": {
                    "models": [
                        {
                            "name": "granite4.1:3b",
                            "digest": "a" * 64,
                            "size": True,
                            "details": {"quantization_level": "Q4_K_M"},
                        }
                    ]
                }
            },
            "size",
        ),
    ],
)
def test_model_verification_rejects_malformed_metadata(
    objects: dict[str, Any], expected: str
) -> None:
    with pytest.raises(AdvisorModelError, match=expected):
        OllamaPolicyAdvisor(advisor_config(), transport=ObjectTransport(objects)).verify()


@pytest.mark.parametrize(
    "ps",
    [
        {"models": {}},
        {"models": []},
        {
            "models": [
                {
                    "name": "granite4.1:3b",
                    "digest": "b" * 64,
                    "size": 1,
                    "size_vram": 0,
                }
            ]
        },
    ],
)
def test_running_model_memory_is_strict(ps: dict[str, Any]) -> None:
    advisor = OllamaPolicyAdvisor(
        advisor_config(),
        transport=ObjectTransport({"/api/ps": ps}),
    )
    with pytest.raises(AdvisorValidationError):
        advisor._model_memory()


@pytest.mark.parametrize(
    "outer",
    [
        {"model": "other:tag", "message": {"role": "assistant", "content": "{}"}, "done": True},
        {
            "model": "granite4.1:3b",
            "message": {"role": "assistant", "content": "{}"},
            "done": False,
        },
        {"model": "granite4.1:3b", "message": {}, "done": True},
        {
            "model": "granite4.1:3b",
            "message": {"role": "assistant", "content": []},
            "done": True,
        },
    ],
)
def test_chat_outer_contract_is_rejected(outer: dict[str, Any]) -> None:
    transport = FakeTransport([json.dumps(outer).encode()])
    advisor = OllamaPolicyAdvisor(
        advisor_config(retry_count=0),
        transport=transport,
    )
    runtime = advisor.verify()
    _, _, bundle = diagnostic_bundle()
    with pytest.raises(AdvisorError):
        advisor.advise(
            bundle.summary,
            allowed_bin_ids=bundle.allowed_bin_ids,
            run_seed=1,
            batch_id=1,
            runtime=runtime,
        )


def test_validation_model_and_helper_edges() -> None:
    _, context, bundle = diagnostic_bundle()
    payload = bundle.summary.model_dump(mode="python")
    with pytest.raises(ValidationError, match="ordered"):
        ScoreDistribution(minimum=0.0, q10=0.8, median=0.2, q90=0.9, maximum=1.0)

    bad_mae = copy.deepcopy(payload)
    bad_mae["validation_mae_trend"] = [-1.0]
    with pytest.raises(ValidationError, match="finite"):
        DiagnosticSummary.model_validate(bad_mae, strict=True)
    duplicate = copy.deepcopy(payload)
    duplicate["bins"].append(copy.deepcopy(duplicate["bins"][0]))
    with pytest.raises(ValidationError, match="unique"):
        DiagnosticSummary.model_validate(duplicate, strict=True)
    bad_scores = copy.deepcopy(payload)
    del bad_scores["acquisition_score_distributions"]["balanced_combined"]
    with pytest.raises(ValidationError, match="invalid set"):
        DiagnosticSummary.model_validate(bad_scores, strict=True)

    with pytest.raises(ValueError, match="aligned"):
        from pricing_mapper.advisor import build_diagnostic_summary

        build_diagnostic_summary(
            context=context,
            validation_rows=[],
            batch_history=[],
            domain=context.domain,
        )
    with pytest.raises(AdvisorValidationError, match="size"):
        assert_diagnostic_payload_safe({"safe": "x" * 64_001})
    with pytest.raises(AdvisorValidationError, match="UTF-8"):
        validate_policy_response(b"\xff", frozenset())
    with pytest.raises(AdvisorValidationError, match="object"):
        validate_policy_response("[]", frozenset())
    with pytest.raises(AdvisorValidationError, match="unsupported"):
        system_prompt("future")
    with pytest.raises(ValueError, match="non-negative"):
        derived_advisor_seed(-1, 0)

    with pytest.raises(ValueError, match="negative"):
        apply_policy(
            context,
            bundle,
            PolicyResponse(policy="balanced", bin_boosts=[]),
            count=-1,
            greedy_diversity_weight=0.5,
        )
    with pytest.raises(ValueError, match="within"):
        apply_policy(
            context,
            bundle,
            PolicyResponse(policy="balanced", bin_boosts=[]),
            count=1,
            greedy_diversity_weight=2.0,
        )
    corrupt = DiagnosticBundle(
        summary=bundle.summary,
        candidate_bin_memberships={
            **bundle.candidate_bin_memberships,
            next(iter(bundle.allowed_bin_ids)): np.asarray([True]),
        },
    )
    selected_bin = next(iter(bundle.allowed_bin_ids))
    with pytest.raises(AdvisorValidationError, match="membership"):
        apply_policy(
            context,
            corrupt,
            PolicyResponse(
                policy="balanced",
                bin_boosts=[{"bin_id": selected_bin, "boost": 1.1}],
            ),
            count=1,
            greedy_diversity_weight=0.5,
        )


def test_injected_decision_provenance_is_revalidated() -> None:
    advisor = OllamaPolicyAdvisor(advisor_config(), transport=FakeTransport())
    runtime = advisor.verify()
    _, _, bundle = diagnostic_bundle()
    decision = advisor.advise(
        bundle.summary,
        allowed_bin_ids=bundle.allowed_bin_ids,
        run_seed=17,
        batch_id=2,
        runtime=runtime,
    )
    assert (
        validate_advisor_decision(
            decision,
            allowed_bin_ids=bundle.allowed_bin_ids,
            batch_id=2,
            run_seed=17,
            runtime=runtime,
        )["batch_id"]
        == 2
    )

    mutations: list[tuple[str, Any]] = [
        ("batch_id", 3),
        ("generation.seed", 0),
        ("runtime.ollama_version", "other"),
        ("memory.digest", "sha256:" + "b" * 64),
        ("attempts", 2),
        ("total_latency_ms", 999.0),
    ]
    for path, value in mutations:
        raw = copy.deepcopy(decision.record)
        target: dict[str, Any] = raw  # type: ignore[assignment]
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
        altered = AdvisorDecision(response=decision.response, record=raw)
        with pytest.raises(AdvisorValidationError):
            validate_advisor_decision(
                altered,
                allowed_bin_ids=bundle.allowed_bin_ids,
                batch_id=2,
                run_seed=17,
                runtime=runtime,
            )

    mismatched = AdvisorDecision(
        response=PolicyResponse(policy="explore", bin_boosts=[]),
        record=decision.record,
    )
    with pytest.raises(AdvisorValidationError, match="response differs"):
        validate_advisor_decision(
            mismatched,
            allowed_bin_ids=bundle.allowed_bin_ids,
            batch_id=2,
            run_seed=17,
            runtime=runtime,
        )
