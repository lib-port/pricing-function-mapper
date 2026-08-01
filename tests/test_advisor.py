from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import tiny_config

from pricing_mapper.acquisition import build_context
from pricing_mapper.advisor import (
    MAX_RESPONSE_BYTES,
    BinBoost,
    OllamaPolicyAdvisor,
    PolicyResponse,
    apply_policy,
    assert_diagnostic_payload_safe,
    build_diagnostic_summary,
    policy_response_schema,
    system_prompt,
    validate_policy_response,
)
from pricing_mapper.config import OllamaConfig, OptimizerConfig
from pricing_mapper.domain import CarQuoteInput, DomainSpec
from pricing_mapper.exceptions import (
    AdvisorError,
    AdvisorModelError,
    AdvisorUnavailable,
    AdvisorValidationError,
    ProviderRejected,
)
from pricing_mapper.orchestration import MappingRun
from pricing_mapper.provider import reference_car_quote

MODEL_DIGEST = "sha256:" + "a" * 64


class FakeTransport:
    def __init__(self, chat_responses: list[bytes | Exception] | None = None) -> None:
        self.chat_responses = list(chat_responses or [])
        self.requests: list[tuple[str, str, bytes | None, float, int]] = []

    def request(
        self,
        method: str,
        path: str,
        payload: bytes | None,
        *,
        timeout: float,
        max_bytes: int,
    ) -> bytes:
        self.requests.append((method, path, payload, timeout, max_bytes))
        if path == "/api/version":
            return b'{"version":"0.32.0"}'
        if path == "/api/tags":
            return json.dumps(
                {
                    "models": [
                        {
                            "name": "granite4.1:3b",
                            "model": "granite4.1:3b",
                            "digest": "a" * 64,
                            "size": 2_099_501_664,
                            "details": {"quantization_level": "Q4_K_M"},
                        }
                    ]
                }
            ).encode()
        if path == "/api/ps":
            return json.dumps(
                {
                    "models": [
                        {
                            "name": "granite4.1:3b",
                            "model": "granite4.1:3b",
                            "digest": "a" * 64,
                            "size": 2_350_000_000,
                            "size_vram": 0,
                        }
                    ]
                }
            ).encode()
        if path != "/api/chat":
            raise AssertionError(path)
        response: bytes | Exception = (
            self.chat_responses.pop(0) if self.chat_responses else chat_response()
        )
        if isinstance(response, Exception):
            raise response
        return response


def chat_response(
    content: str = '{"policy":"balanced","bin_boosts":[]}',
) -> bytes:
    return json.dumps(
        {
            "model": "granite4.1:3b",
            "message": {"role": "assistant", "content": content},
            "done": True,
        },
        separators=(",", ":"),
    ).encode()


def advisor_config(**updates: Any) -> OllamaConfig:
    values = {"required_digest": MODEL_DIGEST, **updates}
    return OllamaConfig(**values)


def diagnostic_bundle() -> Any:
    domain = DomainSpec.default()
    rows = domain.sample_lhs(14, np.random.default_rng(11))
    mapping_rows = rows[:5]
    validation_rows = rows[5:9]
    candidates = rows[9:]
    context = build_context(
        candidate_rows=candidates,
        training_rows=mapping_rows,
        predictive_std=np.asarray([0.2, 0.8, 0.5, 0.4, 0.9]),
        training_residuals=np.asarray([10.0, -5.0, 20.0, 0.0]),
        residual_anchor_rows=validation_rows,
        domain=domain,
    )
    history = [
        {
            "batch_id": 0,
            "status": "evaluated",
            "advisor": None,
            "metrics": {"validation_mae": 100.0},
        },
        {
            "batch_id": 1,
            "status": "evaluated",
            "advisor": {"response": {"policy": "uncertainty"}},
            "metrics": {"validation_mae": 90.0},
        },
    ]
    bundle = build_diagnostic_summary(
        context=context,
        validation_rows=validation_rows,
        batch_history=history,
        domain=domain,
    )
    return domain, context, bundle


def test_diagnostics_prompt_schema_and_leakage_guards() -> None:
    _, _, bundle = diagnostic_bundle()
    payload = bundle.summary.model_dump(mode="json")
    assert payload["mapping_sample_count"] == 5
    assert payload["validation_mae_trend"] == [100.0, 90.0]
    assert payload["previous_policy"] == {
        "policy": "uncertainty",
        "relative_validation_mae_improvement": 0.1,
    }
    assert {item["field_name"] for item in payload["bins"]} == set(
        DomainSpec.default().numeric
    ) | set(DomainSpec.default().categorical)
    assert "policy-advisor-v1" in system_prompt("policy-advisor-v1")
    schema = policy_response_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]["policy"]["enum"]) == {
        "balanced",
        "uncertainty",
        "residual",
        "explore",
        "exploit",
    }
    assert_diagnostic_payload_safe(payload)
    for unsafe in (
        {"premium": 100.0},
        {"nested": {"rows": []}},
        {"audit": {"metric": 1.0}},
        {"safe": float("nan")},
    ):
        with pytest.raises(AdvisorValidationError):
            assert_diagnostic_payload_safe(unsafe)


@pytest.mark.parametrize(
    "raw",
    [
        '{"policy":"unknown","bin_boosts":[]}',
        '{"policy":"balanced","bin_boosts":[],"extra":true}',
        '{"policy":"balanced","bin_boosts":[{"bin_id":"x","boost":1.2}]}',
        '{"policy":"balanced","bin_boosts":[{"bin_id":"x","boost":1.1},'
        '{"bin_id":"x","boost":1.25}]}',
        '{"policy":"balanced","policy":"explore","bin_boosts":[]}',
        '{"policy":"balanced","bin_boosts":[],"number":NaN}',
        '{"policy":"balanced"}',
    ],
)
def test_policy_response_is_strict(raw: str) -> None:
    with pytest.raises(AdvisorValidationError):
        validate_policy_response(raw, frozenset({"x"}))


def test_policy_response_bin_allow_list_and_application() -> None:
    _, context, bundle = diagnostic_bundle()
    allowed = sorted(bundle.allowed_bin_ids)
    response = validate_policy_response(
        json.dumps(
            {
                "policy": "explore",
                "bin_boosts": [{"bin_id": allowed[0], "boost": 1.25}],
            }
        ),
        bundle.allowed_bin_ids,
    )
    application = apply_policy(
        context,
        bundle,
        response,
        count=5,
        greedy_diversity_weight=0.5,
    )
    assert application.exploration_count == 2
    assert len(application.selected_indices) == 3
    assert len(set(application.selected_indices)) == 3
    with pytest.raises(AdvisorValidationError, match="allow-list"):
        validate_policy_response(
            '{"policy":"balanced","bin_boosts":[{"bin_id":"invented","boost":1.1}]}',
            bundle.allowed_bin_ids,
        )
    with pytest.raises(AdvisorValidationError, match="allow-list"):
        apply_policy(
            context,
            bundle,
            PolicyResponse(
                policy="balanced",
                bin_boosts=[BinBoost(bin_id="invented", boost=1.1)],
            ),
            count=2,
            greedy_diversity_weight=0.5,
        )


def test_ollama_contract_verification_request_and_retries() -> None:
    invalid = chat_response('{"policy":"unknown","bin_boosts":[]}')
    transport = FakeTransport([invalid, invalid, chat_response()])
    advisor = OllamaPolicyAdvisor(advisor_config(), transport=transport)
    runtime = advisor.verify()
    _, _, bundle = diagnostic_bundle()
    decision = advisor.advise(
        bundle.summary,
        allowed_bin_ids=bundle.allowed_bin_ids,
        run_seed=17,
        batch_id=3,
        runtime=runtime,
    )
    assert decision.response.policy == "balanced"
    assert decision.record["attempts"] == 3
    assert decision.record["runtime"]["digest"] == MODEL_DIGEST
    assert decision.record["memory"]["resident_size_bytes"] == 2_350_000_000
    chat_requests = [request for request in transport.requests if request[1] == "/api/chat"]
    assert len(chat_requests) == 3
    request = json.loads(chat_requests[0][2] or b"{}")
    assert request["stream"] is False
    assert request["options"]["temperature"] == 0
    assert request["options"]["num_predict"] == 128
    assert request["format"]["additionalProperties"] is False
    outbound = json.loads(request["messages"][1]["content"])
    assert_diagnostic_payload_safe(outbound)


@pytest.mark.parametrize(
    "responses",
    [
        [AdvisorUnavailable("down")] * 3,
        [chat_response("not-json")] * 3,
        [b"x" * (MAX_RESPONSE_BYTES + 1)] * 3,
    ],
)
def test_advisor_exhaustion_fails_closed(responses: list[bytes | Exception]) -> None:
    transport = FakeTransport(responses)
    advisor = OllamaPolicyAdvisor(advisor_config(), transport=transport)
    runtime = advisor.verify()
    _, _, bundle = diagnostic_bundle()
    with pytest.raises(AdvisorError, match="3 attempts"):
        advisor.advise(
            bundle.summary,
            allowed_bin_ids=bundle.allowed_bin_ids,
            run_seed=17,
            batch_id=1,
            runtime=runtime,
        )
    assert len([item for item in transport.requests if item[1] == "/api/chat"]) == 3


def test_model_list_missing_and_wrong_digest_are_rejected() -> None:
    missing = FakeTransport()

    def missing_request(*args: Any, **kwargs: Any) -> bytes:
        if args[1] == "/api/version":
            return b'{"version":"0.32.0"}'
        return b'{"models":[]}'

    missing.request = missing_request  # type: ignore[method-assign]
    with pytest.raises(AdvisorModelError, match="not installed"):
        OllamaPolicyAdvisor(advisor_config(), transport=missing).verify()

    with pytest.raises(AdvisorModelError, match="digest"):
        OllamaPolicyAdvisor(
            advisor_config(required_digest="sha256:" + "b" * 64),
            transport=FakeTransport(),
        ).verify()


class CountingCrashProvider:
    provider_id = "test.hybrid-crash"
    thread_safe = False
    max_concurrency = 1

    def __init__(self, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.calls = 0
        self.failed = False

    def __call__(self, quote: CarQuoteInput) -> float:
        self.calls += 1
        if self.fail_at == self.calls and not self.failed:
            self.failed = True
            raise ProviderRejected("simulated boundary")
        return reference_car_quote(quote)


def hybrid_config(path: Path) -> Any:
    base = tiny_config(path, strategy="bayesian")
    return base.model_copy(update={"optimizer": OptimizerConfig(ollama=advisor_config())})


def mapping_hashes(database: Path) -> list[str]:
    connection = sqlite3.connect(database)
    try:
        return [
            str(row[0])
            for row in connection.execute(
                "SELECT row_hash FROM samples WHERE split = 'mapping' ORDER BY ordinal"
            )
        ]
    finally:
        connection.close()


def test_hybrid_resume_reuses_stored_decision_and_candidate_batch(tmp_path: Path) -> None:
    resumed_transport = FakeTransport()
    resumed_advisor = OllamaPolicyAdvisor(
        advisor_config(),
        transport=resumed_transport,
    )
    provider = CountingCrashProvider(fail_at=16)
    resumed = MappingRun(
        hybrid_config(tmp_path / "resumed"),
        provider=provider,
        advisor=resumed_advisor,
        run_id="hybrid-resume",
    )
    with pytest.raises(ProviderRejected):
        resumed.run()
    connection = sqlite3.connect(resumed.state_database)
    try:
        stored = connection.execute(
            "SELECT advisor_json FROM batches WHERE batch_id = 1"
        ).fetchone()[0]
        assert stored is not None
    finally:
        connection.close()
    resumed_result = resumed.resume()
    resumed_chats = [item for item in resumed_transport.requests if item[1] == "/api/chat"]
    assert len(resumed_chats) == 2

    full_transport = FakeTransport()
    full_result = MappingRun(
        hybrid_config(tmp_path / "full"),
        provider=CountingCrashProvider(),
        advisor=OllamaPolicyAdvisor(advisor_config(), transport=full_transport),
        run_id="hybrid-full",
    ).run()
    assert mapping_hashes(resumed_result.state_database) == mapping_hashes(
        full_result.state_database
    )
    assert resumed_result.evaluation_report["advisor"]["decision_count"] == 2
    assert (
        resumed_result.evaluation_report["advisor"]["maximum_resident_model_bytes"] == 2_350_000_000
    )
    assert resumed_result.evaluation_report["advisor"]["data_shared"]["raw_premiums"] is False


def test_failed_advice_registers_no_batch_and_consumes_no_post_initial_quotes(
    tmp_path: Path,
) -> None:
    transport = FakeTransport([chat_response("bad")] * 3)
    provider = CountingCrashProvider()
    run = MappingRun(
        hybrid_config(tmp_path),
        provider=provider,
        advisor=OllamaPolicyAdvisor(advisor_config(), transport=transport),
        run_id="hybrid-fail-closed",
    )
    with pytest.raises(AdvisorError):
        run.run()
    connection = sqlite3.connect(run.state_database)
    try:
        assert connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM samples WHERE split = 'mapping'").fetchone()[0]
            == 6
        )
    finally:
        connection.close()
    assert provider.calls == 15


def test_unavailable_container_fails_before_state_or_provider_calls(tmp_path: Path) -> None:
    transport = FakeTransport()

    def unavailable(*_: Any, **__: Any) -> bytes:
        raise AdvisorUnavailable("offline")

    transport.request = unavailable  # type: ignore[method-assign]
    provider = CountingCrashProvider()
    run = MappingRun(
        hybrid_config(tmp_path),
        provider=provider,
        advisor=OllamaPolicyAdvisor(advisor_config(), transport=transport),
        run_id="missing-container",
    )
    with pytest.raises(AdvisorModelError):
        run.run()
    assert not run.state_database.exists()
    assert provider.calls == 0


def test_bayesian_only_uses_balanced_local_policy_without_advisor(tmp_path: Path) -> None:
    result = MappingRun(
        tiny_config(tmp_path, strategy="bayesian"),
        provider=CountingCrashProvider(),
        run_id="bayesian-only",
    ).run()
    assert result.mapping_samples == 10
    assert result.evaluation_report["advisor"]["enabled"] is False
    assert result.evaluation_report["advisor"]["decision_count"] == 0
    assert all(batch["advisor"] is None for batch in result.evaluation_report["batch_history"])
