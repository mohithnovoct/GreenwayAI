import json
import logging
import random

from fastapi.testclient import TestClient

from ingress.app import app
from optimizers.prompt_optimizer import optimize_gateway_request, should_optimize
from ingress.normalization import build_gateway_request


def request_for(content: str, **extra):
    payload = {"model": "test-model", "messages": [{"role": "user", "content": content}], **extra}
    return build_gateway_request(payload, None)


def test_rewrites_requested_prompt_practices(monkeypatch):
    monkeypatch.setenv("PROMPT_OPT_JARGON_MAP", '{"foo": "standard term"}')
    result = optimize_gateway_request(request_for("Give me foo alternatives; ask me again for each one."))
    text = result.request.raw_payload["messages"][0]["content"]

    assert "foo (standard term)" in text
    assert "all requested variations in one response" in text
    assert text.startswith("Act as a precise domain expert.")
    assert "no more than 5 bullets" in text
    assert "jargon-expanded" in result.changes
    assert "variations-combined" in result.changes
    assert result.optimized_tokens > result.original_tokens


def test_preserves_existing_role_and_length_constraints():
    result = optimize_gateway_request(request_for("As a database expert, answer in 3 bullets."))
    text = result.request.raw_payload["messages"][0]["content"]

    assert not text.startswith("Act as a precise domain expert.")
    assert "no more than 5 bullets" not in text
    assert "role-added" not in result.changes
    assert "length-constrained" not in result.changes


def test_adds_structured_format_only_when_requested():
    result = optimize_gateway_request(request_for("Summarize this.", response_format={"type": "json_object"}))
    text = result.request.raw_payload["messages"][0]["content"]

    assert "Return only valid JSON" in text
    assert "format-specified" in result.changes


def test_ab_routing_respects_flag_and_percentage(monkeypatch):
    monkeypatch.setenv("PROMPT_OPT_ENABLED", "true")
    assert should_optimize(percentage=0, rng=random.Random(1)) is False
    assert should_optimize(percentage=100, rng=random.Random(1)) is True
    assert should_optimize(percentage=50, rng=random.Random(1)) is True
    monkeypatch.setenv("PROMPT_OPT_ENABLED", "false")
    assert should_optimize(percentage=100, rng=random.Random(1)) is False


def test_selected_arm_is_reported_and_logged(monkeypatch, caplog):
    monkeypatch.setenv("PROMPT_OPT_ENABLED", "true")
    monkeypatch.setenv("PROMPT_OPT_AB_PERCENT", "100")
    monkeypatch.setenv("VLLM_BACKEND_URL", "")
    monkeypatch.setenv("GREENWAY_LOG_PROMPTS", "true")
    caplog.set_level(logging.INFO)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "mock", "messages": [{"role": "user", "content": "Summarize alternatives."}]},
        )

    assert response.status_code == 200
    assert response.headers["X-Greenway-Prompt-Optimization"] == "optimized"
    records = [json.loads(record.message) for record in caplog.records if '"prompt_optimization_arm"' in record.message]
    assert records
    assert records[-1]["original_input_tokens"] < records[-1]["optimized_input_tokens"]
    assert records[-1]["original_prompt"] != records[-1]["optimized_prompt"]
