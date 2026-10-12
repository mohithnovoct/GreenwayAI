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
    monkeypatch.setenv("PROMPT_OPT_WORD_CAP", "87")
    monkeypatch.setenv("PROMPT_OPT_COMBINE_VARIATIONS", "true")
    result = optimize_gateway_request(request_for("Give me foo alternatives; ask me again for each one."))
    text = result.request.raw_payload["messages"][0]["content"]

    assert "foo (standard term)" in text
    assert "all requested variations in one response" in text
    assert text.startswith("Act as a precise domain expert.")
    assert "Answer in under 87 words." in text
    assert "bullet" not in text
    assert "jargon-expanded" in result.changes
    assert "variations-combined" in result.changes
    assert "word-cap-added" in result.changes
    assert result.optimized_tokens > result.original_tokens


def test_word_cap_is_added_even_when_prompt_says_brief(monkeypatch):
    monkeypatch.delenv("PROMPT_OPT_COMBINE_VARIATIONS", raising=False)
    monkeypatch.setenv("PROMPT_OPT_WORD_CAP", "80")
    result = optimize_gateway_request(request_for("Give one brief example of safe retries."))
    text = result.request.raw_payload["messages"][0]["content"]

    assert "Answer in under 80 words." in text
    assert "variations-combined" not in result.changes
    assert "word-cap-added" in result.changes


def test_existing_numeric_word_cap_is_preserved(monkeypatch):
    monkeypatch.setenv("PROMPT_OPT_WORD_CAP", "80")
    result = optimize_gateway_request(request_for("Explain this in under 90 words."))

    assert "under 90 words" in result.optimized_prompt
    assert "under 80 words" not in result.optimized_prompt
    assert "word-cap-added" not in result.changes


def test_preserves_existing_role_without_treating_bullet_cap_as_word_cap(monkeypatch):
    monkeypatch.setenv("PROMPT_OPT_WORD_CAP", "150")
    result = optimize_gateway_request(request_for("As a database expert, answer in 3 bullets."))
    text = result.request.raw_payload["messages"][0]["content"]

    assert not text.startswith("Act as a precise domain expert.")
    assert "Answer in under 150 words." in text
    assert "role-added" not in result.changes
    assert "word-cap-added" in result.changes


def test_short_one_paragraph_assessment_is_not_a_persona_cue(monkeypatch):
    monkeypatch.setenv("PROMPT_OPT_WORD_CAP", "150")
    monkeypatch.setenv("PROMPT_OPT_COMBINE_VARIATIONS", "false")
    sample_six_prompt = (
        "A customer-facing API's median latency stayed flat, but its 95th "
        "percentile doubled after a new dependency was introduced. Analyze "
        "two plausible causes, name the first two measurements you would "
        "collect to locate the delay, and recommend one low-risk mitigation "
        "while investigating. State one caveat that could make those "
        "measurements misleading. Preserve correctness and avoid speculative "
        "tuning. Answer as a short one-paragraph assessment rather than a checklist."
    )

    result = optimize_gateway_request(request_for(sample_six_prompt))

    assert "role-added" in result.changes
    assert "Act as a precise domain expert." in result.optimized_prompt
    assert "word-cap-added" in result.changes
    assert "Answer in under 150 words." in result.optimized_prompt


def test_explicit_expert_persona_is_preserved():
    result = optimize_gateway_request(request_for("As a database expert, explain this."))

    assert "role-added" not in result.changes
    assert not result.optimized_prompt.startswith("Act as a precise domain expert.")


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
    monkeypatch.setenv("PROMPT_OPT_WORD_CAP", "150")
    monkeypatch.setenv("PROMPT_OPT_COMBINE_VARIATIONS", "false")
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
    assert response.headers["X-Greenway-Prompt-Optimization-Changes"] == (
        "role-added,word-cap-added"
    )
    records = [json.loads(record.message) for record in caplog.records if '"prompt_optimization_arm"' in record.message]
    assert records
    assert records[-1]["original_input_tokens"] < records[-1]["optimized_input_tokens"]
    assert records[-1]["original_prompt"] != records[-1]["optimized_prompt"]
