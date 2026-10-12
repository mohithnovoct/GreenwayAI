import asyncio
import json
import json as json_module

import httpx
import pytest

from ingress.normalization import count_tokens
from verify.gemini_probe_round2 import (
    ORDERS,
    PROMPT_FILE,
    request_with_retries,
    response_metrics,
)


def test_probe_prompts_are_within_requested_token_range():
    prompts = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))

    assert len(prompts) == 3
    assert all(
        60 <= count_tokens(item["prompt"], "gemini-2.5-flash") <= 150
        for item in prompts
    )


def test_probe_order_is_counterbalanced():
    assert ORDERS == (
        ("baseline", "optimized"),
        ("optimized", "baseline"),
        ("baseline", "optimized"),
    )


def test_probe_metrics_count_hidden_tokens_and_structure():
    metrics = response_metrics(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "# Summary\n\n- Point one\n- Point two\n  - Detail\n"
                    },
                }
            ],
            "usage": {"completion_tokens": 12, "total_tokens": 115},
        },
        prompt_tokens=100,
    )

    assert metrics == {
        "finish_reason": "length",
        "completion_tokens": 12,
        "hidden_tokens": 3,
        "word_count": 10,
        "top_level_bullet_count": 2,
        "total_bullet_lines": 3,
        "heading_count": 1,
        "counts_are_lower_bounds": True,
    }


def test_retry_records_failed_attempt_and_reuses_same_sample(monkeypatch, tmp_path):
    class StubClient:
        def __init__(self, attempts_path):
            self.attempts_path = attempts_path
            self.call_count = 0
            self.responses = [
                httpx.Response(503, text="temporary unavailable", request=httpx.Request("POST", "https://example.test")),
                httpx.Response(200, text='{"ok":true}', request=httpx.Request("POST", "https://example.test")),
            ]

        async def post(self, url, *, json):
            persisted = json_module.loads(self.attempts_path.read_text(encoding="utf-8"))
            assert len(persisted["attempts"]) == self.call_count + 1
            assert persisted["attempts"][-1]["attempted_at_utc"]
            assert "status_code" not in persisted["attempts"][-1]
            self.call_count += 1
            return self.responses.pop(0)

    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr("verify.gemini_probe_round2.asyncio.sleep", fake_sleep)
    attempts = []
    sample = {"sample_number": 2, "shape": "focused_explanation", "arm": "optimized"}
    attempts_path = tmp_path / "attempts.json"
    response, _ = asyncio.run(
        request_with_retries(
            StubClient(attempts_path),
            url="https://example.test",
            payload={"sample": 2},
            sample=sample,
            attempts=attempts,
            attempts_path=attempts_path,
            last_request_at=None,
            attempt_budget=3,
        )
    )

    assert response.status_code == 200
    assert [attempt["sample_number"] for attempt in attempts] == [2, 2]
    assert [attempt["attempt_number_for_sample"] for attempt in attempts] == [1, 2]
    assert [attempt["status_code"] for attempt in attempts] == [503, 200]
    assert all(attempt["attempted_at_utc"] for attempt in attempts)
    assert attempts[0]["response_body"] == "temporary unavailable"
    assert sleep_calls == [pytest.approx(30)]
