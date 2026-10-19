import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import RedisError

from ingress import app as ingress_app
from optimizers.semantic_cache import (
    DEFAULT_CACHE_TENANT_NAMESPACE,
    DEFAULT_CACHE_TTL_SECONDS,
    CachedResponse,
    ExactResponseCache,
    SentenceTransformerEmbedder,
    exact_cache_key,
    exact_request_is_eligible,
    exact_response_is_eligible,
    exact_response_token_counts,
)


@pytest.fixture
def gateway_harness(monkeypatch):
    class FakeRedis:
        def __init__(self):
            self.values = {}
            self.get_calls = 0
            self.set_calls = 0
            self.fail_get = False
            self.fail_set = False

        async def get(self, key):
            self.get_calls += 1
            if self.fail_get:
                raise RedisError("unavailable")
            return self.values.get(key)

        async def set(self, key, value, ex):
            self.set_calls += 1
            if self.fail_set:
                raise RedisError("unavailable")
            self.values[key] = value

        async def aclose(self):
            return None

    class FakeEmbedder:
        def __init__(self, model_id, revision):
            self.model_id = model_id
            self.dimension = 384

    class FakeBackendClient:
        def __init__(self, original_client):
            self.original_client = original_client
            self.backend_calls = 0
            self.delay_seconds = 0
            self.responses = []

        def build_request(self, method, url, *, json, headers):
            return httpx.Request(method, url, json=json, headers=headers)

        async def send(self, _request, stream=False):
            self.backend_calls += 1
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            if self.responses:
                status_code, body = self.responses.pop(0)
                return httpx.Response(
                    status_code,
                    content=body,
                    headers={"content-type": "application/json"},
                )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "cached result"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 2,
                        "total_tokens": 4,
                    },
                },
            )

        async def aclose(self):
            await self.original_client.aclose()

    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setenv("CACHE_TENANT_NAMESPACE", "gateway-test")
    monkeypatch.setenv("GREENWAY_CARBON_DISABLED", "true")
    fake_redis = FakeRedis()
    monkeypatch.setattr(
        ingress_app.Redis,
        "from_url",
        classmethod(lambda cls, *args, **kwargs: fake_redis),
    )
    monkeypatch.setattr(ingress_app, "SentenceTransformerEmbedder", FakeEmbedder)
    monkeypatch.setattr(
        ingress_app,
        "backend_configuration",
        lambda: (
            "https://provider.example/v1/chat/completions",
            "gemini-2.5-flash",
            {},
            True,
        ),
    )
    recorded_telemetry = []

    def record_metrics(_metrics, **values):
        recorded_telemetry.append(values)

    monkeypatch.setattr(ingress_app, "register_baseline_metrics", lambda: {})
    monkeypatch.setattr(ingress_app, "record_request_metrics", record_metrics)

    with TestClient(ingress_app.app) as client:
        backend = FakeBackendClient(ingress_app.app.state.http_client)
        ingress_app.app.state.http_client = backend
        yield client, fake_redis, backend, recorded_telemetry


def test_exact_cache_key_collision_matrix_covers_request_backend_and_arm():
    payload = {
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "hello", "name": "client"}],
        "temperature": 0,
        "max_tokens": 80,
        "stop": ["END"],
        "top_p": 0.8,
        "seed": 7,
        "response_format": {"type": "json_object"},
    }

    key = exact_cache_key(payload, DEFAULT_CACHE_TENANT_NAMESPACE, "test-backend")

    variants = [
        ({**payload, "model": "other-model"}, "model"),
        ({**payload, "messages": [{"role": "system", "content": "hello", "name": "client"}]}, "message role"),
        ({**payload, "messages": [{"role": "user", "content": "different", "name": "client"}]}, "message content"),
        ({**payload, "messages": [{"role": "user", "content": "hello", "name": "other"}]}, "message name"),
        ({**payload, "messages": [*payload["messages"], {"role": "assistant", "content": "prior"}]}, "message list"),
        ({**payload, "max_tokens": 81}, "max_tokens"),
        ({**payload, "stop": ["STOP"]}, "stop"),
        ({**payload, "top_p": 0.7}, "top_p"),
        ({**payload, "seed": 8}, "seed"),
        ({**payload, "response_format": {"type": "text"}}, "response_format"),
    ]
    for variant, changed_field in variants:
        assert key != exact_cache_key(
            variant,
            DEFAULT_CACHE_TENANT_NAMESPACE,
            "test-backend",
        ), changed_field
    assert key != exact_cache_key(payload, "another-deployment", "test-backend")
    assert key != exact_cache_key(payload, DEFAULT_CACHE_TENANT_NAMESPACE, "another-backend")
    assert key != exact_cache_key(
        payload,
        DEFAULT_CACHE_TENANT_NAMESPACE,
        "test-backend",
        optimizer_arm="optimized",
    )
    assert key != exact_cache_key(
        payload,
        DEFAULT_CACHE_TENANT_NAMESPACE,
        "test-backend",
        tenant_id="trusted-tenant",
    )
    assert ":exact:" in key
    assert DEFAULT_CACHE_TTL_SECONDS == 3600


def test_exact_cache_key_rejects_empty_namespace():
    try:
        exact_cache_key({"messages": []}, "  ", "test-backend")
    except ValueError as error:
        assert str(error) == "Cache tenant namespace must not be empty"
    else:
        raise AssertionError("empty tenant namespace must be rejected")


def test_exact_cache_request_policy_is_deterministic_non_streaming_and_tool_free():
    base = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        "stream": False,
    }

    assert exact_request_is_eligible(base)
    assert not exact_request_is_eligible({**base, "temperature": 0.1})
    assert not exact_request_is_eligible({**base, "stream": True})
    assert not exact_request_is_eligible({**base, "tools": []})
    assert not exact_request_is_eligible({**base, "tool_choice": "none"})
    assert not exact_request_is_eligible({**base, "n": 2})
    assert not exact_request_is_eligible({**base, "n": True})
    assert not exact_request_is_eligible({**base, "temperature": float("nan")})


def test_exact_cache_stores_only_successful_complete_non_tool_responses():
    complete = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hello"},
            }
        ]
    }

    assert exact_response_is_eligible(200, complete)
    assert not exact_response_is_eligible(500, complete)
    assert not exact_response_is_eligible(
        200,
        {"choices": [{**complete["choices"][0], "finish_reason": "length"}]},
    )
    assert not exact_response_is_eligible(
        200,
        {"choices": [{**complete["choices"][0], "message": {"tool_calls": [{}]}}]},
    )


def test_exact_response_cache_round_trips_only_complete_responses():
    class FakeRedis:
        def __init__(self):
            self.values = {}
            self.expirations = {}

        async def get(self, key):
            return self.values.get(key)

        async def set(self, key, value, ex):
            self.values[key] = value
            self.expirations[key] = ex

    async def run():
        redis = FakeRedis()
        cache = ExactResponseCache(redis)
        request_payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
        }
        response_body = (
            b'{ "choices" : [ { "finish_reason":"stop", "message": '
            b'{"role":"assistant","content":"\\u0068ello"} } ] }\r\n'
        )

        assert await cache.set(
            request_payload,
            200,
            response_body,
            backend_identity="test-backend",
            optimizer_arm="baseline",
            original_input_tokens=5,
            decode_tokens=3,
        )
        cached = await cache.get(
            request_payload,
            backend_identity="test-backend",
            optimizer_arm="baseline",
        )
        optimized_arm_cached = await cache.get(
            request_payload,
            backend_identity="test-backend",
            optimizer_arm="optimized",
        )

        assert cached == CachedResponse(
            status_code=200,
            body=response_body,
            original_input_tokens=5,
            decode_tokens=3,
        )
        assert list(redis.expirations.values()) == [DEFAULT_CACHE_TTL_SECONDS]
        assert optimized_arm_cached is None
        assert not await cache.set(
            request_payload,
            200,
            b'{"choices":[{"finish_reason":"length","message":{"content":"partial"}}]}',
            backend_identity="test-backend",
            optimizer_arm="baseline",
            original_input_tokens=5,
            decode_tokens=3,
        )

    asyncio.run(run())


def test_exact_response_cache_fails_open_on_redis_errors(caplog):
    class UnavailableRedis:
        async def get(self, key):
            raise RedisError("unavailable")

        async def set(self, key, value, ex):
            raise RedisError("unavailable")

    async def run():
        cache = ExactResponseCache(UnavailableRedis())
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
        }
        body = (
            b'{"choices":[{"finish_reason":"stop","message":'
            b'{"role":"assistant","content":"hello"}}]}'
        )

        assert await cache.get(
            payload,
            backend_identity="test-backend",
            optimizer_arm="baseline",
        ) is None
        assert not await cache.set(
            payload,
            200,
            body,
            backend_identity="test-backend",
            optimizer_arm="baseline",
            original_input_tokens=2,
            decode_tokens=2,
        )

    asyncio.run(run())
    assert "Exact cache lookup failed; bypassing cache" in caplog.text
    assert "Exact cache write failed; continuing without cache" in caplog.text


def test_gateway_serves_exact_hit_without_second_backend_request(gateway_harness):
    client, _, backend, _ = gateway_harness
    request_body = {
        "model": "ignored-by-gateway",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }

    first = client.post("/v1/chat/completions", json=request_body)
    second = client.post("/v1/chat/completions", json=request_body)

    assert first.status_code == second.status_code == 200
    assert first.headers["x-greenway-cache-tier"] == "miss"
    assert second.headers["x-greenway-cache-tier"] == "exact"
    assert second.headers["x-greenway-telemetry-source"] == "cache_hit"
    assert second.headers["x-greenway-energy-kwh"] == "null"
    assert second.json()["choices"][0]["message"]["content"] == "cached result"
    assert first.content == second.content
    assert backend.backend_calls == 1


def test_single_flight_coalesces_twenty_identical_cache_misses(gateway_harness):
    client, _, backend, telemetry = gateway_harness
    backend.delay_seconds = 0.05
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "parallel identical request"}],
        "temperature": 0,
    }

    with ThreadPoolExecutor(max_workers=20) as executor:
        responses = list(
            executor.map(
                lambda _: client.post("/v1/chat/completions", json=request_body),
                range(20),
            )
        )

    assert all(response.status_code == 200 for response in responses)
    assert backend.backend_calls == 1
    assert sum(response.headers["x-greenway-cache-tier"] == "miss" for response in responses) == 1
    assert sum(response.headers["x-greenway-cache-tier"] == "exact" for response in responses) == 19
    assert len(telemetry) == 20


def test_bypass_header_skips_lookup_and_store_and_records_bypass(gateway_harness, caplog):
    caplog.set_level("INFO", logger="greenwayai.ingress")
    client, redis, backend, telemetry = gateway_harness
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "bypass"}],
            "temperature": 0,
        },
        headers={"X-Greenway-Cache-Bypass": "true"},
    )

    assert response.status_code == 200
    assert response.headers["x-greenway-cache-tier"] == "bypass"
    assert redis.get_calls == 0
    assert redis.set_calls == 0
    assert backend.backend_calls == 1
    assert telemetry[-1]["cache_tier"] == "bypass"
    assert '"cache_tier": "bypass"' in caplog.text


@pytest.mark.parametrize(
    "overrides",
    [{"stream": True}, {"temperature": 0.2}],
    ids=["streaming", "nonzero-temperature"],
)
def test_streaming_and_stochastic_requests_are_never_cached(gateway_harness, overrides):
    client, redis, backend, _ = gateway_harness
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "not cache eligible"}],
        "temperature": 0,
        **overrides,
    }

    response = client.post("/v1/chat/completions", json=request_body)

    assert response.status_code == 200
    assert response.headers["x-greenway-cache-tier"] == "bypass"
    assert redis.get_calls == 0
    assert redis.set_calls == 0
    assert backend.backend_calls == 1


@pytest.mark.parametrize(
    ("status_code", "finish_reason"),
    [(200, "length"), (503, "stop")],
    ids=["truncated-response", "non-200-response"],
)
def test_truncated_and_non_200_responses_are_never_stored(
    gateway_harness,
    status_code,
    finish_reason,
):
    client, redis, backend, _ = gateway_harness
    body = (
        '{"choices":[{"finish_reason":"'
        + finish_reason
        + '","message":{"role":"assistant","content":"partial"}}]}'
    ).encode()
    backend.responses = [(status_code, body), (status_code, body)]
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "must not store"}],
        "temperature": 0,
    }

    first = client.post("/v1/chat/completions", json=request_body)
    second = client.post("/v1/chat/completions", json=request_body)

    assert first.status_code == second.status_code == status_code
    assert first.headers["x-greenway-cache-tier"] == "miss"
    assert second.headers["x-greenway-cache-tier"] == "miss"
    assert redis.set_calls == 0
    assert backend.backend_calls == 2


def test_redis_down_fails_open_with_unavailable_tier(gateway_harness, caplog):
    client, redis, backend, telemetry = gateway_harness
    redis.fail_get = True

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "still serve me"}],
            "temperature": 0,
        },
    )

    assert response.status_code == 200
    assert response.headers["x-greenway-cache-tier"] == "unavailable"
    assert response.json()["choices"][0]["message"]["content"] == "cached result"
    assert backend.backend_calls == 1
    assert telemetry[-1]["cache_tier"] == "unavailable"
    assert "Exact cache lookup failed; bypassing cache" in caplog.text


def test_cache_tiers_are_recorded_for_miss_exact_and_bypass(gateway_harness, caplog):
    caplog.set_level("INFO", logger="greenwayai.ingress")
    client, _, _, telemetry = gateway_harness
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "telemetry sample"}],
        "temperature": 0,
    }

    miss = client.post("/v1/chat/completions", json=request_body)
    hit = client.post("/v1/chat/completions", json=request_body)
    bypass = client.post(
        "/v1/chat/completions",
        json=request_body,
        headers={"X-Greenway-Cache-Bypass": "1"},
    )

    assert [r.headers["x-greenway-cache-tier"] for r in (miss, hit, bypass)] == [
        "miss",
        "exact",
        "bypass",
    ]
    assert [row["cache_tier"] for row in telemetry[-3:]] == [
        "miss",
        "exact",
        "bypass",
    ]
    assert all(f'"cache_tier": "{tier}"' in caplog.text for tier in ("miss", "exact", "bypass"))


def test_untrusted_tenant_request_header_does_not_change_cache_scope(gateway_harness):
    client, _, backend, _ = gateway_harness
    request_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "header is not identity"}],
        "temperature": 0,
    }
    first = client.post(
        "/v1/chat/completions",
        json=request_body,
        headers={"X-Tenant-ID": "tenant-one"},
    )
    second = client.post(
        "/v1/chat/completions",
        json=request_body,
        headers={"X-Tenant-ID": "tenant-two"},
    )

    assert first.headers["x-greenway-cache-tier"] == "miss"
    assert second.headers["x-greenway-cache-tier"] == "exact"
    assert backend.backend_calls == 1


def test_otel_request_metrics_label_each_cache_tier():
    class Metric:
        def __init__(self):
            self.records = []

        def add(self, value, attributes):
            self.records.append((value, attributes))

        def record(self, value, attributes):
            self.records.append((value, attributes))

    from telemetry.otel_export import record_request_metrics

    counter = Metric()
    metrics = {
        "request_counter": counter,
        "energy_histogram": Metric(),
        "carbon_histogram": Metric(),
    }
    for tier in ("exact", "miss", "bypass", "unavailable"):
        record_request_metrics(
            metrics,
            energy_kwh=None,
            carbon_gco2e=None,
            telemetry_source="cache_hit" if tier == "exact" else "parametric_estimated",
            energy_source_tier="unavailable",
            carbon_source_tier="unavailable",
            cache_tier=tier,
        )

    assert [attributes["cache_tier"] for _, attributes in counter.records] == [
        "exact",
        "miss",
        "bypass",
        "unavailable",
    ]


def test_embedder_normalizes_model_output_to_float_list(monkeypatch):
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_id, revision, device):
            calls["model_id"] = model_id
            calls["revision"] = revision
            calls["device"] = device

        def get_embedding_dimension(self):
            return 3

        def encode(self, text, **kwargs):
            calls["text"] = text
            calls["encode_kwargs"] = kwargs
            return [0.25, -0.5, 0.75]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    embedder = SentenceTransformerEmbedder()
    vector = embedder.encode("example")

    assert calls == {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "device": "cpu",
        "text": "example",
        "encode_kwargs": {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        },
    }
    assert embedder.dimension == 3
    assert vector == [0.25, -0.5, 0.75]


def test_enabled_cache_loads_embedding_model_during_application_startup(monkeypatch):
    loaded = []

    class FakeEmbedder:
        def __init__(self, model_id, revision):
            loaded.append((model_id, revision))
            self.model_id = model_id
            self.dimension = 384

    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setenv("CACHE_EMBEDDING_MODEL_ID", "test-embedding-model")
    monkeypatch.setenv("CACHE_EMBEDDING_MODEL_REVISION", "test-revision")
    monkeypatch.setattr(ingress_app, "SentenceTransformerEmbedder", FakeEmbedder)

    with TestClient(ingress_app.app):
        assert loaded == [("test-embedding-model", "test-revision")]
        assert ingress_app.app.state.semantic_embedder.model_id == "test-embedding-model"


def test_disabled_cache_does_not_load_embedding_model(monkeypatch):
    loaded = []

    class UnexpectedEmbedder:
        def __init__(self, model_id, revision):
            loaded.append(model_id)

    monkeypatch.setenv("CACHE_ENABLED", "false")
    monkeypatch.setattr(ingress_app, "SentenceTransformerEmbedder", UnexpectedEmbedder)

    with TestClient(ingress_app.app):
        assert loaded == []
        assert ingress_app.app.state.semantic_embedder is None
