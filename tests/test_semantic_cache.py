import asyncio
import sys
from types import SimpleNamespace

import httpx
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
)


def test_exact_cache_key_covers_request_and_single_tenant_namespace():
    payload = {
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
        "max_tokens": 80,
    }

    key = exact_cache_key(payload, DEFAULT_CACHE_TENANT_NAMESPACE, "test-backend")
    reordered_payload = {
        "max_tokens": 80,
        "temperature": 0,
        "messages": [{"content": "hello", "role": "user"}],
        "model": "gemini-2.5-flash",
    }

    assert key == exact_cache_key(
        reordered_payload,
        DEFAULT_CACHE_TENANT_NAMESPACE,
        "test-backend",
    )
    assert key != exact_cache_key(
        {**payload, "max_tokens": 81},
        DEFAULT_CACHE_TENANT_NAMESPACE,
        "test-backend",
    )
    assert key != exact_cache_key(
        payload,
        "another-deployment",
        "test-backend",
    )
    assert key != exact_cache_key(
        payload,
        DEFAULT_CACHE_TENANT_NAMESPACE,
        "another-backend",
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
            b'{"choices":[{"finish_reason":"stop","message":'
            b'{"role":"assistant","content":"hello"}}]}'
        )

        assert await cache.set(
            request_payload,
            200,
            response_body,
            backend_identity="test-backend",
        )
        cached = await cache.get(request_payload, backend_identity="test-backend")

        assert cached == CachedResponse(status_code=200, body=response_body)
        assert list(redis.expirations.values()) == [DEFAULT_CACHE_TTL_SECONDS]
        assert not await cache.set(
            request_payload,
            200,
            b'{"choices":[{"finish_reason":"length","message":{"content":"partial"}}]}',
            backend_identity="test-backend",
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

        assert await cache.get(payload, backend_identity="test-backend") is None
        assert not await cache.set(
            payload,
            200,
            body,
            backend_identity="test-backend",
        )

    asyncio.run(run())
    assert "Exact cache lookup failed; bypassing cache" in caplog.text
    assert "Exact cache write failed; continuing without cache" in caplog.text


def test_gateway_serves_exact_hit_without_second_backend_request(monkeypatch):
    backend_calls = []

    class FakeRedis:
        def __init__(self):
            self.values = {}

        async def get(self, key):
            return self.values.get(key)

        async def set(self, key, value, ex):
            self.values[key] = value

        async def aclose(self):
            return None

    fake_redis = FakeRedis()

    class FakeEmbedder:
        def __init__(self, model_id, revision):
            self.model_id = model_id
            self.dimension = 384

    def backend_handler(_request):
        backend_calls.append(True)
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

    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setenv("CACHE_TENANT_NAMESPACE", "gateway-test")
    monkeypatch.setenv("GREENWAY_CARBON_DISABLED", "true")
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

    with TestClient(ingress_app.app) as client:
        ingress_app.app.state.http_client._transport = httpx.MockTransport(backend_handler)
        request_body = {
            "model": "ignored-by-gateway",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
        }

        first = client.post("/v1/chat/completions", json=request_body)
        second = client.post("/v1/chat/completions", json=request_body)

    assert first.status_code == second.status_code == 200
    assert first.headers["x-greenway-cache"] == "miss"
    assert second.headers["x-greenway-cache"] == "exact_hit"
    assert second.headers["x-greenway-telemetry-source"] == "cache_hit"
    assert second.headers["x-greenway-energy-kwh"] == "null"
    assert second.json()["choices"][0]["message"]["content"] == "cached result"
    assert len(backend_calls) == 1


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
