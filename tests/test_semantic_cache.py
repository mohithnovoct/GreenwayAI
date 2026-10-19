import asyncio
import hashlib
import math
import sys
import struct
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
    SemanticResponseCache,
    SentenceTransformerEmbedder,
    exact_cache_key,
    exact_request_is_eligible,
    exact_response_is_eligible,
    semantic_embedding_text,
    semantic_filter_tags,
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
            self.fail_commands = set()
            self.commands = []
            self.indexes = {}

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

        async def execute_command(self, *args):
            self.commands.append(args)
            command = args[0].upper()
            if command in self.fail_commands:
                raise RedisError("unavailable")
            if command == "FT._LIST":
                return [name.encode() for name in self.indexes]
            if command == "FT.CREATE":
                index_name = args[1].decode() if isinstance(args[1], bytes) else args[1]
                self.indexes[index_name] = args
                return b"OK"
            if command == "FT.INFO":
                index_name = args[1].decode() if isinstance(args[1], bytes) else args[1]
                if index_name not in self.indexes:
                    raise RedisError("Unknown Index name")
                return [b"index_name", index_name.encode()]
            if command == "HSET":
                key = args[1].decode() if isinstance(args[1], bytes) else args[1]
                fields = args[2:]
                self.values[key] = {
                    (field.decode() if isinstance(field, bytes) else field): value
                    for field, value in zip(fields[::2], fields[1::2])
                }
                return 1
            if command == "EXPIRE":
                return 1
            if command == "EVAL":
                key = args[3]
                fields = args[4:-1]
                self.values[key] = {
                    (field.decode() if isinstance(field, bytes) else field): value
                    for field, value in zip(fields[::2], fields[1::2])
                }
                return 1
            if command == "FT.SEARCH":
                query = args[2]
                query = query.decode() if isinstance(query, bytes) else query
                params_pos = args.index("PARAMS")
                vector = args[params_pos + 3]
                query_vector = struct.unpack(f"{len(vector) // 4}f", vector)
                wanted = {}
                for clause in query.split("=>")[0].strip("()").split():
                    field, value = clause[1:].split(":{", 1)
                    wanted[field] = value[:-1]
                matches = []
                for key, entry in self.values.items():
                    if not isinstance(entry, dict) or not all(entry.get(k) == v for k, v in wanted.items()):
                        continue
                    stored_vector = struct.unpack(
                        f"{len(entry['embedding']) // 4}f",
                        entry["embedding"],
                    )
                    dot = sum(a * b for a, b in zip(query_vector, stored_vector))
                    norm_a = math.sqrt(sum(a * a for a in query_vector))
                    norm_b = math.sqrt(sum(b * b for b in stored_vector))
                    distance = 1 - dot / (norm_a * norm_b)
                    matches.append((distance, key, entry))
                self.last_search = {"query": query, "wanted": wanted, "matches": len(matches)}
                if not matches:
                    return [0]
                distance, key, entry = min(matches, key=lambda match: match[0])
                result_fields = [
                    b"body", entry["body"],
                    b"status_code", str(entry["status_code"]).encode(),
                    b"original_input_tokens", str(entry["original_input_tokens"]).encode(),
                    b"decode_tokens", str(entry["decode_tokens"]).encode(),
                    b"distance", str(distance).encode(),
                ]
                return [1, key.encode(), result_fields]
            raise AssertionError(f"Unexpected Redis command: {args!r}")

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
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
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


@pytest.fixture
def semantic_gateway_harness(gateway_harness):
    client, redis, backend, telemetry = gateway_harness

    class MockEmbedder:
        model_id = "mock-embedding"
        revision = "mock-revision"
        dimension = 3

        def __init__(self):
            self.vectors = {}
            self.delay_seconds = 0
            self.fail = False

        def encode(self, text):
            import time

            if self.fail:
                raise RuntimeError("mock embed failure")
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            return self.vectors.get(text, [1.0, 0.0, 0.0])

    embedder = MockEmbedder()
    executor = ThreadPoolExecutor(max_workers=2)
    cache = SemanticResponseCache(
        redis,
        tenant_namespace="gateway-test",
        model_revision=embedder.revision,
        dimension=embedder.dimension,
        similarity_threshold=0.8,
    )
    asyncio.run(cache.create_index())
    ingress_app.app.state.semantic_cache = cache
    ingress_app.app.state.semantic_embedder = embedder
    ingress_app.app.state.embedding_executor = executor
    ingress_app.app.state.semantic_embedding_timeout = 0.1
    ingress_app.app.state.semantic_cache_enabled = True
    yield client, redis, backend, telemetry, cache, embedder
    executor.shutdown(wait=True)


def _semantic_payload(text="original request"):
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 128,
        "top_p": 1,
        "seed": 5,
    }


def _seed_semantic_response(cache, payload, *, backend="https://provider.example/chat", arm="baseline", tenant=""):
    return asyncio.run(
        cache.set(
            payload,
            200,
            (
                b'{"choices":[{"finish_reason":"stop","message":{"role":"assistant",'
                b'"content":"semantic result"}}],"usage":{"prompt_tokens":7,'
                b'"completion_tokens":3,"total_tokens":10}}'
            ),
            backend_identity=backend,
            optimizer_arm=arm,
            tenant_id=tenant,
            original_input_tokens=7,
            decode_tokens=3,
            embedding=[1.0, 0.0, 0.0],
        )
    )


def _lookup_semantic_response(
    cache,
    payload,
    *,
    backend="https://provider.example/chat",
    arm="baseline",
    tenant="",
    embedding=None,
):
    return asyncio.run(
        cache.lookup(
            payload,
            backend_identity=backend,
            optimizer_arm=arm,
            tenant_id=tenant,
            embedding=embedding or [1.0, 0.0, 0.0],
        )
    )


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
    assert not exact_response_is_eligible(201, complete)
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


def test_semantic_cache_hit_for_paraphrase_with_mock_embeddings(
    semantic_gateway_harness,
    caplog,
):
    caplog.set_level("INFO", logger="greenwayai.ingress")
    client, redis, backend, telemetry, _, embedder = semantic_gateway_harness
    embedder.vectors["Can you explain this request?"] = [1.0, 0.0, 0.0]
    embedder.vectors["Please clarify this request for me."] = [0.99, 0.01, 0.0]
    first_payload = _semantic_payload("Can you explain this request?")
    second_payload = _semantic_payload("Please clarify this request for me.")

    first = client.post("/v1/chat/completions", json=first_payload)
    entries = [value for value in redis.values.values() if isinstance(value, dict)]
    assert len(entries) == 1
    effective_second_payload = {**second_payload, "model": "gemini-2.5-flash"}
    assert {
        field: entries[0][field]
        for field in SemanticResponseCache._TAG_FIELDS
    } == semantic_filter_tags(
        effective_second_payload,
        "gateway-test",
        "https://provider.example/v1/chat/completions",
        "baseline",
    )
    direct_lookup = _lookup_semantic_response(
        ingress_app.app.state.semantic_cache,
        effective_second_payload,
        backend="https://provider.example/v1/chat/completions",
        embedding=[0.99, 0.01, 0.0],
    )
    assert direct_lookup.response is not None, redis.last_search
    second = client.post("/v1/chat/completions", json=second_payload)

    assert first.status_code == second.status_code == 200
    assert first.headers["x-greenway-cache-tier"] == "miss"
    assert second.headers["x-greenway-cache-tier"] == "semantic"
    assert second.content == first.content
    assert second.headers["x-greenway-input-tokens"] == "2"
    assert second.headers["x-greenway-output-tokens"] == "2"
    assert backend.backend_calls == 1
    assert telemetry[-2]["cache_tier"] == "miss"
    assert telemetry[-1]["cache_tier"] == "semantic"
    semantic_rows = [
        record.message
        for record in caplog.records
        if '"cache_tier": "semantic"' in record.message
    ]
    assert len(semantic_rows) == 1
    assert '"semantic_similarity": 0.999949' in semantic_rows[0]


def test_semantic_filter_requires_model_match(semantic_gateway_harness):
    _, _, _, _, cache, _ = semantic_gateway_harness
    payload = _semantic_payload()
    assert _seed_semantic_response(cache, payload)
    changed = {**payload, "model": "different-model"}
    assert _lookup_semantic_response(cache, changed).response is None


def test_semantic_filter_requires_backend_endpoint_match(semantic_gateway_harness):
    _, _, _, _, cache, _ = semantic_gateway_harness
    payload = _semantic_payload()
    assert _seed_semantic_response(cache, payload)
    result = _lookup_semantic_response(
        cache,
        payload,
        backend="https://other-provider.example/chat",
    )
    assert result.response is None


def test_semantic_filter_requires_optimizer_arm_match(semantic_gateway_harness):
    _, _, _, _, cache, _ = semantic_gateway_harness
    payload = _semantic_payload()
    assert _seed_semantic_response(cache, payload)
    result = _lookup_semantic_response(cache, payload, arm="optimized")
    assert result.response is None


def test_semantic_filter_requires_tenant_scope_match(semantic_gateway_harness):
    _, _, _, _, cache, _ = semantic_gateway_harness
    payload = _semantic_payload()
    assert _seed_semantic_response(cache, payload, tenant="tenant-one")
    result = _lookup_semantic_response(cache, payload, tenant="tenant-two")
    assert result.response is None


def test_semantic_filter_requires_system_prompt_match(semantic_gateway_harness):
    _, _, _, _, cache, _ = semantic_gateway_harness
    payload = {
        **_semantic_payload(),
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "original request"},
        ],
    }
    assert _seed_semantic_response(cache, payload)
    changed = {
        **payload,
        "messages": [
            {"role": "system", "content": "Be verbose."},
            payload["messages"][-1],
        ],
    }
    assert _lookup_semantic_response(cache, changed).response is None


def test_semantic_filter_requires_prior_conversation_turns_match(semantic_gateway_harness):
    _, _, _, _, cache, _ = semantic_gateway_harness
    payload = {
        **_semantic_payload(),
        "messages": [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "original request"},
        ],
    }
    assert _seed_semantic_response(cache, payload)
    changed = {
        **payload,
        "messages": [
            payload["messages"][0],
            {"role": "assistant", "content": "different prior answer"},
            payload["messages"][-1],
        ],
    }
    assert _lookup_semantic_response(cache, changed).response is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tokens", 64),
        ("stop", ["STOP"]),
        ("top_p", 0.5),
        ("seed", 6),
        ("response_format", {"type": "json_object"}),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("logit_bias", {"42": 1}),
    ],
    ids=[
        "max_tokens",
        "stop",
        "top_p",
        "seed",
        "response_format",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
    ],
)
def test_semantic_filter_requires_output_parameters_match(
    semantic_gateway_harness,
    field,
    value,
):
    _, _, _, _, cache, _ = semantic_gateway_harness
    payload = _semantic_payload()
    assert _seed_semantic_response(cache, payload)
    changed = {**payload, field: value}
    assert _lookup_semantic_response(cache, changed).response is None


def test_semantic_embedding_uses_only_the_final_user_turn():
    payload = {
        **_semantic_payload("final question"),
        "messages": [
            {"role": "system", "content": "system context"},
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
            {"role": "user", "content": "final question"},
        ],
    }

    assert semantic_embedding_text(payload) == "final question"


def test_semantic_timeout_fails_open_and_tags_unavailable(
    semantic_gateway_harness,
):
    client, _, backend, telemetry, _, embedder = semantic_gateway_harness
    embedder.delay_seconds = 0.2
    ingress_app.app.state.semantic_embedding_timeout = 0.01
    response = client.post(
        "/v1/chat/completions",
        json=_semantic_payload("slow embedding request"),
    )

    assert response.status_code == 200
    assert response.headers["x-greenway-cache-tier"] == "unavailable"
    assert backend.backend_calls == 1
    assert telemetry[-1]["cache_tier"] == "unavailable"


def test_semantic_embedding_error_fails_open_and_tags_unavailable(
    semantic_gateway_harness,
):
    client, _, backend, telemetry, _, embedder = semantic_gateway_harness
    embedder.fail = True
    response = client.post(
        "/v1/chat/completions",
        json=_semantic_payload("embedding error request"),
    )

    assert response.status_code == 200
    assert response.headers["x-greenway-cache-tier"] == "unavailable"
    assert backend.backend_calls == 1
    assert telemetry[-1]["cache_tier"] == "unavailable"


def test_semantic_redis_error_fails_open_and_tags_unavailable(
    semantic_gateway_harness,
):
    client, redis, backend, telemetry, _, _ = semantic_gateway_harness
    redis.fail_commands.add("FT.SEARCH")
    response = client.post(
        "/v1/chat/completions",
        json=_semantic_payload("redis error request"),
    )

    assert response.status_code == 200
    assert response.headers["x-greenway-cache-tier"] == "unavailable"
    assert backend.backend_calls == 1
    assert telemetry[-1]["cache_tier"] == "unavailable"


def test_semantic_cache_model_revision_creates_a_new_index(
    semantic_gateway_harness,
):
    _, redis, _, _, _, _ = semantic_gateway_harness
    cache = SemanticResponseCache(
        redis,
        tenant_namespace="gateway-test",
        model_revision="next-revision",
        dimension=3,
        similarity_threshold=0.8,
    )
    assert asyncio.run(cache.create_index())
    assert asyncio.run(cache.create_index())
    created = [
        command[1]
        for command in redis.commands
        if command[0] == "FT.CREATE"
    ]
    assert len(created) == 2
    assert created[0] != created[1]
    assert cache.index_name.endswith(hashlib.sha256(b"next-revision").hexdigest()[:16])


def test_semantic_cache_disabled_never_loads_or_queries_index(gateway_harness):
    _, redis, _, _ = gateway_harness
    assert ingress_app.app.state.semantic_cache is None
    assert ingress_app.app.state.semantic_embedder is None
    assert not any(command[0].startswith("FT.") for command in redis.commands)


def test_twenty_concurrent_misses_coalesce_with_semantic_cache_enabled(
    semantic_gateway_harness,
):
    client, _, backend, _, _, _ = semantic_gateway_harness
    backend.delay_seconds = 0.05
    request_body = _semantic_payload("concurrent semantic-enabled request")

    with ThreadPoolExecutor(max_workers=20) as executor:
        responses = list(
            executor.map(
                lambda _: client.post("/v1/chat/completions", json=request_body),
                range(20),
            )
        )

    assert all(response.status_code == 200 for response in responses)
    assert backend.backend_calls == 1
    assert sum(r.headers["x-greenway-cache-tier"] == "miss" for r in responses) == 1
    assert sum(r.headers["x-greenway-cache-tier"] == "exact" for r in responses) == 19


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
    thread_calls = []

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
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda value: thread_calls.append(("intraop", value)),
            set_num_interop_threads=lambda value: thread_calls.append(("interop", value)),
        ),
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
    assert thread_calls == [("intraop", 2), ("interop", 2)]


def test_enabled_cache_loads_embedding_model_during_application_startup(monkeypatch):
    loaded = []

    class StartupRedis:
        indexes = set()

        async def execute_command(self, *args):
            if args[0] == "FT._LIST":
                return [name.encode() for name in self.indexes]
            if args[0] == "FT.CREATE":
                self.indexes.add(args[1])
                return b"OK"
            if args[0] == "FT.INFO":
                return [b"index_name", args[1].encode()]
            raise AssertionError(args)

        async def aclose(self):
            return None

    class FakeEmbedder:
        def __init__(self, model_id, revision):
            loaded.append((model_id, revision))
            self.model_id = model_id
            self.revision = revision
            self.dimension = 384

    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "true")
    monkeypatch.setenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.99")
    monkeypatch.setenv("CACHE_EMBEDDING_MODEL_ID", "test-embedding-model")
    monkeypatch.setenv("CACHE_EMBEDDING_MODEL_REVISION", "test-revision")
    monkeypatch.setattr(ingress_app, "SentenceTransformerEmbedder", FakeEmbedder)
    monkeypatch.setattr(
        ingress_app.Redis,
        "from_url",
        classmethod(lambda cls, *args, **kwargs: StartupRedis()),
    )

    with TestClient(ingress_app.app):
        assert loaded == [("test-embedding-model", "test-revision")]
        assert ingress_app.app.state.semantic_embedder.model_id == "test-embedding-model"
        assert ingress_app.app.state.embedding_executor._max_workers == 2
        assert ingress_app.app.state.semantic_cache.index_name.endswith(
            hashlib.sha256(b"test-revision").hexdigest()[:16]
        )


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
