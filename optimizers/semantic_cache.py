import hashlib
import json
import logging
import math
import asyncio
import struct
import uuid
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError


logger = logging.getLogger("greenwayai.semantic_cache")
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_CACHE_TTL_SECONDS = 3600
DEFAULT_CACHE_TENANT_NAMESPACE = "greenwayai-single-tenant"
DEFAULT_SEMANTIC_CACHE_TIMEOUT_MS = 100


def exact_cache_key(
    payload: dict[str, Any],
    tenant_namespace: str,
    backend_identity: str,
    optimizer_arm: str = "baseline",
    tenant_id: str = "",
) -> str:
    namespace = tenant_namespace.strip()
    if not namespace:
        raise ValueError("Cache tenant namespace must not be empty")
    canonical_payload = json.dumps(
        {
            "backend_identity": backend_identity,
            "optimizer_arm": optimizer_arm,
            "request": payload,
            "tenant_id": tenant_id,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    namespace_hash = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    request_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    return f"greenwayai:cache:v1:{namespace_hash}:exact:{request_hash}"


def exact_request_is_eligible(payload: dict[str, Any]) -> bool:
    if payload.get("stream", False):
        return False
    temperature = payload.get("temperature", 1.0)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature != 0
    ):
        return False
    choices = payload.get("n", 1)
    if isinstance(choices, bool) or not isinstance(choices, int) or choices != 1:
        return False
    if any(field in payload for field in ("tools", "tool_choice", "functions", "function_call")):
        return False
    return True


def exact_response_token_counts(
    response_payload: dict[str, Any],
    fallback_input_tokens: int,
    fallback_decode_tokens: int,
) -> tuple[int, int]:
    usage = response_payload.get("usage")
    if not isinstance(usage, dict):
        return fallback_input_tokens, fallback_decode_tokens
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens <= 0
        or isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens <= 0
    ):
        return fallback_input_tokens, fallback_decode_tokens
    if isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or total_tokens < 0:
        return prompt_tokens, completion_tokens
    return prompt_tokens, max(completion_tokens, total_tokens - prompt_tokens)


def exact_response_is_eligible(status_code: int, payload: dict[str, Any]) -> bool:
    if status_code != 200:
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return False
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        return False
    message = choice.get("message")
    if not isinstance(message, dict):
        return False
    if message.get("tool_calls") or message.get("function_call") or message.get("refusal"):
        return False
    return True


@dataclass(frozen=True)
class CachedResponse:
    status_code: int
    body: bytes
    original_input_tokens: int
    decode_tokens: int


@dataclass(frozen=True)
class CacheLookup:
    response: CachedResponse | None
    available: bool


@dataclass(frozen=True)
class SemanticLookup:
    response: CachedResponse | None
    available: bool
    similarity: float | None = None


@dataclass(frozen=True)
class SingleFlight:
    key: str
    future: asyncio.Future[CachedResponse | None]


class ExactResponseCache:
    def __init__(
        self,
        redis: Redis,
        tenant_namespace: str = DEFAULT_CACHE_TENANT_NAMESPACE,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Cache TTL must be a positive number of seconds")
        exact_cache_key({}, tenant_namespace, "")
        self._redis = redis
        self._tenant_namespace = tenant_namespace
        self._ttl_seconds = ttl_seconds
        self._index_available = True
        self._inflight: dict[str, asyncio.Future[CachedResponse | None]] = {}
        self._inflight_guard = asyncio.Lock()

    async def lookup(
        self,
        request_payload: dict[str, Any],
        backend_identity: str,
        optimizer_arm: str,
        tenant_id: str = "",
    ) -> CacheLookup:
        if not exact_request_is_eligible(request_payload):
            return CacheLookup(response=None, available=True)
        key = exact_cache_key(
            request_payload,
            self._tenant_namespace,
            backend_identity,
            optimizer_arm,
            tenant_id,
        )
        try:
            stored = await self._redis.get(key)
        except RedisError as error:
            logger.warning("Exact cache lookup failed; bypassing cache: %s", error)
            return CacheLookup(response=None, available=False)
        if stored is None:
            return CacheLookup(response=None, available=True)
        try:
            entry = json.loads(stored)
            status_code = entry["status_code"]
            body = entry["body"]
            original_input_tokens = entry["original_input_tokens"]
            decode_tokens = entry["decode_tokens"]
            response_payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Ignoring malformed exact cache entry")
            return CacheLookup(response=None, available=True)
        if (
            not isinstance(entry, dict)
            or isinstance(status_code, bool)
            or not isinstance(status_code, int)
            or not isinstance(body, str)
            or isinstance(original_input_tokens, bool)
            or not isinstance(original_input_tokens, int)
            or isinstance(decode_tokens, bool)
            or not isinstance(decode_tokens, int)
            or not isinstance(response_payload, dict)
            or not exact_response_is_eligible(status_code, response_payload)
        ):
            logger.warning("Ignoring ineligible exact cache entry")
            return CacheLookup(response=None, available=True)
        return CacheLookup(
            response=CachedResponse(
                status_code=status_code,
                body=body.encode("utf-8"),
                original_input_tokens=original_input_tokens,
                decode_tokens=decode_tokens,
            ),
            available=True,
        )

    async def get(
        self,
        request_payload: dict[str, Any],
        backend_identity: str,
        optimizer_arm: str = "baseline",
        tenant_id: str = "",
    ) -> CachedResponse | None:
        return (
            await self.lookup(
                request_payload,
                backend_identity,
                optimizer_arm,
                tenant_id,
            )
        ).response

    async def begin_singleflight(
        self,
        request_payload: dict[str, Any],
        backend_identity: str,
        optimizer_arm: str,
        tenant_id: str = "",
    ) -> tuple[SingleFlight | None, CachedResponse | None]:
        key = exact_cache_key(
            request_payload,
            self._tenant_namespace,
            backend_identity,
            optimizer_arm,
            tenant_id,
        )
        while True:
            async with self._inflight_guard:
                future = self._inflight.get(key)
                if future is None:
                    future = asyncio.get_running_loop().create_future()
                    self._inflight[key] = future
                    return SingleFlight(key=key, future=future), None
            response = await asyncio.shield(future)
            if response is not None:
                return None, response

    async def finish_singleflight(
        self,
        flight: SingleFlight,
        response: CachedResponse | None,
    ) -> None:
        async with self._inflight_guard:
            if self._inflight.get(flight.key) is flight.future:
                del self._inflight[flight.key]
            if not flight.future.done():
                flight.future.set_result(response)

    async def set(
        self,
        request_payload: dict[str, Any],
        status_code: int,
        response_body: bytes,
        backend_identity: str,
        optimizer_arm: str,
        original_input_tokens: int,
        decode_tokens: int,
        tenant_id: str = "",
    ) -> bool:
        if not exact_request_is_eligible(request_payload):
            return False
        try:
            body = response_body.decode("utf-8")
            response_payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(response_payload, dict) or not exact_response_is_eligible(status_code, response_payload):
            return False
        entry = json.dumps(
            {
                "status_code": status_code,
                "body": body,
                "original_input_tokens": original_input_tokens,
                "decode_tokens": decode_tokens,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        key = exact_cache_key(
            request_payload,
            self._tenant_namespace,
            backend_identity,
            optimizer_arm,
            tenant_id,
        )
        try:
            await self._redis.set(key, entry, ex=self._ttl_seconds)
        except RedisError as error:
            logger.warning("Exact cache write failed; continuing without cache: %s", error)
            return False
        return True


def _hash_tag(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semantic_embedding_text(payload: dict[str, Any]) -> str | None:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return None
    final_message = messages[-1]
    text = final_message.get("content")
    if final_message.get("role") != "user" or not isinstance(text, str) or not text.strip():
        return None
    return text


def semantic_filter_tags(
    payload: dict[str, Any],
    tenant_namespace: str,
    backend_identity: str,
    optimizer_arm: str,
    tenant_id: str = "",
) -> dict[str, str]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        raise ValueError("Semantic cache requires a final user message")
    remaining_messages = [*messages[:-1], {**messages[-1], "content": ""}]
    parameters = {key: value for key, value in payload.items() if key != "messages"}
    return {
        "model_tag": _hash_tag(payload.get("model")),
        "backend_tag": _hash_tag(backend_identity),
        "optimizer_tag": _hash_tag(optimizer_arm),
        "tenant_tag": _hash_tag({"namespace": tenant_namespace, "tenant_id": tenant_id}),
        "conversation_tag": _hash_tag(remaining_messages),
        "parameters_tag": _hash_tag(parameters),
    }


class SemanticResponseCache:
    _TAG_FIELDS = (
        "model_tag",
        "backend_tag",
        "optimizer_tag",
        "tenant_tag",
        "conversation_tag",
        "parameters_tag",
    )

    def __init__(
        self,
        redis: Redis,
        *,
        tenant_namespace: str,
        model_revision: str,
        dimension: int,
        similarity_threshold: float,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if ttl_seconds <= 0:
            raise ValueError("Cache TTL must be a positive number of seconds")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("Semantic similarity threshold must be between 0 and 1")
        if not tenant_namespace.strip():
            raise ValueError("Cache tenant namespace must not be empty")
        revision_hash = hashlib.sha256(model_revision.encode("utf-8")).hexdigest()[:16]
        self._redis = redis
        self._tenant_namespace = tenant_namespace
        self._ttl_seconds = ttl_seconds
        self._dimension = dimension
        self._similarity_threshold = similarity_threshold
        self.index_name = f"greenwayai_semantic_{revision_hash}"
        self.key_prefix = f"greenwayai:semantic:{revision_hash}:"
        self._index_available = False

    async def create_index(self) -> bool:
        try:
            existing_indexes = await self._redis.execute_command("FT._LIST")
            if self.index_name not in {
                value.decode("utf-8") if isinstance(value, bytes) else value
                for value in existing_indexes
            }:
                schema: list[Any] = ["SCHEMA"]
                for field in self._TAG_FIELDS:
                    schema.extend((field, "TAG"))
                schema.extend(
                    (
                        "embedding",
                        "VECTOR",
                        "HNSW",
                        "6",
                        "TYPE",
                        "FLOAT32",
                        "DIM",
                        str(self._dimension),
                        "DISTANCE_METRIC",
                        "COSINE",
                    )
                )
                try:
                    await self._redis.execute_command(
                        "FT.CREATE",
                        self.index_name,
                        "ON",
                        "HASH",
                        "PREFIX",
                        "1",
                        self.key_prefix,
                        *schema,
                    )
                except RedisError:
                    refreshed = await self._redis.execute_command("FT._LIST")
                    names = {
                        value.decode("utf-8") if isinstance(value, bytes) else value
                        for value in refreshed
                    }
                    if self.index_name not in names:
                        raise
            await self._redis.execute_command("FT.INFO", self.index_name)
        except RedisError as error:
            self._index_available = False
            logger.warning("Semantic cache index initialization failed: %s", error)
            return False
        self._index_available = True
        return True

    async def lookup(
        self,
        payload: dict[str, Any],
        backend_identity: str,
        optimizer_arm: str,
        tenant_id: str,
        embedding: list[float],
    ) -> SemanticLookup:
        if not exact_request_is_eligible(payload):
            return SemanticLookup(response=None, available=True)
        if not self._index_available:
            return SemanticLookup(response=None, available=False)
        if len(embedding) != self._dimension or not all(math.isfinite(value) for value in embedding):
            logger.warning("Semantic embedding has invalid dimensions or values")
            return SemanticLookup(response=None, available=False)
        tags = semantic_filter_tags(
            payload,
            self._tenant_namespace,
            backend_identity,
            optimizer_arm,
            tenant_id,
        )
        tag_query = " ".join(
            f"@{field}:{{{tags[field]}}}" for field in self._TAG_FIELDS
        )
        query = f"({tag_query})=>[KNN 1 @embedding $vector AS distance]"
        vector = struct.pack(f"{len(embedding)}f", *embedding)
        try:
            result = await self._redis.execute_command(
                "FT.SEARCH",
                self.index_name,
                query,
                "PARAMS",
                "2",
                "vector",
                vector,
                "RETURN",
                "5",
                "body",
                "status_code",
                "original_input_tokens",
                "decode_tokens",
                "distance",
                "DIALECT",
                "2",
                "LIMIT",
                "0",
                "1",
            )
        except RedisError as error:
            logger.warning("Semantic cache lookup failed; bypassing cache: %s", error)
            return SemanticLookup(response=None, available=False)
        if not isinstance(result, (list, tuple)) or not result or int(result[0]) == 0:
            return SemanticLookup(response=None, available=True)
        try:
            fields = result[2]
            record = {
                (key.decode("utf-8") if isinstance(key, bytes) else key):
                (value.decode("utf-8") if isinstance(value, bytes) else value)
                for key, value in zip(fields[::2], fields[1::2])
            }
            distance = float(record["distance"])
            # RediSearch cosine distance is 1 - cosine similarity.
            similarity = 1.0 - distance
            body = record["body"]
            status_code = int(record["status_code"])
            original_input_tokens = int(record["original_input_tokens"])
            decode_tokens = int(record["decode_tokens"])
            response_payload = json.loads(body)
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring malformed semantic cache entry")
            return SemanticLookup(response=None, available=True)
        if (
            similarity < self._similarity_threshold
            or not isinstance(response_payload, dict)
            or not exact_response_is_eligible(status_code, response_payload)
        ):
            return SemanticLookup(response=None, available=True, similarity=similarity)
        return SemanticLookup(
            response=CachedResponse(
                status_code=status_code,
                body=body.encode("utf-8"),
                original_input_tokens=original_input_tokens,
                decode_tokens=decode_tokens,
            ),
            available=True,
            similarity=similarity,
        )

    async def set(
        self,
        payload: dict[str, Any],
        status_code: int,
        response_body: bytes,
        backend_identity: str,
        optimizer_arm: str,
        tenant_id: str,
        original_input_tokens: int,
        decode_tokens: int,
        embedding: list[float],
    ) -> bool:
        if not self._index_available or not exact_request_is_eligible(payload):
            return False
        if len(embedding) != self._dimension or not all(math.isfinite(value) for value in embedding):
            logger.warning("Semantic embedding has invalid dimensions or values")
            return False
        try:
            body = response_body.decode("utf-8")
            response_payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(response_payload, dict) or not exact_response_is_eligible(status_code, response_payload):
            return False
        tags = semantic_filter_tags(
            payload,
            self._tenant_namespace,
            backend_identity,
            optimizer_arm,
            tenant_id,
        )
        record: dict[str, Any] = {
            **tags,
            "body": body,
            "status_code": status_code,
            "original_input_tokens": original_input_tokens,
            "decode_tokens": decode_tokens,
            "embedding": struct.pack(f"{len(embedding)}f", *embedding),
        }
        key = f"{self.key_prefix}{uuid.uuid4().hex}"
        fields: list[Any] = []
        for field, value in record.items():
            fields.extend((field, value))
        try:
            await self._redis.execute_command(
                "EVAL",
                "redis.call('HSET', KEYS[1], unpack(ARGV, 1, #ARGV - 1)); "
                "redis.call('EXPIRE', KEYS[1], ARGV[#ARGV]); return 1",
                "1",
                key,
                *fields,
                str(self._ttl_seconds),
            )
        except RedisError as error:
            logger.warning("Semantic cache write failed; continuing without cache: %s", error)
            return False
        return True


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        revision: str = DEFAULT_MODEL_REVISION,
        device: str = "cpu",
    ) -> None:
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        from sentence_transformers import SentenceTransformer

        logger.info(
            "semantic embedding model loading model_id=%s revision=%s device=%s",
            model_id,
            revision,
            device,
        )
        self._model: Any = SentenceTransformer(
            model_id,
            revision=revision,
            device=device,
        )
        self.model_id = model_id
        self.revision = revision
        self.dimension = self._model.get_embedding_dimension()
        if not self.dimension:
            raise RuntimeError(f"Embedding model {model_id!r} did not report its dimension")
        logger.info(
            "semantic embedding model loaded model_id=%s revision=%s dimensions=%s device=%s",
            model_id,
            revision,
            self.dimension,
            device,
        )

    def encode(self, text: str) -> list[float]:
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [float(value) for value in embedding]
