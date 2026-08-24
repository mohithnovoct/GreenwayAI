import hashlib
import json
import re
import uuid
from functools import lru_cache
from typing import Any

import tiktoken

from shared.schema import GatewayRequest, SLAClass

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 256
DEFAULT_ENCODING = tiktoken.get_encoding("cl100k_base")


def canonicalize(value: str) -> str:
    return " ".join(value.split())


def sanitize(value: str) -> str:
    return "\n".join(re.sub(r"[ \t]+", " ", line) for line in value.strip().split("\n"))


def _sanitize_line(value: str) -> str:
    return re.sub(r"[ \t]+", " ", value.strip())


def _clean_content(value: Any, *, text_value: bool = False) -> Any:
    if isinstance(value, str):
        return sanitize(value) if text_value else value
    if isinstance(value, list):
        return [_clean_content(part) for part in value]
    if isinstance(value, dict):
        # Placeholder recursion until multimodal support is scoped in a later phase.
        return {
            key: _clean_content(part, text_value=key == "text")
            for key, part in value.items()
        }
    return value


def _canonical_content(value: Any, *, text_value: bool = False) -> Any:
    if isinstance(value, str):
        return canonicalize(value) if text_value else value
    if isinstance(value, list):
        return [_canonical_content(part) for part in value]
    if isinstance(value, dict):
        return {
            key: _canonical_content(part, text_value=key == "text")
            for key, part in value.items()
        }
    return value


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["messages"] = [
        {
            **message,
            "role": _sanitize_line(str(message.get("role", ""))).lower(),
            "content": _clean_content(message.get("content", ""), text_value=True),
        }
        for message in normalized.get("messages", [])
    ]
    normalized.setdefault("temperature", DEFAULT_TEMPERATURE)
    normalized.setdefault("top_p", DEFAULT_TOP_P)
    normalized.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    return json.loads(json.dumps(normalized, sort_keys=True, separators=(",", ":")))


def prompt_text(payload: dict[str, Any]) -> str:
    parts = []
    for message in payload.get("messages", []):
        content = message.get("content", "")
        rendered = canonicalize(content) if isinstance(content, str) else json.dumps(
            _canonical_content(content), sort_keys=True, separators=(",", ":")
        )
        parts.append(f"{message.get('role', '')}: {rendered}")
    return "\n".join(parts)


@lru_cache(maxsize=64)
def _encoding_for_model(model: str) -> Any:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return DEFAULT_ENCODING


def count_tokens(text: str, model: str) -> int:
    return len(_encoding_for_model(model).encode(text))


def build_gateway_request(payload: dict[str, Any], sla_header: str | None) -> GatewayRequest:
    normalized = normalize_payload(payload)
    prompt = prompt_text(normalized)
    return GatewayRequest(
        request_id=str(uuid.uuid4()),
        raw_payload=normalized,
        normalized_prompt=prompt,
        content_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        input_tokens=count_tokens(prompt, str(normalized.get("model", "gpt-3.5-turbo"))),
        sla_class=SLAClass.DEFERRABLE if sla_header == "deferrable" else SLAClass.INTERACTIVE,
    )