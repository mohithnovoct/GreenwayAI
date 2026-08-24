import hashlib
import json
import uuid
from typing import Any

import tiktoken

from shared.schema import GatewayRequest, SLAClass

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 256


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _clean_content(value: Any) -> Any:
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        return [_clean_content(part) for part in value]
    return value


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["messages"] = [
        {
            **message,
            "role": _clean_text(str(message.get("role", ""))).lower(),
            "content": _clean_content(message.get("content", "")),
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
        rendered = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
        parts.append(f"{message.get('role', '')}: {rendered}")
    return "\n".join(parts)


def count_tokens(text: str, model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def build_gateway_request(payload: dict[str, Any], sla_header: str | None) -> GatewayRequest:
    normalized = normalize_payload(payload)
    prompt = prompt_text(normalized)
    return GatewayRequest(
        request_id=uuid.uuid4().hex,
        raw_payload=normalized,
        normalized_prompt=prompt,
        content_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        input_tokens=count_tokens(prompt, str(normalized.get("model", "gpt-3.5-turbo"))),
        sla_class=SLAClass.DEFERRABLE if sla_header == "deferrable" else SLAClass.INTERACTIVE,
    )