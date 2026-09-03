from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any

from ingress.normalization import canonicalize, count_tokens, prompt_text
from shared.schema import GatewayRequest

DEFAULT_JARGON = {
    "MOER": "marginal operating emissions rate",
    "gCO2e": "grams of carbon dioxide equivalent",
    "p50": "50th percentile",
    "p95": "95th percentile",
    "vLLM": "vLLM language model serving engine",
}

_ROLE_PATTERN = re.compile(r"\b(as an?|you are|act as|role|persona|expert in)\b", re.IGNORECASE)
_LENGTH_PATTERN = re.compile(
    r"\b\d+\s*(?:words?|sentences?|bullets?|items?|points?|characters?)\b|"
    r"\b(?:brief|concise|short|detailed|length|limit|under|maximum|max|min(?:imum)?)\b",
    re.IGNORECASE,
)
_VARIATION_PATTERN = re.compile(
    r"\b(?:variations?|alternatives?|options?|versions?|approaches?|examples?)\b",
    re.IGNORECASE,
)
_ITERATION_PATTERN = re.compile(r"\b(?:again|iterat(?:e|ion)|one at a time|separately|re-?prompt)\b", re.IGNORECASE)


@dataclass(frozen=True)
class OptimizationResult:
    request: GatewayRequest
    original_prompt: str
    optimized_prompt: str
    original_tokens: int
    optimized_tokens: int
    optimized: bool
    changes: tuple[str, ...]


def optimizer_enabled() -> bool:
    return os.getenv("PROMPT_OPT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def should_optimize(*, percentage: float | None = None, rng: random.Random | None = None) -> bool:
    if not optimizer_enabled():
        return False
    configured = os.getenv("PROMPT_OPT_AB_PERCENT", "100") if percentage is None else percentage
    try:
        percentage_value = max(0.0, min(100.0, float(configured)))
    except ValueError:
        percentage_value = 100.0
    if percentage_value <= 0:
        return False
    if percentage_value >= 100:
        return True
    return (rng or random).random() * 100 < percentage_value


def _jargon_map() -> dict[str, str]:
    mapping = dict(DEFAULT_JARGON)
    configured = os.getenv("PROMPT_OPT_JARGON_MAP", "").strip()
    if configured:
        try:
            values = json.loads(configured)
            if isinstance(values, dict):
                mapping.update({str(key): str(value) for key, value in values.items()})
        except json.JSONDecodeError:
            pass
    return mapping


def _replace_jargon(text: str) -> tuple[str, bool]:
    result = text
    changed = False
    for shorthand, expansion in _jargon_map().items():
        pattern = re.compile(rf"(?<![\w]){re.escape(shorthand)}(?![\w])")
        replacement = f"{shorthand} ({expansion})"
        result, count = pattern.subn(replacement, result, count=1)
        changed = changed or count > 0
    return result, changed


def _optimize_text(text: str, *, structured_output: bool) -> tuple[str, list[str]]:
    changes: list[str] = []
    optimized, jargon_changed = _replace_jargon(text)
    if jargon_changed:
        changes.append("jargon-expanded")

    if _VARIATION_PATTERN.search(optimized) and (_ITERATION_PATTERN.search(optimized) or "all" not in optimized.lower()):
        optimized = f"{optimized.rstrip()} Provide all requested variations in one response, labeled separately."
        changes.append("variations-combined")

    if not _ROLE_PATTERN.search(optimized):
        optimized = f"Act as a precise domain expert. {optimized}"
        changes.append("role-added")

    if not _LENGTH_PATTERN.search(optimized):
        optimized = f"{optimized.rstrip()} Keep the response concise and organized into no more than 5 bullets."
        changes.append("length-constrained")

    if structured_output and not re.search(r"\b(?:JSON|schema|structured|field(?:s)?|format)\b", optimized, re.IGNORECASE):
        optimized = f'{optimized.rstrip()} Return only valid JSON with the fields required by the response schema.'
        changes.append("format-specified")

    return canonicalize(optimized), changes


def optimize_gateway_request(gateway_request: GatewayRequest) -> OptimizationResult:
    original_prompt = gateway_request.normalized_prompt or prompt_text(gateway_request.raw_payload)
    original_tokens = gateway_request.input_tokens or 0
    payload = json.loads(json.dumps(gateway_request.raw_payload))
    structured_output = isinstance(payload.get("response_format"), dict)
    changes: list[str] = []

    for message in payload.get("messages", []):
        if message.get("role") in {"user", "system"} and isinstance(message.get("content"), str):
            message["content"], message_changes = _optimize_text(message["content"], structured_output=structured_output)
            changes.extend(message_changes)

    optimized_prompt = prompt_text(payload)
    model = str(payload.get("model", "gpt-3.5-turbo"))
    optimized_request = gateway_request.model_copy(
        update={
            "raw_payload": payload,
            "normalized_prompt": optimized_prompt,
            "content_hash": hashlib.sha256(optimized_prompt.encode("utf-8")).hexdigest(),
            "input_tokens": count_tokens(optimized_prompt, model),
        }
    )
    return OptimizationResult(
        request=optimized_request,
        original_prompt=original_prompt,
        optimized_prompt=optimized_prompt,
        original_tokens=original_tokens,
        optimized_tokens=optimized_request.input_tokens or 0,
        optimized=bool(changes),
        changes=tuple(dict.fromkeys(changes)),
    )
