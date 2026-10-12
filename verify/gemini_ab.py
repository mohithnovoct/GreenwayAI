"""Small counterbalanced Gemini A/B run with raw per-request evidence."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
import re
import statistics
import time
from typing import Any

import httpx

from ingress.interleaved_ab import PAYLOAD
from optimizers.prompt_optimizer import optimize_gateway_request
from ingress.normalization import build_gateway_request


async def call(client: httpx.AsyncClient, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> tuple[dict[str, Any], httpx.Headers, float]:
    started = time.perf_counter()
    response = await client.post(url, json=payload, headers=headers)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), response.headers, elapsed_ms


def usage(payload: dict[str, Any]) -> tuple[int, int]:
    values = payload.get("usage", {})
    return int(values.get("prompt_tokens", 0)), int(values.get("completion_tokens", 0))


def estimate_gateway_tax_ms(gateway_ms: float, headers: httpx.Headers) -> float | None:
    """Estimate gateway overhead by subtracting its measured upstream round trip."""
    upstream_ms = headers.get("x-greenway-round-trip-ms")
    return gateway_ms - float(upstream_ms) if upstream_ms is not None else None


def compliance(text: str) -> dict[str, Any]:
    bullets = [line for line in text.splitlines() if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line)]
    return {
        "bullet_count": len(bullets),
        "five_or_fewer_bullets": len(bullets) <= 5,
        "expert_persona_signal": bool(re.search(r"\b(as a|as an|domain expert|expert|specialist|professional)\b", text, re.IGNORECASE)),
    }


async def main(args: argparse.Namespace) -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required")
    original_request = build_gateway_request(PAYLOAD, None)
    optimized_request = optimize_gateway_request(original_request).request
    payloads = {
        "baseline": {**PAYLOAD, "max_tokens": 256},
        "optimized": {**optimized_request.raw_payload, "max_tokens": 256},
    }
    prompts = {"baseline": original_request.normalized_prompt, "optimized": optimized_request.normalized_prompt}
    records: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for round_number in range(1, args.rounds + 1):
            order = ("baseline", "optimized") if round_number % 2 else ("optimized", "baseline")
            for arm in order:
                gateway_url = args.baseline_gateway if arm == "baseline" else args.optimized_gateway
                attempted_at_utc = datetime.now(timezone.utc).isoformat()
                print(
                    json.dumps(
                        {
                            "event": "request_attempt",
                            "round": round_number,
                            "arm": arm,
                            "attempted_at_utc": attempted_at_utc,
                        }
                    ),
                    flush=True,
                )
                gateway_payload, gateway_headers, gateway_ms = await call(client, gateway_url, payloads[arm])
                input_tokens, output_tokens = usage(gateway_payload)
                response_text = gateway_payload.get("choices", [{}])[0].get("message", {}).get("content", "")
                records.append({
                    "round": round_number,
                    "arm": arm,
                    "attempted_at_utc": attempted_at_utc,
                    "prompt": prompts[arm],
                    "response": gateway_payload,
                    "response_text": response_text,
                    "compliance": compliance(response_text),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "energy_kwh": gateway_headers.get("x-greenway-energy-kwh"),
                    "carbon_gco2e": gateway_headers.get("x-greenway-carbon-gco2e"),
                    "gateway_ms": gateway_ms,
                    "gateway_tax_ms": estimate_gateway_tax_ms(gateway_ms, gateway_headers),
                })
                print(json.dumps(records[-1], ensure_ascii=True), flush=True)

    for arm in ("baseline", "optimized"):
        rows = [row for row in records if row["arm"] == arm]
        gateway_tax_values = [
            row["gateway_tax_ms"] for row in rows if row["gateway_tax_ms"] is not None
        ]
        print(json.dumps({
            "aggregate": arm,
            "requests": len(rows),
            "mean_input_tokens": statistics.fmean(row["input_tokens"] for row in rows),
            "mean_output_tokens": statistics.fmean(row["output_tokens"] for row in rows),
            "mean_gateway_tax_ms": statistics.fmean(gateway_tax_values) if gateway_tax_values else None,
            "compliance_rate": statistics.fmean(row["compliance"]["five_or_fewer_bullets"] for row in rows),
        }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-gateway", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--optimized-gateway", default="http://127.0.0.1:8002/v1/chat/completions")
    parser.add_argument("--rounds", type=int, default=10)
    asyncio.run(main(parser.parse_args()))
