"""Run the quota-bounded, counterbalanced Gemini 2.5 Flash probe."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from ingress.normalization import build_gateway_request, count_tokens
from optimizers.prompt_optimizer import optimize_gateway_request

MODEL = "gemini-2.5-flash"
MAX_TOKENS = 1024
REQUEST_INTERVAL_SECONDS = 16
PROMPT_FILE = Path(__file__).with_name("gemini_probe_round2_prompts.json")
DEFAULT_RESULTS_FILE = Path(__file__).with_name("gemini_probe_round2_results.json")
DEFAULT_ATTEMPTS_FILE = Path(__file__).with_name("gemini_probe_round2_retry_attempts.json")
ORDERS = (("baseline", "optimized"), ("optimized", "baseline"), ("baseline", "optimized"))
RETRY_STATUSES = {429, 500, 503}
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 30


def response_metrics(payload: dict[str, Any], prompt_tokens: int) -> dict[str, Any]:
    choices = payload.get("choices", [])
    choice = choices[0] if choices else {}
    usage = payload.get("usage", {})
    text = choice.get("message", {}).get("content", "")
    if not isinstance(text, str):
        text = ""
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    hidden_tokens = (
        max(0, total_tokens - prompt_tokens - completion_tokens)
        if isinstance(total_tokens, int) and isinstance(completion_tokens, int)
        else None
    )
    return {
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": completion_tokens,
        "hidden_tokens": hidden_tokens,
        "word_count": len(text.split()),
        "top_level_bullet_count": len(re.findall(r"(?m)^(?:[-*+]|\d+[.)])\s+", text)),
        "total_bullet_lines": len(re.findall(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+", text)),
        "heading_count": len(re.findall(r"(?m)^#{1,6}\s+", text)),
        "counts_are_lower_bounds": choice.get("finish_reason") != "stop",
    }


def persist_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def request_with_retries(
    client: httpx.AsyncClient,
    *,
    url: str,
    payload: dict[str, Any],
    sample: dict[str, Any],
    attempts: list[dict[str, Any]],
    attempts_path: Path,
    last_request_at: float | None,
    attempt_budget: int,
) -> tuple[httpx.Response, float]:
    retry_number = 0
    while True:
        minimum_spacing = REQUEST_INTERVAL_SECONDS
        if attempts and attempts[-1]["sample_number"] == sample["sample_number"] and retry_number > 0:
            minimum_spacing = RETRY_DELAY_SECONDS
        if last_request_at is not None:
            wait_seconds = minimum_spacing - (time.monotonic() - last_request_at)
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
        if len(attempts) >= attempt_budget:
            raise RuntimeError(
                f"API-attempt budget ({attempt_budget}) exhausted; stopping before using the two reserved requests"
            )

        attempt = {
            **sample,
            "attempt_number_for_sample": retry_number + 1,
            "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            response = await client.post(url, json=payload)
        except httpx.HTTPError as error:
            attempt.update({"transport_error": str(error), "status_code": None, "response_body": None})
            attempts.append(attempt)
            persist_json(attempts_path, {"attempts": attempts})
            raise

        attempt["status_code"] = response.status_code
        attempt["response_body"] = response.text
        attempts.append(attempt)
        persist_json(attempts_path, {"attempts": attempts})

        if response.status_code in RETRY_STATUSES and retry_number < MAX_RETRIES:
            retry_number += 1
            last_request_at = time.monotonic()
            continue
        response.raise_for_status()
        return response, time.monotonic()


async def main(args: argparse.Namespace) -> None:
    prompts = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    tasks: list[tuple[int, str, str, str, str]] = []
    for shape_index, item in enumerate(prompts):
        prompt = item["prompt"]
        token_count = count_tokens(prompt, MODEL)
        if not 60 <= token_count <= 150:
            raise ValueError(f"{item['shape']} prompt has {token_count} tokens; expected 60-150")
        for arm in ORDERS[shape_index]:
            url = args.baseline_gateway if arm == "baseline" else args.optimized_gateway
            tasks.append((shape_index + 1, item["shape"], arm, prompt, url))
    if args.dry_run:
        print(json.dumps([
            {
                "shape_number": shape_number,
                "shape": shape,
                "arm": arm,
                "url": url,
                "prompt_tokens_tiktoken_estimate": count_tokens(prompt, MODEL),
            }
            for shape_number, shape, arm, prompt, url in tasks
        ], indent=2))
        return
    results: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    last_request_at: float | None = None
    async with httpx.AsyncClient(timeout=120.0) as client:
        for request_number, (shape_number, shape, arm, prompt, url) in enumerate(tasks, start=1):
            baseline_request = build_gateway_request(
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": MAX_TOKENS,
                    "reasoning_effort": "none",
                },
                None,
            )
            effective_request = (
                optimize_gateway_request(baseline_request).request
                if arm == "optimized"
                else baseline_request
            )
            request_payload = {
                **baseline_request.raw_payload,
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "reasoning_effort": "none",
            }
            sample = {
                "sample_number": request_number,
                "shape_number": shape_number,
                "shape": shape,
                "arm": arm,
                "url": url,
                "model": MODEL,
                "reasoning_effort": "none",
                "max_tokens": MAX_TOKENS,
            }
            response, last_request_at = await request_with_retries(
                client,
                url=url,
                payload=request_payload,
                sample=sample,
                attempts=attempts,
                attempts_path=args.attempts_output,
                last_request_at=last_request_at,
                attempt_budget=args.max_api_attempts,
            )
            response_payload = response.json()
            response_headers = response.headers
            response_record = {
                "sample": request_number,
                "shape_number": shape_number,
                "shape": shape,
                "arm": arm,
                "url": url,
                "model": MODEL,
                "reasoning_effort": "none",
                "max_tokens": MAX_TOKENS,
                "prompt": prompt,
                "effective_prompt": effective_request.raw_payload["messages"][-1]["content"],
                "prompt_tokens_tiktoken_estimate": count_tokens(prompt, MODEL),
                "raw_response": response_payload,
                "metrics": response_metrics(response_payload, int(response_payload.get("usage", {}).get("prompt_tokens", 0))),
                "gateway": {
                    "token_count_source": response_headers.get("x-greenway-token-source"),
                    "telemetry_source": response_headers.get("x-greenway-telemetry-source"),
                    "energy_source_tier": response_headers.get("x-greenway-energy-source"),
                    "energy_kwh": response_headers.get("x-greenway-energy-kwh"),
                    "carbon_source_tier": response_headers.get("x-greenway-carbon-source"),
                    "carbon_gco2e": response_headers.get("x-greenway-carbon-gco2e"),
                },
                "attempt_numbers": [
                    index
                    for index, attempt in enumerate(attempts, start=1)
                    if attempt["sample_number"] == request_number
                ],
            }
            results.append(response_record)
            persist_json(args.output, {"samples": results, "attempts": attempts})
            print(json.dumps(response_record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-gateway", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--optimized-gateway", default="http://127.0.0.1:8002/v1/chat/completions")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--attempts-output", type=Path, default=DEFAULT_ATTEMPTS_FILE)
    parser.add_argument(
        "--max-api-attempts",
        type=int,
        required=True,
        help="Maximum total requests, including retries; leave at least two displayed RPD requests unused.",
    )
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(main(parser.parse_args()))
