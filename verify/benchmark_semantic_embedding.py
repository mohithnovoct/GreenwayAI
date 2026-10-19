import asyncio
import json
import math
import statistics
import time
from datetime import datetime, timezone

import torch

from optimizers.semantic_cache import (
    DEFAULT_MODEL_REVISION,
    SentenceTransformerEmbedder,
)


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
PROMPT_TEMPLATE = (
    "Design an API gateway cache for repeated language-model requests. Define a "
    "safe key using tenant, model, canonical messages, and output-affecting "
    "parameters. Describe exact lookup before inference, safe storage of complete "
    "responses, expiration, and tenant isolation. Explain fail-open behavior when "
    "Redis or embedding lookup is unavailable, and distinguish complete answers "
    "from truncated ones. State how concurrent identical misses should coalesce "
    "and which metrics expose hits, misses, bypasses, and modeled avoided energy. "
    "Case {case_id:03d}."
)
PROMPT_COUNT = 100
WARMUP_CALLS = 5
CPU_THREADS = 2


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


async def main() -> None:
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(CPU_THREADS)
    embedder = SentenceTransformerEmbedder(MODEL_ID, DEFAULT_MODEL_REVISION)
    prompts = [
        PROMPT_TEMPLATE.format(case_id=case_id)
        for case_id in range(1, PROMPT_COUNT + 1)
    ]
    token_counts = [
        len(embedder._model.tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in prompts
    ]
    if min(token_counts) < 90 or max(token_counts) > 110:
        raise ValueError(
            f"Prompts must each contain 90-110 model tokens; observed "
            f"{min(token_counts)}-{max(token_counts)}"
        )

    for warmup_index in range(WARMUP_CALLS):
        await asyncio.to_thread(embedder.encode, prompts[warmup_index])

    latency_ms: list[float] = []
    for prompt in prompts:
        started = time.perf_counter_ns()
        await asyncio.to_thread(embedder.encode, prompt)
        latency_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    print(
        json.dumps(
            {
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_id": MODEL_ID,
                "model_revision": DEFAULT_MODEL_REVISION,
                "dimension": embedder.dimension,
                "device": "cpu",
                "cpu_quota": 2,
                "torch_intraop_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "warmup_calls": WARMUP_CALLS,
                "measured_calls": len(latency_ms),
                "prompt_token_counts": {
                    "min": min(token_counts),
                    "max": max(token_counts),
                    "mean": statistics.mean(token_counts),
                },
                "latency_ms": {
                    "p50_nearest_rank": percentile_nearest_rank(latency_ms, 0.50),
                    "p95_nearest_rank": percentile_nearest_rank(latency_ms, 0.95),
                    "samples": latency_ms,
                },
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
