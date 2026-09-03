"""Interleaved A/B benchmark to control for host drift.

Alternates small batches (50 requests) between enabled and disabled gateways
over ~10 rounds each, recording resource state at batch start. This ensures
both arms are sampled throughout the same drifting host conditions.
"""

import argparse
import asyncio
import os
import statistics
import time

import httpx

PAYLOAD = {"model": "greenway-mock", "messages": [{"role": "user", "content": "Hello from GreenwayAI."}]}


async def measure_batch(
    client: httpx.AsyncClient,
    gateway_url: str,
    backend_url: str,
    count: int,
    concurrency: int,
) -> tuple[list[float], list[float]]:
    """Measure a single batch of requests."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one() -> tuple[float, float]:
        async with semaphore:
            started = time.perf_counter()
            response = await client.post(f"{backend_url}/v1/chat/completions", json=PAYLOAD)
            response.raise_for_status()
            backend_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            response = await client.post(f"{gateway_url}/v1/chat/completions", json=PAYLOAD)
            response.raise_for_status()
            gateway_ms = (time.perf_counter() - started) * 1000
            return backend_ms, gateway_ms

    pairs = await asyncio.gather(*(one() for _ in range(count)))
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(int(len(values) * fraction), len(values) - 1)]


def resource_state() -> tuple[float, float | None]:
    load_average = os.getloadavg()[0]
    available_memory_mib = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            values = dict(line.split(":", 1) for line in meminfo if ":" in line)
        available_memory_kib = values.get("MemAvailable", "").split()[0]
        available_memory_mib = float(available_memory_kib) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return load_average, available_memory_mib


def format_resource_state(label: str) -> str:
    load_average, available_memory_mib = resource_state()
    memory = f"{available_memory_mib:.1f} MiB" if available_memory_mib is not None else "unknown"
    return f"{label}: load_avg_1m={load_average:.2f} available_memory={memory}"


async def main(args: argparse.Namespace) -> None:
    print("=== INTERLEAVED A/B BENCHMARK ===")
    print(f"Batch size: {args.batch_size} requests, Concurrency: {args.concurrency}")
    print(f"Rounds: {args.rounds} (A/B pair = {args.batch_size * 2} total requests per round)")
    print()

    enabled_batches: list[list[float]] = []
    disabled_batches: list[list[float]] = []
    rows: list[tuple[int, str, str, float, str, float]] = []

    print(f"resource {format_resource_state('start')}")
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        for round_idx in range(args.rounds):
            print(f"--- Round {round_idx + 1}/{args.rounds} ---")
            first_enabled = round_idx % 2 == 0
            arm_order = "enabled -> disabled" if first_enabled else "disabled -> enabled"
            batches: dict[str, list[float]] = {}
            resources: dict[str, str] = {}
            for arm in ("enabled", "disabled") if first_enabled else ("disabled", "enabled"):
                resources[arm] = format_resource_state(f"before {arm}")
                gateway = args.enabled_gateway if arm == "enabled" else args.disabled_gateway
                _, taxes = await measure_batch(client, gateway, args.backend, args.batch_size, args.concurrency)
                batches[arm] = taxes
                print(f"resource {resources[arm]}")
                print(f"  {arm} p50={statistics.median(taxes):.2f}ms (n={len(taxes)})")
            enabled_taxes = batches["enabled"]
            disabled_taxes = batches["disabled"]
            enabled_batches.append(enabled_taxes)
            disabled_batches.append(disabled_taxes)
            rows.append((round_idx + 1, arm_order, resources["enabled"], statistics.median(enabled_taxes), resources["disabled"], statistics.median(disabled_taxes)))
            print()

    # Aggregate across all rounds
    all_enabled = [tax for batch in enabled_batches for tax in batch]
    all_disabled = [tax for batch in disabled_batches for tax in batch]

    print("=== AGGREGATE RESULTS ===")
    print()
    print(f"Total requests per arm: {len(all_enabled)}")
    print()
    print("arm        p50_tax_ms  p95_tax_ms  p99_tax_ms")
    for name, values in (("enabled", all_enabled), ("disabled", all_disabled)):
        p50 = statistics.median(values)
        p95 = percentile(values, 0.95)
        p99 = percentile(values, 0.99)
        print(f"{name:<10} {p50:10.2f} {p95:10.2f} {p99:10.2f}")

    # Difference
    enabled_p50 = statistics.median(all_enabled)
    disabled_p50 = statistics.median(all_disabled)
    diff = enabled_p50 - disabled_p50

    print()
    print(f"Difference (enabled - disabled) at p50: {diff:+.2f} ms")
    if abs(diff) < 5:
        print("→ No significant difference (< 5ms)")
    elif diff > 0:
        print(f"→ Enabled is slower by {diff:.2f}ms (possible carbon path cost)")
    else:
        print(f"→ Disabled is slower by {abs(diff):.2f}ms (carbon path not a cause)")

    print()
    print("round | arm order             | before enabled | enabled p50 | before disabled | disabled p50")
    for number, order, enabled_resource, enabled_p50, disabled_resource, disabled_p50 in rows:
        print(f"{number:5d} | {order:<21} | {enabled_resource} | {enabled_p50:11.2f} | {disabled_resource} | {disabled_p50:12.2f}")
    
    print()
    print(f"resource {format_resource_state('end')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interleaved A/B benchmark with host-drift control")
    parser.add_argument("--enabled-gateway", default="http://localhost:8000", help="Carbon-enabled gateway URL")
    parser.add_argument("--disabled-gateway", default="http://localhost:8002", help="Carbon-disabled gateway URL")
    parser.add_argument("--backend", default="http://localhost:8001", help="Backend URL")
    parser.add_argument("--batch-size", type=int, default=50, help="Requests per batch")
    parser.add_argument("--rounds", type=int, default=10, help="Number of A/B alternation rounds")
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="Concurrent requests per batch")
    asyncio.run(main(parser.parse_args()))
