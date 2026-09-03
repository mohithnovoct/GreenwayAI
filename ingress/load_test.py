"""Compare the full gateway path with a direct backend HTTP path."""

import argparse
import asyncio
import os
import statistics
import time

import httpx

PAYLOAD = {"model": "greenway-mock", "messages": [{"role": "user", "content": "Hello from GreenwayAI."}]}


async def measure_pair(
    client: httpx.AsyncClient,
    gateway_url: str,
    backend_url: str,
    count: int,
    concurrency: int,
) -> tuple[list[float], list[float]]:
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


async def measure_ab_sample(
    client: httpx.AsyncClient,
    gateway_url: str,
    count: int,
    concurrency: int,
) -> dict[str, list[float]]:
    semaphore = asyncio.Semaphore(concurrency)
    values: dict[str, list[float]] = {"optimized": [], "baseline": []}

    async def one() -> None:
        async with semaphore:
            response = await client.post(f"{gateway_url}/v1/chat/completions", json=PAYLOAD)
            response.raise_for_status()
            arm = response.headers.get("X-Greenway-Prompt-Optimization", "unknown")
            if arm not in values:
                return
            for name in ("Energy-kWh", "Carbon-gCO2e"):
                value = response.headers.get(f"X-Greenway-{name}")
                if value not in (None, "null"):
                    values.setdefault(f"{arm}:{name}", []).append(float(value))

    await asyncio.gather(*(one() for _ in range(count)))
    return values


def print_ab_report(values: dict[str, list[float]]) -> None:
    print("A/B telemetry report (mean per request)")
    for metric in ("Energy-kWh", "Carbon-gCO2e"):
        optimized = values.get(f"optimized:{metric}", [])
        baseline = values.get(f"baseline:{metric}", [])
        if not optimized or not baseline:
            print(f"{metric}: unavailable (both arms need reported telemetry)")
            continue
        optimized_mean = statistics.fmean(optimized)
        baseline_mean = statistics.fmean(baseline)
        delta = optimized_mean - baseline_mean
        print(
            f"{metric}: optimized={optimized_mean:.12g} baseline={baseline_mean:.12g} "
            f"delta={delta:+.12g} ({len(optimized)}/{len(baseline)} samples)"
        )


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


def print_resource_state(label: str) -> None:
    load_average, available_memory_mib = resource_state()
    memory = f"{available_memory_mib:.1f} MiB" if available_memory_mib is not None else "unknown"
    print(f"resource {label}: load_avg_1m={load_average:.2f} available_memory={memory}")


async def main(args: argparse.Namespace) -> None:
    print_resource_state("before")
    async with httpx.AsyncClient(timeout=30.0) as client:
        backend, gateway = await measure_pair(client, args.gateway, args.backend, args.requests, args.concurrency)
    print_resource_state("after")
    print("path       p50_ms   p95_ms   p99_ms")
    for name, values in (("gateway", gateway), ("backend", backend)):
        print(f"{name:<10} {statistics.median(values):7.2f} {percentile(values, .95):8.2f} {percentile(values, .99):8.2f}")
    paired_tax = [gateway[index] - backend[index] for index in range(len(gateway))]
    gateway_tax = [statistics.median(paired_tax), percentile(paired_tax, .95), percentile(paired_tax, .99)]
    print(f"gateway tax p50/p95/p99: {gateway_tax[0]:.2f}/{gateway_tax[1]:.2f}/{gateway_tax[2]:.2f} ms")
    if args.ab_sample:
        async with httpx.AsyncClient(timeout=30.0) as client:
            values = await measure_ab_sample(client, args.gateway, args.ab_sample, args.concurrency)
        print_ab_report(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="http://localhost:8000")
    parser.add_argument("--backend", default="http://localhost:8001")
    parser.add_argument("-n", "--requests", type=int, default=100)
    parser.add_argument("-c", "--concurrency", type=int, default=10)
    parser.add_argument("--ab-sample", type=int, default=0, help="Collect per-arm energy/carbon means from gateway headers.")
    asyncio.run(main(parser.parse_args()))