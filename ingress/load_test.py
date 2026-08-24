"""Compare the full gateway path with a direct backend HTTP path."""

import argparse
import asyncio
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


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(int(len(values) * fraction), len(values) - 1)]


async def main(args: argparse.Namespace) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        backend, gateway = await measure_pair(client, args.gateway, args.backend, args.requests, args.concurrency)
    print("path       p50_ms   p95_ms   p99_ms")
    for name, values in (("gateway", gateway), ("backend", backend)):
        print(f"{name:<10} {statistics.median(values):7.2f} {percentile(values, .95):8.2f} {percentile(values, .99):8.2f}")
    paired_tax = [gateway[index] - backend[index] for index in range(len(gateway))]
    gateway_tax = [statistics.median(paired_tax), percentile(paired_tax, .95), percentile(paired_tax, .99)]
    print(f"gateway tax p50/p95/p99: {gateway_tax[0]:.2f}/{gateway_tax[1]:.2f}/{gateway_tax[2]:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="http://localhost:8000")
    parser.add_argument("--backend", default="http://localhost:8001")
    parser.add_argument("-n", "--requests", type=int, default=100)
    parser.add_argument("-c", "--concurrency", type=int, default=10)
    asyncio.run(main(parser.parse_args()))