import asyncio
import time

import httpx


async def handle_client(reader, writer):
    await asyncio.sleep(0.2)
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK")
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def send_one(client: httpx.AsyncClient, idx: int, url: str) -> None:
    trace_started = time.perf_counter()
    pool_acquire_ms = 0.0

    async def trace(name: str, info: dict) -> None:
        nonlocal pool_acquire_ms
        if name == "connection.connect_tcp.started":
            pool_acquire_ms = (time.perf_counter() - trace_started) * 1000.0

    request = client.build_request("GET", url)
    request.extensions["trace"] = trace
    response = await client.send(request)
    print(f"request {idx}: pool_acquire_ms={pool_acquire_ms:.3f}, status={response.status_code}")


async def main() -> None:
    server = await asyncio.start_server(handle_client, "127.0.0.1", 18081)
    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        ) as client:
            await asyncio.gather(*(send_one(client, i, "http://127.0.0.1:18081/") for i in range(2)))
    finally:
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
