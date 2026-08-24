import json
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from ingress.mock_backend import mock_response
from ingress.normalization import build_gateway_request

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger("greenwayai.ingress")

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.http_client = httpx.AsyncClient(timeout=120.0)
    try:
        yield
    finally:
        await application.state.http_client.aclose()


app = FastAPI(title="GreenwayAI Gateway", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_greenway_sla: str | None = Header(default=None),
) -> Response:
    started = time.perf_counter()
    payload = await request.json()
    normalization_started = time.perf_counter()
    gateway_request = await asyncio.to_thread(
        build_gateway_request,
        payload,
        x_greenway_sla.strip().lower() if x_greenway_sla else None,
    )
    normalization_ms = (time.perf_counter() - normalization_started) * 1000
    metadata = {
        "X-Greenway-Request-ID": gateway_request.request_id,
        "X-Greenway-Content-Hash": gateway_request.content_hash or "",
        "X-Greenway-Input-Tokens": str(gateway_request.input_tokens),
    }
    backend_url = os.getenv("VLLM_BACKEND_URL", "").strip().rstrip("/")
    proxy_started = time.perf_counter()
    try:
        if not backend_url:
            response = JSONResponse(mock_response(gateway_request.raw_payload), headers=metadata)
        else:
            backend_response = await request.app.state.http_client.post(
                f"{backend_url}/v1/chat/completions",
                json=gateway_request.raw_payload,
                headers={**metadata, "Content-Type": "application/json"},
            )
            response_headers = {
                key: value
                for key, value in backend_response.headers.items()
                if key.lower() not in {"content-length", "transfer-encoding", "connection"}
            }
            response_headers.update(metadata)
            response = Response(
                content=backend_response.content,
                status_code=backend_response.status_code,
                headers=response_headers,
            )
    except httpx.HTTPError as error:
        response = JSONResponse({"error": {"message": f"Backend unavailable: {error}"}}, status_code=502, headers=metadata)
    proxy_ms = (time.perf_counter() - proxy_started) * 1000
    logger.info(json.dumps({
        "request_id": gateway_request.request_id,
        "content_hash": gateway_request.content_hash,
        "input_tokens": gateway_request.input_tokens,
        "normalization_ms": round(normalization_ms, 3),
        "proxy_ms": round(proxy_ms, 3),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }))
    return response