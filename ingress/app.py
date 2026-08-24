import json
import logging
import os
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from ingress.normalization import build_gateway_request

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger("greenwayai.ingress")

app = FastAPI(title="GreenwayAI Gateway", version="0.2.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _mock_response(payload: dict[str, Any]) -> dict[str, Any]:
    message = payload.get("messages", [{}])[-1]
    content = message.get("content", "") if isinstance(message, dict) else ""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", "greenway-mock"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_greenway_sla: str | None = Header(default=None),
) -> Response:
    started = time.perf_counter()
    payload = await request.json()
    gateway_request = build_gateway_request(payload, x_greenway_sla.strip().lower() if x_greenway_sla else None)
    metadata = {
        "X-Greenway-Request-ID": gateway_request.request_id,
        "X-Greenway-Content-Hash": gateway_request.content_hash or "",
        "X-Greenway-Input-Tokens": str(gateway_request.input_tokens),
    }
    backend_url = os.getenv("VLLM_BACKEND_URL", "").strip().rstrip("/")
    try:
        if not backend_url:
            response = JSONResponse(_mock_response(gateway_request.raw_payload), headers=metadata)
        else:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{backend_url}/v1/chat/completions",
                    json=gateway_request.raw_payload,
                    headers={**metadata, "Content-Type": "application/json"},
                )
            response.headers.update(metadata)
    except httpx.HTTPError as error:
        response = JSONResponse({"error": {"message": f"Backend unavailable: {error}"}}, status_code=502, headers=metadata)
    logger.info(json.dumps({
        "request_id": gateway_request.request_id,
        "content_hash": gateway_request.content_hash,
        "input_tokens": gateway_request.input_tokens,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }))
    return response