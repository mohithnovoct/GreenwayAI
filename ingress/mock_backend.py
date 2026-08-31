import asyncio
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI

app = FastAPI(title="GreenwayAI Mock Backend", version="0.1.0")


def mock_response(payload: dict[str, Any]) -> dict[str, Any]:
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
    delay_seconds = float(os.getenv("GREENWAY_MOCK_DELAY_SECONDS", "0"))
    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)
    return mock_response(payload)