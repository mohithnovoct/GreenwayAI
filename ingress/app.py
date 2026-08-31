import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from carbon.carbon_engine import CarbonEngine
from ingress.mock_backend import mock_response
from ingress.normalization import build_gateway_request, count_tokens
from telemetry.energy_model import estimate_energy_kwh
from telemetry.nvml_sampler import NVMLPowerSampler
from telemetry.parametric import ParametricEnergyEstimator

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger("greenwayai.ingress")

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.executor = ThreadPoolExecutor(max_workers=50, thread_name_prefix="greenway-ingress")
    asyncio.get_running_loop().set_default_executor(application.state.executor)
    application.state.http_client = httpx.AsyncClient(
        timeout=120.0,
        limits=httpx.Limits(
            max_keepalive_connections=int(os.getenv("GREENWAY_MAX_KEEPALIVE_CONNECTIONS", "100")),
            max_connections=int(os.getenv("GREENWAY_MAX_CONNECTIONS", "100")),
        ),
    )
    application.state.nvml_sampler = NVMLPowerSampler()
    application.state.nvml_sampler.start()
    application.state.parametric_estimator = ParametricEnergyEstimator()
    application.state.carbon_engine = CarbonEngine(region=os.getenv("GREENWAY_REGION", "us-east1"), provider=os.getenv("GREENWAY_PROVIDER", "google"))
    application.state.carbon_moer_task = asyncio.create_task(application.state.carbon_engine.refresh_moer_cache())
    try:
        yield
    finally:
        if not application.state.carbon_moer_task.done():
            application.state.carbon_moer_task.cancel()
        application.state.nvml_sampler.stop()
        await application.state.http_client.aclose()
        application.state.executor.shutdown(wait=True)


app = FastAPI(title="GreenwayAI Gateway", version="0.2.0", lifespan=lifespan)


def request_energy_window_start(trace_started: float, pool_acquire_ms: float) -> float:
    return trace_started + (pool_acquire_ms / 1000.0)


def request_energy_window_end(trace_started: float, pool_acquire_ms: float, round_trip_ms: float) -> float:
    return trace_started + ((pool_acquire_ms + round_trip_ms) / 1000.0)


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
    pool_acquire_ms = 0.0
    round_trip_ms = 0.0
    response_read_ms = 0.0
    output_tokens = 0
    energy_kwh = 0.0
    carbon_gco2e = 0.0
    moer_kg_per_kwh = 0.0
    carbon_source_tier = "static_fallback"
    hardware_energy_kwh = 0.0
    energy_source_tier: str | None = None
    response_payload: dict[str, Any] = {}
    sampler = getattr(request.app.state, "nvml_sampler", None)
    parametric_estimator = getattr(request.app.state, "parametric_estimator", None)
    parametric_mode = os.getenv("GREENWAY_ENERGY_MODE", "nvml").strip().lower() == "parametric"
    telemetry_source = "unknown"
    if parametric_mode:
        telemetry_source = "parametric_estimated"
    elif sampler is not None:
        telemetry_source = sampler.telemetry_source
    try:
        if not backend_url:
            response_payload = mock_response(gateway_request.raw_payload)
            response = JSONResponse(response_payload, headers=metadata)
        else:
            trace_started = time.perf_counter()

            async def trace(name: str, info: dict[str, Any]) -> None:
                nonlocal pool_acquire_ms
                if name == "connection.connect_tcp.started":
                    pool_acquire_ms = (time.perf_counter() - trace_started) * 1000
                    if sampler is not None:
                        sampler.register_request(
                            gateway_request.request_id,
                            start_ts=request_energy_window_start(trace_started, pool_acquire_ms),
                            end_ts=trace_started,
                        )

            if sampler is not None and not parametric_mode:
                sampler.register_request(gateway_request.request_id, start_ts=trace_started, end_ts=trace_started)
            backend_request = request.app.state.http_client.build_request(
                "POST",
                f"{backend_url}/v1/chat/completions",
                json=gateway_request.raw_payload,
                headers={**metadata, "Content-Type": "application/json"},
            )
            backend_request.extensions["trace"] = trace
            backend_response = await request.app.state.http_client.send(backend_request, stream=True)
            round_trip_ms = (time.perf_counter() - trace_started) * 1000 - pool_acquire_ms
            response_body_started = time.perf_counter()
            response_body = await backend_response.aread()
            response_read_ms = (time.perf_counter() - response_body_started) * 1000
            await backend_response.aclose()
            if sampler is not None and not parametric_mode:
                sampler.complete_request(
                    gateway_request.request_id,
                    end_ts=request_energy_window_end(trace_started, pool_acquire_ms, round_trip_ms),
                )
                hardware_energy_kwh = sampler.estimate_request_energy_kwh(gateway_request.request_id)
            response_headers = {
                key: value
                for key, value in backend_response.headers.items()
                if key.lower() not in {"content-length", "transfer-encoding", "connection"}
            }
            response_headers.update(metadata)
            response_payload = json.loads(response_body.decode("utf-8")) if response_body else {}
            response = Response(
                content=response_body,
                status_code=backend_response.status_code,
                headers=response_headers,
            )
    except httpx.HTTPError as error:
        response = JSONResponse({"error": {"message": f"Backend unavailable: {error}"}}, status_code=502, headers=metadata)

    def _count_response_tokens(payload: dict[str, Any]) -> int:
        choices = payload.get("choices", [])
        if not choices:
            return 0
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return count_tokens(content, str(gateway_request.raw_payload.get("model", "gpt-3.5-turbo")))
        if isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        texts.append(text)
            return count_tokens("\n".join(texts), str(gateway_request.raw_payload.get("model", "gpt-3.5-turbo")))
        return 0

    if response_payload:
        output_tokens = await asyncio.to_thread(_count_response_tokens, response_payload)
        if parametric_mode and parametric_estimator is not None:
            estimate = parametric_estimator.estimate(
                model=str(gateway_request.raw_payload.get("model", "unknown")),
                input_tokens=gateway_request.input_tokens or 0,
                output_tokens=output_tokens,
            )
            energy_kwh = estimate.energy_kwh
            energy_source_tier = estimate.energy_source_tier
        else:
            energy_kwh = estimate_energy_kwh(
                input_tokens=gateway_request.input_tokens or 0,
                output_tokens=output_tokens,
            ) + hardware_energy_kwh
            if os.getenv("GREENWAY_CARBON_DISABLED", "false").strip().lower() == "true":
                moer_kg_per_kwh = request.app.state.carbon_engine._get_static_moer()
                carbon_source_tier = "static_fallback"
            else:
                moer_kg_per_kwh, carbon_source_tier = await request.app.state.carbon_engine.get_moer_with_tier()
        carbon_gco2e = request.app.state.carbon_engine.estimate_carbon_gco2e(
            energy_kwh=energy_kwh,
            moer_kg_per_kwh=moer_kg_per_kwh,
        )
        proxy_ms = (time.perf_counter() - proxy_started) * 1000
        response.headers["X-Greenway-Output-Tokens"] = str(output_tokens)
        response.headers["X-Greenway-Energy-kWh"] = str(round(energy_kwh, 12))
        response.headers["X-Greenway-Carbon-gCO2e"] = str(round(carbon_gco2e, 12))
        response.headers["X-Greenway-MOER-kgPerKWh"] = str(round(moer_kg_per_kwh, 12))
        response.headers["X-Greenway-Carbon-Source"] = carbon_source_tier
        response.headers["X-Greenway-Telemetry-Source"] = telemetry_source
        response.headers["X-Greenway-Energy-Source"] = energy_source_tier or "nvml"
        response.headers["X-Greenway-Normalization-ms"] = str(round(normalization_ms, 3))
        response.headers["X-Greenway-Proxy-ms"] = str(round(proxy_ms, 3))
        response.headers["X-Greenway-Pool-Acquire-ms"] = str(round(pool_acquire_ms, 3))
        response.headers["X-Greenway-Round-Trip-ms"] = str(round(round_trip_ms, 3))
        response.headers["X-Greenway-Response-Read-ms"] = str(round(response_read_ms, 3))
        response.headers["X-Greenway-Latency-ms"] = str(round((time.perf_counter() - started) * 1000, 3))
    else:
        proxy_ms = (time.perf_counter() - proxy_started) * 1000
    return response