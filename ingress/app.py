import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time
import uuid
import ipaddress
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from carbon.carbon_engine import CarbonEngine
from ingress.mock_backend import mock_response
from ingress.normalization import build_gateway_request, count_tokens
from optimizers.prompt_optimizer import optimize_gateway_request, should_optimize
from optimizers.semantic_cache import DEFAULT_MODEL_REVISION, SentenceTransformerEmbedder
from telemetry.energy_model import estimate_energy_kwh
from telemetry.nvml_sampler import NVMLPowerSampler
from telemetry.parametric import ParametricEnergyEstimator
from telemetry.otel_export import record_request_metrics, register_baseline_metrics

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger("greenwayai.ingress")

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.semantic_embedder = None
    if os.getenv("CACHE_ENABLED", "false").strip().lower() == "true":
        application.state.semantic_embedder = SentenceTransformerEmbedder(
            model_id=os.getenv(
                "CACHE_EMBEDDING_MODEL_ID",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            revision=os.getenv(
                "CACHE_EMBEDDING_MODEL_REVISION",
                DEFAULT_MODEL_REVISION,
            ),
        )
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
    application.state.metrics = register_baseline_metrics()
    application.state.dashboard_points = []
    application.state.carbon_engine = CarbonEngine(region=os.getenv("GREENWAY_REGION", "us-east1"), provider=os.getenv("GREENWAY_PROVIDER", "google"))
    application.state.carbon_moer_task = asyncio.create_task(application.state.carbon_engine.refresh_moer_cache())
    try:
        yield
    finally:
        if not application.state.carbon_moer_task.done():
            application.state.carbon_moer_task.cancel()
        application.state.nvml_sampler.stop()
        await application.state.carbon_engine.close()
        await application.state.http_client.aclose()
        application.state.executor.shutdown(wait=True)


app = FastAPI(title="GreenwayAI Gateway", version="0.2.0", lifespan=lifespan)

GEMINI_BACKEND_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def is_remote_backend_url(backend_url: str) -> bool:
    host = urlparse(backend_url).hostname
    if not host:
        return False
    normalized_host = host.lower().rstrip(".")
    if normalized_host in {"localhost", "localhost.localdomain"}:
        return False
    try:
        return not ipaddress.ip_address(normalized_host).is_private and normalized_host not in {"0.0.0.0", "::1"}
    except ValueError:
        return "." in normalized_host


def backend_configuration() -> tuple[str, str, dict[str, str], bool]:
    configured_mode = os.getenv("GREENWAY_BACKEND_MODE", "").strip().lower()
    mode = configured_mode or ("vllm" if os.getenv("VLLM_BACKEND_URL", "").strip() else "mock")
    if mode == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required when GREENWAY_BACKEND_MODE=gemini")
        return f"{GEMINI_BACKEND_URL}/chat/completions", "gemini-2.5-flash", {"Authorization": f"Bearer {api_key}"}, True
    if mode in {"mock", ""}:
        return "", "", {}, False
    backend_url = os.getenv("VLLM_BACKEND_URL", "").strip().rstrip("/")
    return (f"{backend_url}/v1/chat/completions" if backend_url else ""), "", {}, is_remote_backend_url(backend_url)


def request_energy_window_start(trace_started: float, pool_acquire_ms: float) -> float:
    return trace_started + (pool_acquire_ms / 1000.0)


def request_energy_window_end(trace_started: float, pool_acquire_ms: float, round_trip_ms: float) -> float:
    return trace_started + ((pool_acquire_ms + round_trip_ms) / 1000.0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/dashboard-data")
def dashboard_data(request: Request) -> list[dict[str, Any]]:
    return request.app.state.dashboard_points


@app.get("/dashboard", response_class=Response)
def dashboard() -> Response:
    return Response(content="""<!doctype html><html><head><meta charset='utf-8'><title>GreenwayAI carbon baseline</title><script src='https://cdn.jsdelivr.net/npm/chart.js'></script></head><body><h1>gCO2e per request</h1><canvas id='chart'></canvas><script>fetch('/dashboard-data').then(r=>r.json()).then(points=>new Chart(document.getElementById('chart'),{type:'line',data:{labels:points.map(p=>p.request_id),datasets:[{label:'gCO2e/request',data:points.map(p=>p.carbon_gco2e),pointBackgroundColor:points.map(p=>p.color)}]},options:{plugins:{tooltip:{callbacks:{afterLabel:c=>{const p=points[c.dataIndex];return `Telemetry: ${p.telemetry_source}, energy: ${p.energy_source_tier}, carbon: ${p.carbon_source_tier}`}}}}}}));</script></body></html>""", media_type="text/html")


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
    original_prompt = gateway_request.normalized_prompt or ""
    optimization_arm = "optimized" if should_optimize() else "baseline"
    if optimization_arm == "optimized":
        optimization_result = await asyncio.to_thread(optimize_gateway_request, gateway_request)
        gateway_request = optimization_result.request
        optimized_prompt = optimization_result.optimized_prompt
        optimized_tokens = optimization_result.optimized_tokens
        optimization_changes = optimization_result.changes
    else:
        optimized_prompt = original_prompt
        optimized_tokens = gateway_request.input_tokens or 0
        optimization_changes = ()
    metadata = {
        "X-Greenway-Request-ID": gateway_request.request_id,
        "X-Greenway-Content-Hash": gateway_request.content_hash or "",
        "X-Greenway-Input-Tokens": str(gateway_request.input_tokens),
        "X-Greenway-Prompt-Optimization": optimization_arm,
        "X-Greenway-Prompt-Optimization-Changes": ",".join(optimization_changes),
    }
    try:
        backend_url, backend_model, backend_headers, remote_backend = backend_configuration()
    except RuntimeError as error:
        return JSONResponse({"error": {"message": str(error)}}, status_code=503, headers=metadata)
    if backend_model:
        gateway_request.raw_payload["model"] = backend_model
    proxy_started = time.perf_counter()
    pool_acquire_ms = 0.0
    round_trip_ms = 0.0
    response_read_ms = 0.0
    output_tokens = 0
    output_tokens_hidden = 0
    decode_tokens = 0
    reported_input_tokens: int | None = None
    energy_kwh: float | None = None
    carbon_gco2e: float | None = None
    moer_kg_per_kwh: float | None = None
    carbon_source_tier = "unavailable"
    hardware_energy_kwh: float | None = None
    energy_source_tier = "unavailable"
    response_payload: dict[str, Any] = {}
    sampler = getattr(request.app.state, "nvml_sampler", None)
    parametric_estimator = getattr(request.app.state, "parametric_estimator", None)
    parametric_mode = bool(backend_url) and (
        remote_backend or os.getenv("GREENWAY_ENERGY_MODE", "nvml").strip().lower() == "parametric"
    )
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
                        try:
                            sampler.register_request(
                                gateway_request.request_id,
                                start_ts=request_energy_window_start(trace_started, pool_acquire_ms),
                                end_ts=trace_started,
                            )
                        except Exception as error:
                            logger.warning("NVML request registration failed: %s", error)

            if sampler is not None and not parametric_mode:
                try:
                    sampler.register_request(gateway_request.request_id, start_ts=trace_started, end_ts=trace_started)
                except Exception as error:
                    logger.warning("NVML request registration failed: %s", error)
            backend_request = request.app.state.http_client.build_request(
                "POST",
                backend_url,
                json=gateway_request.raw_payload,
                headers={**metadata, **backend_headers, "Content-Type": "application/json"},
            )
            backend_request.extensions["trace"] = trace
            backend_response = await request.app.state.http_client.send(backend_request, stream=True)
            round_trip_ms = (time.perf_counter() - trace_started) * 1000 - pool_acquire_ms
            response_body_started = time.perf_counter()
            response_body = await backend_response.aread()
            response_read_ms = (time.perf_counter() - response_body_started) * 1000
            await backend_response.aclose()
            if sampler is not None and not parametric_mode:
                try:
                    sampler.complete_request(
                        gateway_request.request_id,
                        end_ts=request_energy_window_end(trace_started, pool_acquire_ms, round_trip_ms),
                    )
                    hardware_energy_kwh = sampler.estimate_request_energy_kwh(gateway_request.request_id)
                    energy_source_tier = "nvml"
                except Exception as error:
                    logger.warning("NVML request energy estimation failed: %s", error)
                    hardware_energy_kwh = None
                    energy_source_tier = "unavailable"
            response_headers = {
                key: value
                for key, value in backend_response.headers.items()
                if key.lower() not in {"content-length", "transfer-encoding", "connection", "content-encoding"}
            }
            response_headers.update(metadata)
            decoded_payload = json.loads(response_body.decode("utf-8")) if response_body else {}
            response_payload = decoded_payload if isinstance(decoded_payload, dict) else {}
            response = Response(
                content=response_body,
                status_code=backend_response.status_code,
                headers=response_headers,
            )
    except httpx.HTTPError as error:
        response = JSONResponse({"error": {"message": f"Backend unavailable: {error}"}}, status_code=502, headers=metadata)

    def _reported_usage(payload: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None, None, None
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
            return None, None, None
        total_tokens = usage.get("total_tokens")
        if not isinstance(total_tokens, int) or total_tokens < 0:
            total_tokens = None
        return (
            prompt_tokens if prompt_tokens > 0 else None,
            completion_tokens if completion_tokens > 0 else None,
            total_tokens,
        )

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
        reported_input_tokens, reported_output_tokens, reported_total_tokens = _reported_usage(response_payload)
        estimated_input_tokens = gateway_request.input_tokens or 0
        estimated_output_tokens = await asyncio.to_thread(_count_response_tokens, response_payload)
        output_tokens = (
            reported_output_tokens
            if reported_output_tokens is not None
            else estimated_output_tokens
        )
        decode_tokens = output_tokens
        if reported_input_tokens is not None and reported_output_tokens is not None and reported_total_tokens is not None:
            decode_tokens = max(output_tokens, reported_total_tokens - reported_input_tokens)
        output_tokens_hidden = decode_tokens - output_tokens
        input_tokens = reported_input_tokens if reported_input_tokens is not None else estimated_input_tokens
        token_count_source = (
            "provider"
            if reported_input_tokens is not None and reported_output_tokens is not None
            else "tiktoken_estimated"
            if reported_input_tokens is None and reported_output_tokens is None
            else "mixed"
        )
        if parametric_mode and parametric_estimator is not None:
            estimate = parametric_estimator.estimate(
                model=str(gateway_request.raw_payload.get("model", "unknown")),
                input_tokens=input_tokens,
                output_tokens=decode_tokens,
            )
            energy_kwh = estimate.energy_kwh
            energy_source_tier = estimate.energy_source_tier
        else:
            if hardware_energy_kwh is not None:
                energy_kwh = hardware_energy_kwh
        if os.getenv("GREENWAY_CARBON_DISABLED", "false").strip().lower() == "true":
            moer_kg_per_kwh = request.app.state.carbon_engine._get_static_moer()
            carbon_source_tier = "static_fallback"
        elif energy_kwh is not None:
            try:
                moer_kg_per_kwh, carbon_source_tier = await request.app.state.carbon_engine.get_moer_with_tier()
            except Exception as error:
                logger.warning("Carbon telemetry unavailable: %s", error)
        if energy_kwh is not None and moer_kg_per_kwh is not None:
            carbon_gco2e = request.app.state.carbon_engine.estimate_carbon_gco2e(
                energy_kwh=energy_kwh,
                moer_kg_per_kwh=moer_kg_per_kwh,
            )
        elif moer_kg_per_kwh is None:
            carbon_source_tier = "unavailable"
        proxy_ms = (time.perf_counter() - proxy_started) * 1000
        response.headers["X-Greenway-Input-Tokens"] = str(input_tokens)
        response.headers["X-Greenway-Output-Tokens"] = str(output_tokens)
        response.headers["X-Greenway-Token-Source"] = token_count_source
        response.headers["X-Greenway-Energy-kWh"] = str(round(energy_kwh, 12)) if energy_kwh is not None else "null"
        response.headers["X-Greenway-Carbon-gCO2e"] = str(round(carbon_gco2e, 12)) if carbon_gco2e is not None else "null"
        response.headers["X-Greenway-MOER-kgPerKWh"] = str(round(moer_kg_per_kwh, 12)) if moer_kg_per_kwh is not None else "null"
        response.headers["X-Greenway-Carbon-Source"] = carbon_source_tier
        response.headers["X-Greenway-Telemetry-Source"] = telemetry_source
        response.headers["X-Greenway-Energy-Source"] = energy_source_tier
        response.headers["X-Greenway-Normalization-ms"] = str(round(normalization_ms, 3))
        response.headers["X-Greenway-Proxy-ms"] = str(round(proxy_ms, 3))
        response.headers["X-Greenway-Pool-Acquire-ms"] = str(round(pool_acquire_ms, 3))
        response.headers["X-Greenway-Round-Trip-ms"] = str(round(round_trip_ms, 3))
        response.headers["X-Greenway-Response-Read-ms"] = str(round(response_read_ms, 3))
        response.headers["X-Greenway-Latency-ms"] = str(round((time.perf_counter() - started) * 1000, 3))
        logger.info(json.dumps({
            "request_id": gateway_request.request_id,
            "content_hash": gateway_request.content_hash,
            "input_tokens": input_tokens,
            "prompt_optimization_arm": optimization_arm,
            "token_count_source": token_count_source,
            **({
                "original_prompt": original_prompt,
                "optimized_prompt": optimized_prompt,
            } if os.getenv("GREENWAY_LOG_PROMPTS", "false").strip().lower() == "true" else {}),
            "original_input_tokens": optimized_tokens if optimization_arm == "baseline" else optimization_result.original_tokens,
            "optimized_input_tokens": optimized_tokens,
            "prompt_optimization_changes": list(optimization_changes),
            "output_tokens": output_tokens,
            "output_tokens_visible": output_tokens,
            "output_tokens_hidden": output_tokens_hidden,
            "decode_tokens": decode_tokens,
            "energy_kwh": round(energy_kwh, 12) if energy_kwh is not None else None,
            "carbon_gco2e": round(carbon_gco2e, 12) if carbon_gco2e is not None else None,
            "moer_kg_per_kwh": round(moer_kg_per_kwh, 12) if moer_kg_per_kwh is not None else None,
            "carbon_source_tier": carbon_source_tier,
            "normalization_ms": round(normalization_ms, 3),
            "proxy_ms": round(proxy_ms, 3),
            "pool_acquire_ms": round(pool_acquire_ms, 3),
            "round_trip_ms": round(round_trip_ms, 3),
            "response_read_ms": round(response_read_ms, 3),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "telemetry_source": telemetry_source,
            "energy_source_tier": energy_source_tier,
        }))
        record_request_metrics(
            request.app.state.metrics,
            energy_kwh=energy_kwh,
            carbon_gco2e=carbon_gco2e,
            telemetry_source=telemetry_source,
            energy_source_tier=energy_source_tier,
            carbon_source_tier=carbon_source_tier,
        )
        request.app.state.dashboard_points.append({
            "request_id": gateway_request.request_id,
            "carbon_gco2e": carbon_gco2e,
            "telemetry_source": telemetry_source,
            "energy_source_tier": energy_source_tier,
            "carbon_source_tier": carbon_source_tier,
            "color": "#cc3333" if carbon_gco2e is None else "#26734d",
        })
        del request.app.state.dashboard_points[:-100]
    else:
        proxy_ms = (time.perf_counter() - proxy_started) * 1000
    return response