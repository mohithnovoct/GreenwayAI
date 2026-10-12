import types
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

import telemetry.nvml_sampler as nvml_sampler_module
from ingress.app import app, is_remote_backend_url, request_energy_window_start
from ingress.mock_backend import mock_response
from telemetry.energy_model import EnergyModel
from telemetry.nvml_sampler import NVMLPowerSampler


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "prefill_coeff", "decode_coeff", "expected"),
    [
        (1000, 200, 1.5e-7, 4.0e-7, 0.00023),
        (500, 150, 2.0e-7, 3.0e-7, 0.000145),
    ],
)
def test_energy_model_matches_hand_calculation(input_tokens, output_tokens, prefill_coeff, decode_coeff, expected):
    model = EnergyModel(prefill_coeff=prefill_coeff, decode_coeff=decode_coeff)
    assert model.estimate_energy_kwh(input_tokens=input_tokens, output_tokens=output_tokens) == pytest.approx(expected)


def test_nvml_sampler_integration_matches_manual_window():
    sampler = NVMLPowerSampler()
    request_id = "req-42"
    sampler.register_request(request_id, start_ts=0.0, end_ts=2.5)
    sampler.add_sample(0.0, 100.0)
    sampler.add_sample(0.5, 200.0)
    sampler.add_sample(1.5, 300.0)
    sampler.add_sample(2.0, 150.0)

    energy = sampler.estimate_request_energy_kwh(request_id)

    # Sample windows: [0.0, 0.5) 100W, [0.5, 1.5) 200W, [1.5, 2.0) 300W, [2.0, 2.5) 150W.
    expected = ((100.0 * 0.5) + (200.0 * 1.0) + (300.0 * 0.5) + (150.0 * 0.5)) / 3_600_000
    assert energy == pytest.approx(expected)


def test_request_energy_window_start_excludes_pool_wait():
    assert request_energy_window_start(trace_started=10.0, pool_acquire_ms=250.0) == pytest.approx(10.25)


@pytest.mark.parametrize(
    ("url", "expected_remote"),
    [
        ("http://localhost:8001/v1/chat/completions", False),
        ("http://127.0.0.1:8001/v1/chat/completions", False),
        ("http://mock-backend:8001/v1/chat/completions", False),
        ("https://api.example.com/v1/chat/completions", True),
    ],
)
def test_backend_url_remote_classification_uses_host(url, expected_remote):
    assert is_remote_backend_url(url) is expected_remote


def test_app_lifecycle_creates_nvml_sampler():
    with TestClient(app) as client:
        assert hasattr(app.state, "nvml_sampler")
        assert isinstance(app.state.nvml_sampler, NVMLPowerSampler)


def test_nvml_sampler_ignores_queue_delay_for_identical_tokens():
    sampler = NVMLPowerSampler()
    for ts in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75):
        sampler.add_sample(ts, 100.0)

    sampler.register_request("queued-fast", start_ts=0.0, end_ts=0.5)
    sampler.register_request("queued-slow", start_ts=1.0, end_ts=1.5)

    fast_energy = sampler.estimate_request_energy_kwh("queued-fast")
    slow_energy = sampler.estimate_request_energy_kwh("queued-slow")
    assert fast_energy == pytest.approx(slow_energy)


def test_nvml_sampler_shares_overlapping_power_between_two_requests():
    sampler = NVMLPowerSampler()
    sampler.register_request("one", start_ts=0.0, end_ts=0.5)
    sampler.register_request("two", start_ts=0.25, end_ts=0.75)
    sampler.add_sample(0.0, 100.0)
    sampler.add_sample(1.0, 100.0)

    first = sampler.estimate_request_energy_kwh("one")
    second = sampler.estimate_request_energy_kwh("two")
    equivalent_total_window = 100.0 * 0.75 / 3_600_000.0

    assert first + second == pytest.approx(equivalent_total_window)


def test_nvml_sampler_shares_overlapping_power_across_three_requests():
    sampler = NVMLPowerSampler()
    for request_id in ("one", "two", "three"):
        sampler.register_request(request_id, start_ts=0.0, end_ts=0.75)
    sampler.add_sample(0.0, 120.0)
    sampler.add_sample(1.0, 120.0)

    energies = [sampler.estimate_request_energy_kwh(request_id) for request_id in ("one", "two", "three")]
    expected_each = 120.0 * 0.75 / 3 / 3_600_000.0

    assert energies == pytest.approx([expected_each] * 3)


def test_nvml_sampler_prunes_completed_request_registry_after_burst():
    sampler = NVMLPowerSampler()
    for index in range(1000):
        request_id = f"burst-{index}"
        sampler.register_request(request_id, start_ts=0.0, end_ts=0.01)
        sampler.add_sample(0.0, 100.0)
        sampler.add_sample(1.0, 100.0)
        sampler.complete_request(request_id, end_ts=0.01)
        sampler.estimate_request_energy_kwh(request_id)

    assert len(sampler.get_inflight_requests()) == 0
    assert len(sampler._samples) == 0


def test_nvml_sampler_falls_back_to_mock_mode_when_library_missing(monkeypatch):
    class FakeNVMLErrorLibraryNotFound(Exception):
        pass

    class FakePynvml:
        NVMLError_LibraryNotFound = FakeNVMLErrorLibraryNotFound

        @staticmethod
        def nvmlInit():
            raise FakeNVMLErrorLibraryNotFound("missing lib")

    monkeypatch.setattr(nvml_sampler_module, "pynvml", FakePynvml)
    sampler = NVMLPowerSampler()
    sampler.start()

    assert sampler.telemetry_source == "mock_nvml"
    assert sampler._mock_mode is True
    assert sampler._mock_sample_watts() > 0.0
    sampler.stop()

    response = TestClient(app).get("/health")
    assert response.status_code == 200


def test_gateway_fails_open_when_all_carbon_providers_raise(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("VLLM_BACKEND_URL", "")
    with TestClient(app) as client:
        engine = app.state.carbon_engine
        engine._moer_cache = None
        engine._moer_cache_time = None
        engine._refresh_task = None

        async def raise_outage():
            raise RuntimeError("forced provider outage")

        monkeypatch.setattr(engine, "_fetch_from_electricity_maps", raise_outage)
        monkeypatch.setattr(engine, "_fetch_from_watttime", raise_outage)
        response = client.post("/v1/chat/completions", json={"model": "mock", "messages": [{"role": "user", "content": "hello"}]})

    assert response.status_code == 200
    assert response.headers["X-Greenway-Carbon-gCO2e"] == "null"
    assert response.headers["X-Greenway-Carbon-Source"] == "unavailable"
    records = [record.message for record in caplog.records if "carbon_gco2e" in record.message]
    assert records
    assert json.loads(records[-1])["carbon_gco2e"] is None


def test_gateway_fails_open_when_nvml_raises_mid_request(monkeypatch):
    monkeypatch.setattr("ingress.app.backend_configuration", lambda: ("http://backend/v1/chat/completions", "", {}, False))
    monkeypatch.setenv("GREENWAY_ENERGY_MODE", "nvml")
    with TestClient(app) as client:
        async def send(request, **kwargs):
            return httpx.Response(200, json=mock_response({"model": "mock", "messages": [{"content": "hello"}]}), request=request)

        monkeypatch.setattr(app.state.http_client, "send", send)
        monkeypatch.setattr(app.state.nvml_sampler, "estimate_request_energy_kwh", lambda request_id: (_ for _ in ()).throw(RuntimeError("forced NVML outage")))
        response = client.post("/v1/chat/completions", json={"model": "mock", "messages": [{"role": "user", "content": "hello"}]})

    assert response.status_code == 200
    assert response.headers["X-Greenway-Energy-kWh"] == "null"
    assert response.headers["X-Greenway-Carbon-gCO2e"] == "null"
    assert response.headers["X-Greenway-Energy-Source"] == "unavailable"


def test_nvml_energy_is_hardware_only(monkeypatch):
    monkeypatch.setattr("ingress.app.backend_configuration", lambda: ("http://localhost/v1/chat/completions", "", {}, False))
    monkeypatch.setenv("GREENWAY_ENERGY_MODE", "nvml")
    monkeypatch.setenv("GREENWAY_CARBON_DISABLED", "true")

    with TestClient(app) as client:
        async def send(request, **kwargs):
            response = mock_response({"model": "mock", "messages": [{"content": "hello"}]})
            response["usage"] = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            return httpx.Response(200, json=response, request=request)

        monkeypatch.setattr(app.state.http_client, "send", send)
        monkeypatch.setattr(app.state.nvml_sampler, "estimate_request_energy_kwh", lambda request_id: 0.000123)
        response = client.post("/v1/chat/completions", json={"model": "mock", "messages": [{"role": "user", "content": "hello"}]})

    assert response.status_code == 200
    assert response.headers["X-Greenway-Energy-kWh"] == "0.000123"
    assert response.headers["X-Greenway-Energy-Source"] == "nvml"


@pytest.mark.parametrize("energy_mode", ["nvml", "parametric"])
def test_mock_backend_reports_energy_as_unavailable(monkeypatch, caplog, energy_mode):
    monkeypatch.setenv("GREENWAY_BACKEND_MODE", "mock")
    monkeypatch.setenv("VLLM_BACKEND_URL", "")
    monkeypatch.setenv("GREENWAY_ENERGY_MODE", energy_mode)
    monkeypatch.setenv("GREENWAY_CARBON_DISABLED", "true")
    caplog.set_level(logging.INFO, logger="greenwayai.ingress")

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "mock", "messages": [{"role": "user", "content": "mock request"}]},
        )
        dashboard_response = client.get("/dashboard-data")

    assert response.status_code == 200
    assert response.headers["X-Greenway-Energy-kWh"] == "null"
    assert response.headers["X-Greenway-Carbon-gCO2e"] == "null"
    assert response.headers["X-Greenway-Telemetry-Source"] == "mock_nvml"
    assert response.headers["X-Greenway-Energy-Source"] == "unavailable"
    assert response.headers["X-Greenway-Token-Source"] == "tiktoken_estimated"
    assert int(response.headers["X-Greenway-Input-Tokens"]) > 0
    assert int(response.headers["X-Greenway-Output-Tokens"]) > 0
    assert response.headers["X-Greenway-Carbon-Source"] == "static_fallback"
    records = [
        json.loads(record.message)
        for record in caplog.records
        if '"prompt_optimization_arm"' in record.message
    ]
    assert records[-1]["energy_kwh"] is None
    assert records[-1]["energy_source_tier"] == "unavailable"
    assert records[-1]["input_tokens"] > 0
    assert records[-1]["output_tokens"] > 0
    assert records[-1]["token_count_source"] == "tiktoken_estimated"
    assert records[-1]["carbon_gco2e"] is None
    assert records[-1]["carbon_source_tier"] == "static_fallback"

    assert dashboard_response.status_code == 200
    assert dashboard_response.json()[-1]["carbon_gco2e"] is None
    assert dashboard_response.json()[-1]["carbon_source_tier"] == "static_fallback"


def test_remote_backend_forces_parametric_energy(monkeypatch):
    monkeypatch.setenv("GREENWAY_BACKEND_MODE", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("GREENWAY_ENERGY_MODE", raising=False)
    monkeypatch.setenv("GREENWAY_CARBON_DISABLED", "true")
    payload = {
        "model": "gemini-2.5-flash",
        "choices": [{"message": {"role": "assistant", "content": "answer"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }

    with TestClient(app) as client:
        async def send(request, **kwargs):
            return httpx.Response(200, json=payload, request=request)

        monkeypatch.setattr(app.state.http_client, "send", send)
        response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})

    assert response.status_code == 200
    assert response.headers["X-Greenway-Telemetry-Source"] == "parametric_estimated"
    assert response.headers["X-Greenway-Energy-Source"] == "lca"


def test_gemini_backend_uses_api_usage_and_bearer_auth(monkeypatch):
    monkeypatch.setenv("GREENWAY_BACKEND_MODE", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GREENWAY_CARBON_DISABLED", "true")
    captured = {}
    payload = {
        "id": "chatcmpl-real-shaped",
        "object": "chat.completion",
        "model": "gemini-2.5-flash",
        "choices": [{"message": {"role": "assistant", "content": "A real-shaped answer."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 41, "completion_tokens": 7, "total_tokens": 48},
    }

    with TestClient(app) as client:
        async def send(request, **kwargs):
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers["authorization"]
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=payload, request=request)

        monkeypatch.setattr(app.state.http_client, "send", send)
        response = client.post(
            "/v1/chat/completions",
            json={"model": "ignored", "messages": [{"role": "user", "content": "A short test prompt."}]},
        )

    assert response.status_code == 200
    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert captured["authorization"] == "Bearer test-gemini-key"
    assert captured["body"]["model"] == "gemini-2.5-flash"
    assert response.headers["X-Greenway-Input-Tokens"] == "41"
    assert response.headers["X-Greenway-Output-Tokens"] == "7"
    assert response.headers["X-Greenway-Token-Source"] == "provider"


@pytest.mark.parametrize(
    ("usage", "expected_visible", "expected_hidden", "expected_decode", "expected_energy"),
    [
        (
            {"prompt_tokens": 155, "completion_tokens": 9, "total_tokens": 407},
            9,
            243,
            252,
            2.229e-4,
        ),
        (
            {"prompt_tokens": 155, "completion_tokens": 512, "total_tokens": 667},
            512,
            0,
            512,
            4.049e-4,
        ),
    ],
)
def test_gemini_hidden_tokens_are_included_in_parametric_energy(
    monkeypatch,
    caplog,
    usage,
    expected_visible,
    expected_hidden,
    expected_decode,
    expected_energy,
):
    monkeypatch.setenv("GREENWAY_BACKEND_MODE", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GREENWAY_CARBON_DISABLED", "true")
    caplog.set_level(logging.INFO)
    payload = {
        "model": "gemini-2.5-flash",
        "choices": [{"message": {"role": "assistant", "content": "Visible answer."}, "finish_reason": "length"}],
        "usage": usage,
    }

    with TestClient(app) as client:
        async def send(request, **kwargs):
            return httpx.Response(200, json=payload, request=request)

        monkeypatch.setattr(app.state.http_client, "send", send)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "A short test prompt."}]},
        )

    assert response.status_code == 200
    assert response.headers["X-Greenway-Output-Tokens"] == str(expected_visible)
    assert response.headers["X-Greenway-Token-Source"] == "provider"
    records = [
        json.loads(record.message)
        for record in caplog.records
        if '"prompt_optimization_arm"' in record.message
    ]
    record = records[-1]
    assert record["output_tokens_visible"] == expected_visible
    assert record["output_tokens_hidden"] == expected_hidden
    assert record["decode_tokens"] == expected_decode
    assert record["token_count_source"] == "provider"
    assert record["energy_kwh"] == pytest.approx(expected_energy)


def test_prompt_bodies_are_not_logged_by_default(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_BACKEND_URL", "")
    monkeypatch.delenv("GREENWAY_LOG_PROMPTS", raising=False)
    caplog.set_level(logging.INFO)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "mock", "messages": [{"role": "user", "content": "private prompt marker"}]},
        )

    assert response.status_code == 200
    records = [record.message for record in caplog.records if '"prompt_optimization_arm"' in record.message]
    assert records
    assert "private prompt marker" not in records[-1]
