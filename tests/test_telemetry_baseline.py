import types
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

import telemetry.nvml_sampler as nvml_sampler_module
from ingress.app import app, request_energy_window_start
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
    monkeypatch.setenv("VLLM_BACKEND_URL", "http://backend")
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
