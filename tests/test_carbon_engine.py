import asyncio

import pytest

from carbon.carbon_engine import CarbonEngine, apply_infrastructure_multipliers


@pytest.mark.parametrize(
    ("energy_kwh", "moer_kg_per_kwh", "provider", "expected"),
    [
        (1.0, 0.25, "google", 1.0 * 0.25 * 1.04 * 1.08 * 1.09 * 1000.0),
        (2.5, 0.15, "aws", 2.5 * 0.15 * 1.04 * 1.08 * 1.15 * 1000.0),
    ],
)
def test_estimate_carbon_gco2e_matches_hand_calculation(energy_kwh, moer_kg_per_kwh, provider, expected):
    engine = CarbonEngine(region="us-east1", provider=provider)
    assert engine.estimate_carbon_gco2e(energy_kwh=energy_kwh, moer_kg_per_kwh=moer_kg_per_kwh) == pytest.approx(expected)


def test_apply_infrastructure_multipliers_chain_matches_exact_value():
    raw = 0.31
    provider = "google"
    expected = raw * 1.04 * 1.08 * 1.09
    assert apply_infrastructure_multipliers(raw, provider) == pytest.approx(expected)


def test_carbon_engine_prefers_electricitymaps_primary(monkeypatch):
    engine = CarbonEngine(region="us-east1", provider="google")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"carbonIntensity": 400}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setenv("ELECTRICITYMAPS_API_KEY", "abc")
    monkeypatch.delenv("WATTTIME_API_KEY", raising=False)
    monkeypatch.setattr("carbon.carbon_engine.httpx.AsyncClient", lambda timeout=10.0: FakeClient())

    value, tier = asyncio.run(engine.refresh_moer_cache())
    assert tier == "electricitymaps"
    assert value == pytest.approx(0.4)


def test_carbon_engine_falls_back_to_watttime(monkeypatch):
    engine = CarbonEngine(region="us-east1", provider="google")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"moer": 0.22}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.delenv("ELECTRICITYMAPS_API_KEY", raising=False)
    monkeypatch.setenv("WATTTIME_API_KEY", "abc")
    monkeypatch.setattr("carbon.carbon_engine.httpx.AsyncClient", lambda timeout=10.0: FakeClient())

    value, tier = asyncio.run(engine.refresh_moer_cache())
    assert tier == "watttime"
    assert value == pytest.approx(0.22)


def test_carbon_engine_uses_static_default_when_no_live_keys(monkeypatch):
    engine = CarbonEngine(region="us-east1", provider="google")
    monkeypatch.delenv("ELECTRICITYMAPS_API_KEY", raising=False)
    monkeypatch.delenv("WATTTIME_API_KEY", raising=False)

    value, tier = asyncio.run(engine.refresh_moer_cache())
    assert tier == "static_fallback"
    assert value == pytest.approx(0.31)
