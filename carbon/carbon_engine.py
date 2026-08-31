from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("greenwayai.carbon")

DEFAULT_STATIC_MOER = {
    "us-east1": 0.31,
    "us-west1": 0.20,
    "europe-west1": 0.15,
    "asia-southeast1": 0.42,
    "default": 0.28,
}


class CarbonEngine:
    def __init__(self, *, region: str = "us-east1", provider: str = "google") -> None:
        self.region = region
        self.provider = provider.lower()
        self.provider_pue = {"google": 1.09, "aws": 1.15, "azure": 1.17}.get(self.provider, 1.09)
        self.marginal_multiplier = 1.04 * 1.08 * self.provider_pue

    def _get_static_moer(self) -> float:
        return DEFAULT_STATIC_MOER.get(self.region, DEFAULT_STATIC_MOER["default"])

    async def fetch_moer_kg_per_kwh(self) -> float:
        electricity_maps_key = os.getenv("ELECTRICITYMAPS_API_KEY")
        watt_time_key = os.getenv("WATTTIME_API_KEY")

        if electricity_maps_key:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"https://api.electricitymap.io/v3/carbon-intensity/latest?zone={self.region}",
                        headers={"X-Bearer-Token": electricity_maps_key},
                    )
                    if response.status_code == 200:
                        payload = response.json()
                        intensity = payload.get("carbonIntensity")
                        if intensity is not None:
                            return float(intensity) / 1000.0
            except Exception:
                pass

        if watt_time_key:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"https://api.watttime.org/v2/data?ba={self.region}",
                        headers={"Authorization": f"Bearer {watt_time_key}"},
                    )
                    if response.status_code == 200:
                        payload = response.json()
                        moer = payload.get("moer")
                        if moer is not None:
                            return float(moer)
            except Exception:
                pass

        logger.warning("Both Electricity Maps and WattTime failed for %s; using static regional-average MOER fallback.", self.region)
        return self._get_static_moer()

    def estimate_carbon_gco2e(self, *, energy_kwh: float, moer_kg_per_kwh: float | None = None) -> float:
        moer = moer_kg_per_kwh if moer_kg_per_kwh is not None else self._get_static_moer()
        return energy_kwh * moer * 1000.0 * self.marginal_multiplier

    async def estimate_request_carbon_gco2e(self, *, energy_kwh: float) -> tuple[float, float, str]:
        moer = await self.fetch_moer_kg_per_kwh()
        carbon = self.estimate_carbon_gco2e(energy_kwh=energy_kwh, moer_kg_per_kwh=moer)
        return carbon, moer, "live"
