from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger("greenwayai.carbon")

# Static regional-average MOER values are coarse defaults used when no live grid API
# is available. These are intentionally conservative approximations for dev/CI usage
# and can be replaced by region-specific values from Electricity Maps or a local
# emissions inventory when the deployment is configured for production reporting.
DEFAULT_STATIC_MOER = {
    "us-east1": 0.31,
    "us-west1": 0.20,
    "europe-west1": 0.15,
    "asia-southeast1": 0.42,
    "default": 0.28,
}

PROVIDER_PUE = {
    "google": 1.09,
    "aws": 1.15,
    "azure": 1.17,
}


def apply_infrastructure_multipliers(raw_moer_kg_per_kwh: float, provider: str) -> float:
    """Apply the infrastructure multiplier chain: power efficiency x transmission x provider PUE."""
    provider_name = provider.lower()
    pue = PROVIDER_PUE.get(provider_name, PROVIDER_PUE["google"])
    return raw_moer_kg_per_kwh * 1.04 * 1.08 * pue


class CarbonEngine:
    def __init__(self, *, region: str = "us-east1", provider: str = "google", moer_cache_ttl_seconds: int = 300) -> None:
        self.region = region
        self.provider = provider.lower()
        self.provider_pue = PROVIDER_PUE.get(self.provider, PROVIDER_PUE["google"])
        self.marginal_multiplier = 1.04 * 1.08 * self.provider_pue
        self.moer_cache_ttl_seconds = moer_cache_ttl_seconds
        self._moer_cache: float | None = None
        self._moer_cache_tier: str | None = None
        self._moer_cache_time: float | None = None
        self._refresh_lock: asyncio.Lock | None = None
        self._refresh_task: asyncio.Task[tuple[float, str]] | None = None

    @property
    def refresh_lock(self) -> asyncio.Lock:
        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        return self._refresh_lock

    def _get_static_moer(self) -> float:
        return float(DEFAULT_STATIC_MOER.get(self.region, DEFAULT_STATIC_MOER["default"]))

    def _apply_infrastructure_multipliers(self, raw_moer_kg_per_kwh: float) -> float:
        return apply_infrastructure_multipliers(raw_moer_kg_per_kwh, self.provider)

    async def _fetch_from_electricity_maps(self) -> tuple[float, str] | None:
        key = os.getenv("ELECTRICITYMAPS_API_KEY")
        if not key:
            return None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://api.electricitymap.io/v3/carbon-intensity/latest?zone={self.region}",
                    headers={"X-Bearer-Token": key},
                )
                if response.status_code != 200:
                    logger.warning("Electricity Maps returned status %s for region %s.", response.status_code, self.region)
                    return None
                payload = response.json()
                intensity = payload.get("carbonIntensity")
                if intensity is None:
                    logger.warning("Electricity Maps payload for %s missing carbonIntensity: %s", self.region, payload)
                    return None
                # Electricity Maps reports gCO2eq/kWh; convert to kgCO2eq/kWh so the
                # final carbon formula can consistently use energy_kwh * moer_kg_per_kwh * 1000.
                return float(intensity) / 1000.0, "electricitymaps"
        except Exception as exc:  # pragma: no cover - network error path validated via unit tests
            logger.warning("Electricity Maps request failed for %s: %s", self.region, exc)
            return None

    async def _fetch_from_watttime(self) -> tuple[float, str] | None:
        key = os.getenv("WATTTIME_API_KEY")
        if not key:
            return None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://api.watttime.org/v2/data?ba={self.region}",
                    headers={"Authorization": f"Bearer {key}"},
                )
                if response.status_code != 200:
                    logger.warning("WattTime returned status %s for region %s.", response.status_code, self.region)
                    return None
                payload = response.json()
                moer = payload.get("moer")
                if moer is None:
                    logger.warning("WattTime payload for %s missing moer: %s", self.region, payload)
                    return None
                return float(moer), "watttime"
        except Exception as exc:  # pragma: no cover - network error path validated via unit tests
            logger.warning("WattTime request failed for %s: %s", self.region, exc)
            return None

    async def refresh_moer_cache(self) -> tuple[float, str]:
        provider_failed = False
        for fetcher in (self._fetch_from_electricity_maps, self._fetch_from_watttime):
            try:
                result = await fetcher()
            except Exception as exc:
                provider_failed = True
                logger.warning("MOER provider failed for %s: %s", self.region, exc)
                continue
            if result is not None:
                moer, tier = result
                self._moer_cache = moer
                self._moer_cache_tier = tier
                self._moer_cache_time = time.monotonic()
                return moer, tier

        if provider_failed:
            raise RuntimeError(f"all configured MOER providers failed for {self.region}")

        fallback = self._get_static_moer()
        self._moer_cache = fallback
        self._moer_cache_tier = "static_fallback"
        self._moer_cache_time = time.monotonic()
        logger.warning("MOER fetch failed for %s; using static regional-average fallback (%s kg/kWh).", self.region, fallback)
        return fallback, "static_fallback"

    async def fetch_moer_kg_per_kwh(self) -> float:
        now = time.monotonic()
        if self._moer_cache is not None and self._moer_cache_time is not None and (now - self._moer_cache_time) < self.moer_cache_ttl_seconds:
            return self._moer_cache

        if self._refresh_task is not None and not self._refresh_task.done():
            if self._moer_cache is not None:
                return self._moer_cache
            moer, _ = await self._refresh_task
            return moer

        self._refresh_task = asyncio.create_task(self.refresh_moer_cache())
        try:
            if self._moer_cache is not None:
                return self._moer_cache
            moer, _ = await self._refresh_task
            return moer
        finally:
            if self._refresh_task.done():
                self._refresh_task = None

    async def get_moer_with_tier(self) -> tuple[float, str]:
        now = time.monotonic()
        if self._moer_cache is not None and self._moer_cache_time is not None and (now - self._moer_cache_time) < self.moer_cache_ttl_seconds:
            return self._moer_cache, self._moer_cache_tier or "static_fallback"

        if self._refresh_task is not None and not self._refresh_task.done():
            if self._moer_cache is not None:
                return self._moer_cache, self._moer_cache_tier or "static_fallback"
            return await self._refresh_task

        self._refresh_task = asyncio.create_task(self.refresh_moer_cache())
        try:
            if self._moer_cache is not None:
                return self._moer_cache, self._moer_cache_tier or "static_fallback"
            moer, tier = await self._refresh_task
            return moer, tier
        finally:
            if self._refresh_task.done():
                self._refresh_task = None

    def estimate_carbon_gco2e(self, *, energy_kwh: float, moer_kg_per_kwh: float | None = None) -> float:
        moer = moer_kg_per_kwh if moer_kg_per_kwh is not None else self._get_static_moer()
        adjusted_moer = self._apply_infrastructure_multipliers(moer)
        # The project uses energy_kwh as kWh. MOER from live APIs is expressed as
        # gCO2eq/kWh, and we convert to kgCO2eq/kWh internally before multiplying by
        # 1000 to produce final gCO2e. This keeps the unit conversion explicit and
        # auditable in one place.
        return energy_kwh * adjusted_moer * 1000.0

    async def estimate_request_carbon_gco2e(self, *, energy_kwh: float) -> tuple[float, float, str]:
        moer, tier = await self.get_moer_with_tier()
        carbon = self.estimate_carbon_gco2e(energy_kwh=energy_kwh, moer_kg_per_kwh=moer)
        return carbon, moer, tier
