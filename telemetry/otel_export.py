from __future__ import annotations

import logging
from typing import Any

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger("greenwayai.telemetry")


def build_meter_provider() -> MeterProvider:
    resource = Resource.create({"service.name": "greenwayai"})
    provider = MeterProvider(resource=resource)
    return provider


def register_baseline_metrics() -> dict[str, Any]:
    provider = build_meter_provider()
    meter = provider.get_meter("greenwayai")

    metrics_state = {
        "request_counter": meter.create_counter("gw.requests.total", unit="1", description="Total requests processed."),
        "energy_histogram": meter.create_histogram("gw.energy.consumed_kwh", unit="kWh", description="Energy consumed per request."),
        "carbon_histogram": meter.create_histogram("gw.carbon.emitted_gco2e", unit="gCO2e", description="Carbon emitted per request."),
        "cache_hit_rate": meter.create_counter("gw.cache.hit_rate", unit="1", description="Cache hit rate placeholder."),
        "energy_saved": meter.create_counter("gw.energy.saved_kwh", unit="kWh", description="Placeholder for saved energy."),
        "carbon_avoided": meter.create_counter("gw.carbon.avoided_gco2e", unit="gCO2e", description="Placeholder for avoided carbon."),
    }

    # These placeholders stay as always-zero until later optimization phases add real cache and reuse logic.
    metrics_state["cache_hit_rate"].add(0)
    metrics_state["energy_saved"].add(0)
    metrics_state["carbon_avoided"].add(0)
    return metrics_state
