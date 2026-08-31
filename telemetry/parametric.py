from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("greenwayai.telemetry")

VALID_TIERS = {"vendor", "lca", "throughput_fallback"}


@dataclass(frozen=True)
class ParametricEstimate:
    energy_kwh: float
    energy_source_tier: str


class ParametricEnergyEstimator:
    """Cached, configuration-driven energy estimates for non-local models."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path is not None else Path(__file__).with_name("parametric_coefficients.json")
        with path.open(encoding="utf-8") as config_file:
            self._config = json.load(config_file)
        self._fallback = self._validate_coefficients(self._config["throughput_fallback"], "throughput_fallback")

    @staticmethod
    def _validate_coefficients(coefficients: dict[str, object], tier: str) -> tuple[float, float]:
        if tier not in VALID_TIERS:
            raise ValueError(f"Unsupported parametric energy tier: {tier}")
        return (
            float(coefficients["prefill_kwh_per_input_token"]),
            float(coefficients["decode_kwh_per_output_token"]),
        )

    def estimate(self, *, model: str, input_tokens: int, output_tokens: int) -> ParametricEstimate:
        configured = self._config.get("models", {}).get(model)
        if configured is None:
            logger.warning(
                "Parametric energy model %s has no configured coefficients; using low-confidence throughput fallback.",
                model,
            )
            tier = "throughput_fallback"
            prefill_coeff, decode_coeff = self._fallback
        else:
            tier = str(configured["tier"])
            prefill_coeff, decode_coeff = self._validate_coefficients(configured, tier)

        energy_kwh = (prefill_coeff * float(input_tokens)) + (decode_coeff * float(output_tokens))
        return ParametricEstimate(energy_kwh=energy_kwh, energy_source_tier=tier)
