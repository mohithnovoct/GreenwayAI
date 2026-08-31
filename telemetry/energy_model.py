from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyModel:
    """Simple kWh/token model for baseline and parametric carbon accounting.

    The decode term covers all generated output tokens, including any future
    reasoning-chain tokens that are returned as part of the response body.
    """

    prefill_coeff: float = 1.5e-7
    decode_coeff: float = 4.0e-7

    def estimate_energy_kwh(self, *, input_tokens: int, output_tokens: int) -> float:
        return (self.prefill_coeff * float(input_tokens)) + (self.decode_coeff * float(output_tokens))


def estimate_energy_kwh(*, input_tokens: int, output_tokens: int, prefill_coeff: float = 1.5e-7, decode_coeff: float = 4.0e-7) -> float:
    return EnergyModel(prefill_coeff=prefill_coeff, decode_coeff=decode_coeff).estimate_energy_kwh(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
