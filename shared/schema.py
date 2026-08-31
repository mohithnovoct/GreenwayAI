from enum import Enum
from typing import Any

from pydantic import BaseModel


class SLAClass(str, Enum):
    INTERACTIVE = "interactive"
    DEFERRABLE = "deferrable"


class GatewayRequest(BaseModel):
    request_id: str
    raw_payload: dict[str, Any]
    normalized_prompt: str | None = None
    content_hash: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    energy_kwh: float | None = None
    carbon_gco2e: float | None = None
    telemetry_tier: str | None = None
    sla_class: SLAClass = SLAClass.INTERACTIVE