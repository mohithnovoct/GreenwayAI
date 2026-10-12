import httpx
import pytest

from verify.gemini_ab import estimate_gateway_tax_ms


def test_estimate_gateway_tax_uses_gateway_upstream_round_trip():
    headers = httpx.Headers({"X-Greenway-Round-Trip-ms": "12.5"})

    assert estimate_gateway_tax_ms(32.0, headers) == pytest.approx(19.5)


def test_estimate_gateway_tax_is_unavailable_without_upstream_timing():
    assert estimate_gateway_tax_ms(32.0, httpx.Headers()) is None
