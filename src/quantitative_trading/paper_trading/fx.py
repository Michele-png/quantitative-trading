"""EUR/USD exchange-rate retrieval via Polygon."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class FxRateError(RuntimeError):
    """Raised when an FX rate cannot be retrieved."""


@dataclass(frozen=True)
class FxRate:
    """EUR/USD rate at a point in time."""

    as_of: date
    eur_usd: float
    source: str


class PolygonFxClient:
    """Retrieve EUR/USD rates from Polygon's forex aggregate API."""

    def __init__(self, api_key: str, *, base_url: str = "https://api.polygon.io") -> None:
        if not api_key:
            raise ValueError("api_key must not be empty.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @retry(
        retry=retry_if_exception_type((requests.RequestException, FxRateError)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def get_eur_usd(self, as_of: date | None = None) -> FxRate:
        """Return the latest available EUR/USD close up to ``as_of``."""
        target = as_of or date.today()
        start = target - timedelta(days=7)
        url = (
            f"{self.base_url}/v2/aggs/ticker/C:EURUSD/range/1/day/"
            f"{start.isoformat()}/{target.isoformat()}"
        )
        payload = self._get(url, {"adjusted": "true", "sort": "desc", "limit": "1"})
        results = payload.get("results") or []
        if not results:
            raise FxRateError("Polygon returned no EUR/USD aggregate results.")
        rate = float(results[0]["c"])
        if rate <= 0:
            raise FxRateError(f"Polygon returned invalid EUR/USD rate: {rate}.")
        return FxRate(as_of=target, eur_usd=rate, source="polygon:C:EURUSD")

    def _get(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        request_params = {**params, "apiKey": self.api_key}
        response = requests.get(url, params=request_params, timeout=30)
        if response.status_code >= 400:
            raise FxRateError(f"Polygon FX request failed with HTTP {response.status_code}.")
        payload = response.json()
        if payload.get("status") in {"ERROR", "NOT_AUTHORIZED"}:
            reason = payload.get("error") or payload.get("message")
            raise FxRateError(f"Polygon FX request failed: {reason}")
        return payload
