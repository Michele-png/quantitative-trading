"""Alpaca paper-trading adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from quantitative_trading.paper_trading.models import AccountSnapshot, OrderResult


class BrokerError(RuntimeError):
    """Raised when Alpaca rejects or cannot service a broker request."""


@dataclass(frozen=True)
class PositionSnapshot:
    """Current Alpaca position metadata."""

    symbol: str
    qty: float
    market_value: float
    raw: dict[str, Any]


class AlpacaPaperBroker:
    """Minimal paper-only Alpaca Trading API client."""

    def __init__(
        self,
        *,
        api_key_id: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        if base_url.rstrip("/") != "https://paper-api.alpaca.markets":
            raise BrokerError("AlpacaPaperBroker refuses non-paper base URLs.")
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "APCA-API-KEY-ID": api_key_id,
            "APCA-API-SECRET-KEY": secret_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def get_account(self) -> AccountSnapshot:
        """Fetch account state from Alpaca."""
        payload = self._request("GET", "/v2/account")
        return AccountSnapshot(
            currency=str(payload.get("currency", "")),
            cash=float(payload.get("cash", 0.0)),
            buying_power=float(payload.get("buying_power", 0.0)),
            portfolio_value=float(payload.get("portfolio_value", 0.0)),
            trading_blocked=bool(payload.get("trading_blocked")),
        )

    def get_clock(self) -> dict[str, Any]:
        """Return Alpaca market-clock metadata."""
        return self._request("GET", "/v2/clock")

    def is_market_open(self) -> bool:
        """Return whether Alpaca says the market is currently open."""
        return bool(self.get_clock().get("is_open"))

    def list_positions(self) -> list[PositionSnapshot]:
        """List open positions."""
        payload = self._request("GET", "/v2/positions")
        return [
            PositionSnapshot(
                symbol=str(position.get("symbol", "")),
                qty=float(position.get("qty", 0.0)),
                market_value=float(position.get("market_value", 0.0)),
                raw=position,
            )
            for position in payload
        ]

    def close_all_positions(self) -> list[dict[str, Any]]:
        """Close all positions before opening the new weekly trade."""
        payload = self._request("DELETE", "/v2/positions", params={"cancel_orders": "true"})
        if isinstance(payload, list):
            return payload
        return [payload]

    def submit_notional_market_buy(self, *, symbol: str, notional_usd: float) -> OrderResult:
        """Submit a day market buy order using Alpaca fractional notional sizing."""
        if notional_usd <= 0:
            raise BrokerError("notional_usd must be positive.")
        payload = self._request(
            "POST",
            "/v2/orders",
            json_body={
                "symbol": symbol.upper(),
                "notional": f"{notional_usd:.2f}",
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
            },
        )
        return OrderResult(
            order_id=str(payload.get("id", "")),
            symbol=str(payload.get("symbol", symbol.upper())),
            side=str(payload.get("side", "buy")),
            status=str(payload.get("status", "")),
            submitted_at=payload.get("submitted_at"),
            raw=payload,
        )

    @retry(
        retry=retry_if_exception_type((requests.RequestException, BrokerError)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            headers=self._headers,
            params=params,
            json=json_body,
            timeout=30,
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            raise BrokerError(
                f"Alpaca {method} {path} failed with HTTP {response.status_code}: {detail}"
            )
        if not response.text:
            return {}
        return response.json()
