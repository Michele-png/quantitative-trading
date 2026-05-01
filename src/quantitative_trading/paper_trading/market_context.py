"""Market, news, and universe context for weekly decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from quantitative_trading.paper_trading.candidate_scoring import CandidateScorer
from quantitative_trading.paper_trading.models import CandidateSnapshot, MarketContext

DEFAULT_LIQUID_SP500_SCREEN: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "GOOGL": "Alphabet Class A",
    "GOOG": "Alphabet Class C",
    "AVGO": "Broadcom",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "LLY": "Eli Lilly",
    "V": "Visa",
    "UNH": "UnitedHealth",
    "XOM": "Exxon Mobil",
    "MA": "Mastercard",
    "COST": "Costco",
    "HD": "Home Depot",
    "PG": "Procter & Gamble",
    "NFLX": "Netflix",
    "BAC": "Bank of America",
    "KO": "Coca-Cola",
    "CRM": "Salesforce",
    "AMD": "Advanced Micro Devices",
    "WMT": "Walmart",
    "DIS": "Walt Disney",
}
"""Liquid S&P 500 fallback used when full grouped-market data is unavailable."""


class MarketDataError(RuntimeError):
    """Raised when required market context cannot be built."""


class SP500UniverseProvider:
    """Fetch the current S&P 500 universe from Wikipedia."""

    WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def fetch(self) -> dict[str, str]:
        """Return mapping of ticker symbol to company name."""
        tables = pd.read_html(self.WIKI_URL)
        if not tables:
            raise MarketDataError("Could not read S&P 500 table from Wikipedia.")
        table = tables[0]
        if "Symbol" not in table.columns or "Security" not in table.columns:
            raise MarketDataError("Wikipedia S&P 500 table did not contain expected columns.")
        return {
            str(row["Symbol"]).replace(".", "-").upper(): str(row["Security"])
            for _, row in table.iterrows()
        }


class PolygonMarketDataClient:
    """Small Polygon REST client for stock aggregates and news."""

    def __init__(self, api_key: str, *, base_url: str = "https://api.polygon.io") -> None:
        if not api_key:
            raise ValueError("api_key must not be empty.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def latest_grouped_daily(
        self,
        *,
        as_of: date,
        max_lookback_days: int = 7,
    ) -> tuple[date, list[dict[str, Any]]]:
        """Return the most recent grouped US stock daily aggregate at or before ``as_of``."""
        for offset in range(max_lookback_days + 1):
            target = as_of - timedelta(days=offset)
            payload = self._get(
                f"/v2/aggs/grouped/locale/us/market/stocks/{target.isoformat()}",
                {"adjusted": "true"},
            )
            results = payload.get("results") or []
            if results:
                return target, results
        raise MarketDataError("Polygon returned no grouped stock aggregates in lookback window.")

    def daily_bars(self, symbol: str, *, start: date, end: date) -> list[dict[str, Any]]:
        """Return daily adjusted OHLCV bars for ``symbol``."""
        payload = self._get(
            f"/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}",
            {"adjusted": "true", "sort": "asc", "limit": "5000"},
        )
        return payload.get("results") or []

    def news_headlines(self, symbol: str, *, since: datetime, limit: int = 3) -> list[str]:
        """Return recent Polygon news headlines for a ticker."""
        payload = self._get(
            "/v2/reference/news",
            {
                "ticker": symbol,
                "published_utc.gte": since.isoformat().replace("+00:00", "Z"),
                "order": "desc",
                "sort": "published_utc",
                "limit": str(limit),
            },
        )
        results = payload.get("results") or []
        return [str(item.get("title", "")).strip() for item in results if item.get("title")]

    @retry(
        retry=retry_if_exception_type((requests.RequestException, MarketDataError)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            params={**params, "apiKey": self.api_key},
            timeout=30,
        )
        if response.status_code >= 400:
            raise MarketDataError(
                f"Polygon request {path} failed with HTTP {response.status_code}."
            )
        payload = response.json()
        if payload.get("status") in {"ERROR", "NOT_AUTHORIZED"}:
            reason = payload.get("error") or payload.get("message")
            raise MarketDataError(f"Polygon request failed: {reason}")
        return payload


@dataclass(frozen=True)
class MarketContextBuilder:
    """Build the current-week context shown to the zero-shot LLM."""

    polygon: PolygonMarketDataClient
    universe_provider: SP500UniverseProvider
    scorer: CandidateScorer

    def build(
        self,
        *,
        trade_week: date,
        symbols: list[str] | None = None,
        preselect_count: int = 25,
        max_candidates: int = 10,
    ) -> MarketContext:
        """Build market context for a weekly decision."""
        universe = self._load_universe(symbols)
        data_notes = []
        grouped_by_symbol: dict[str, dict[str, Any]] = {}
        try_grouped = symbols is None
        if try_grouped:
            try:
                aggregate_date, grouped = self.polygon.latest_grouped_daily(as_of=trade_week)
                grouped_by_symbol = {str(item.get("T", "")).upper(): item for item in grouped}
                raw_candidates = self._build_raw_candidates(
                    universe=universe,
                    grouped_by_symbol=grouped_by_symbol,
                    aggregate_date=aggregate_date,
                    preselect_count=preselect_count,
                )
                data_notes.append(
                    "Stock aggregates are from "
                    f"{aggregate_date.isoformat()}, the latest Polygon grouped day found."
                )
            except MarketDataError:
                universe = DEFAULT_LIQUID_SP500_SCREEN
                aggregate_date = trade_week
                raw_candidates = self._build_candidates_from_bars(
                    universe=universe,
                    aggregate_date=aggregate_date,
                    preselect_count=preselect_count,
                )
                data_notes.append(
                    "Polygon grouped-market data was unavailable; used the liquid "
                    "S&P 500 fallback screen with per-ticker bars."
                )
        else:
            aggregate_date = trade_week
            raw_candidates = self._build_candidates_from_bars(
                universe=universe,
                aggregate_date=aggregate_date,
                preselect_count=preselect_count,
            )
            data_notes.append("Used caller-supplied ticker subset with per-ticker bars.")

        scored = self.scorer.score(raw_candidates)[:max_candidates]
        market_summary = self._market_summary(
            aggregate_date=aggregate_date,
            grouped_by_symbol=grouped_by_symbol,
        )
        if not grouped_by_symbol:
            market_summary = self._market_summary_from_bars(aggregate_date=aggregate_date)
        return MarketContext(
            trade_week=trade_week,
            generated_at=datetime.now(UTC),
            universe_size=len(universe),
            candidates=scored,
            market_summary=market_summary,
            data_notes=[
                *data_notes,
                "Candidate ranking is deterministic; the LLM only makes the final "
                "zero-shot selection.",
            ],
        )

    def _load_universe(self, symbols: list[str] | None) -> dict[str, str]:
        if symbols:
            return {symbol.upper(): symbol.upper() for symbol in symbols}
        try:
            return self.universe_provider.fetch()
        except Exception:  # noqa: BLE001
            return DEFAULT_LIQUID_SP500_SCREEN

    def _build_raw_candidates(
        self,
        *,
        universe: dict[str, str],
        grouped_by_symbol: dict[str, dict[str, Any]],
        aggregate_date: date,
        preselect_count: int,
    ) -> list[CandidateSnapshot]:
        liquid_rows = []
        for symbol, name in universe.items():
            row = grouped_by_symbol.get(symbol)
            if not row or not row.get("c") or not row.get("v"):
                continue
            dollar_volume = float(row["c"]) * float(row["v"])
            liquid_rows.append((symbol, name, row, dollar_volume))
        liquid_rows.sort(key=lambda item: item[3], reverse=True)

        since_news = datetime.combine(
            aggregate_date - timedelta(days=7),
            datetime.min.time(),
            tzinfo=UTC,
        )
        candidates = []
        for symbol, name, row, dollar_volume in liquid_rows[:preselect_count]:
            price = float(row["c"])
            previous_close = float(row["o"]) if row.get("o") else None
            day_return = (price / previous_close - 1) if previous_close else None
            five_day_return = self._five_day_return(
                symbol=symbol,
                aggregate_date=aggregate_date,
                price=price,
            )
            headlines = self._safe_news_headlines(symbol=symbol, since=since_news)
            candidates.append(
                CandidateSnapshot(
                    symbol=symbol,
                    name=name,
                    price=price,
                    previous_close=previous_close,
                    day_return=day_return,
                    five_day_return=five_day_return,
                    dollar_volume=dollar_volume,
                    news_headlines=headlines,
                )
            )
        if not candidates:
            raise MarketDataError("No S&P 500 candidates could be built from Polygon data.")
        return candidates

    def _build_candidates_from_bars(
        self,
        *,
        universe: dict[str, str],
        aggregate_date: date,
        preselect_count: int,
    ) -> list[CandidateSnapshot]:
        since_news = datetime.combine(
            aggregate_date - timedelta(days=7),
            datetime.min.time(),
            tzinfo=UTC,
        )
        candidates = []
        for symbol, name in list(universe.items())[:preselect_count]:
            bars = self.polygon.daily_bars(
                symbol,
                start=aggregate_date - timedelta(days=30),
                end=aggregate_date,
            )
            if len(bars) < 2:
                continue
            latest = bars[-1]
            previous = bars[-2]
            price = float(latest["c"])
            previous_close = float(previous["c"])
            day_return = price / previous_close - 1 if previous_close else None
            closes = [float(bar["c"]) for bar in bars if bar.get("c")]
            five_day_return = price / closes[-5] - 1 if len(closes) >= 5 and closes[-5] else None
            volume = float(latest.get("v", 0.0))
            candidates.append(
                CandidateSnapshot(
                    symbol=symbol,
                    name=name,
                    price=price,
                    previous_close=previous_close,
                    day_return=day_return,
                    five_day_return=five_day_return,
                    dollar_volume=price * volume,
                    news_headlines=self._safe_news_headlines(symbol=symbol, since=since_news),
                )
            )
        if not candidates:
            raise MarketDataError("No candidates could be built from per-ticker Polygon bars.")
        return candidates

    def _five_day_return(self, *, symbol: str, aggregate_date: date, price: float) -> float | None:
        bars = self.polygon.daily_bars(
            symbol,
            start=aggregate_date - timedelta(days=14),
            end=aggregate_date,
        )
        closes = [float(bar["c"]) for bar in bars if bar.get("c")]
        if len(closes) < 5 or closes[-5] == 0:
            return None
        return price / closes[-5] - 1

    def _safe_news_headlines(self, *, symbol: str, since: datetime) -> list[str]:
        try:
            return self.polygon.news_headlines(symbol, since=since, limit=3)
        except MarketDataError:
            return []

    @staticmethod
    def _market_summary(
        *,
        aggregate_date: date,
        grouped_by_symbol: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {"aggregate_date": aggregate_date.isoformat()}
        for symbol in ("SPY", "QQQ", "DIA"):
            row = grouped_by_symbol.get(symbol)
            if row and row.get("c") and row.get("o"):
                summary[symbol] = {
                    "close": float(row["c"]),
                    "day_return": float(row["c"]) / float(row["o"]) - 1,
                    "volume": float(row.get("v", 0.0)),
                }
        return summary

    def _market_summary_from_bars(self, *, aggregate_date: date) -> dict[str, Any]:
        summary: dict[str, Any] = {"aggregate_date": aggregate_date.isoformat()}
        for symbol in ("SPY", "QQQ", "DIA"):
            try:
                bars = self.polygon.daily_bars(
                    symbol,
                    start=aggregate_date - timedelta(days=7),
                    end=aggregate_date,
                )
            except MarketDataError:
                continue
            if len(bars) >= 2:
                latest = bars[-1]
                previous = bars[-2]
                summary[symbol] = {
                    "close": float(latest["c"]),
                    "day_return": float(latest["c"]) / float(previous["c"]) - 1,
                    "volume": float(latest.get("v", 0.0)),
                }
        return summary
