"""Price data access built on yfinance."""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf


class PriceClient:
    """Small yfinance wrapper used by labels, valuation, and backtests."""

    def __init__(self) -> None:
        self._history_cache: dict[str, pd.DataFrame] = {}

    def get_history(self, ticker: str) -> pd.DataFrame:
        """Return cached adjusted price history for a ticker."""
        symbol = ticker.upper()
        if symbol not in self._history_cache:
            hist = yf.Ticker(symbol).history(period="max", auto_adjust=False)
            if not isinstance(hist.index, pd.DatetimeIndex):
                hist.index = pd.to_datetime(hist.index)
            self._history_cache[symbol] = hist.sort_index()
        return self._history_cache[symbol]

    def get_close_at(self, ticker: str, as_of: date) -> float | None:
        """Return the latest close at or before ``as_of``."""
        history = self.get_history(ticker)
        if history.empty:
            return None
        eligible = history[history.index.date <= as_of]
        if eligible.empty:
            return None
        col = "Close" if "Close" in eligible.columns else "Adj Close"
        return float(eligible[col].iloc[-1])

    def split_factor_since(self, ticker: str, since: date) -> float:
        """Return cumulative split factor after ``since``."""
        splits = yf.Ticker(ticker.upper()).splits
        if splits is None or len(splits) == 0:
            return 1.0
        if not isinstance(splits.index, pd.DatetimeIndex):
            splits.index = pd.to_datetime(splits.index)
        eligible = splits[splits.index.date > since]
        if len(eligible) == 0:
            return 1.0
        return float(eligible.prod())

    def forward_total_return_cagr(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> float | None:
        """Compute annualised adjusted-close return over a future window."""
        history = self.get_history(ticker)
        if history.empty:
            return None
        price_col = "Adj Close" if "Adj Close" in history.columns else "Close"
        after_start = history[history.index.date > start_date]
        through_end = after_start[after_start.index.date <= end_date]
        if through_end.empty:
            return None
        start_price = float(through_end[price_col].iloc[0])
        end_price = float(through_end[price_col].iloc[-1])
        if start_price <= 0 or end_price <= 0:
            return None
        years = (end_date - start_date).days / 365.25
        if years <= 0:
            return None
        return (end_price / start_price) ** (1 / years) - 1

    def was_delisted_before(self, ticker: str, end_date: date) -> bool:
        """Heuristic: last known price before requested end implies possible delisting."""
        history = self.get_history(ticker)
        if history.empty:
            return False
        return history.index.date[-1] < end_date
