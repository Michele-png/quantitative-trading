"""yfinance price wrapper with point-in-time discipline and disk caching.

Two distinct use cases, both supported PIT-correctly:

    1. Decision-time price: at agent-evaluation date T, what was the closing
       price an investor would have observed?  Use `get_close_at(ticker, T)`.
       Only data with index <= T is consulted.

    2. Forward-return label: from trade date T0 to label horizon T1 (e.g. 5y
       later), what was the annualized total return?  Use
       `forward_total_return_cagr(ticker, T0, T1)`.  This *does* read prices
       in (T0, T1] — but the label step is conceptually separate from the
       agent's view, so the agent never sees it.

Two columns matter:
    * `Close`: as-traded price on each date. Used for the *decision*.
    * `Adj Close`: split- and dividend-adjusted (assumes dividends reinvested
       at ex-date close). Used for *total-return* computation.

Cache strategy:
    * One parquet per ticker, holds full available daily history.
    * Cache is permanent unless `force_refresh=True` is passed. This is the
      right policy for a reproducible backtesting workload — fresh data is
      irrelevant for historical analysis. Delete the parquet file manually
      if you want to force a re-fetch.

Delisting semantics:
    * If a stock has no data at the requested label end_date, the label uses
      the *last available* Adj Close as the exit price. This is the correct
      unbiased treatment for a mixed sample of acquired and bankrupt firms:
      acquisitions show up as the merger price, bankruptcies as the near-zero
      final price. Caller must accept this convention or do their own
      delisting handling.

Important caveats:
    * yfinance can silently change column conventions between versions
      (auto_adjust default flipped). We pin `auto_adjust=False` so both
      `Close` and `Adj Close` are returned.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from quantitative_trading.config import get_config

log = logging.getLogger(__name__)


class PriceError(RuntimeError):
    """Raised when price data cannot be fetched or interpreted."""


class PriceClient:
    """yfinance-backed price client. PIT-disciplined; caches per ticker on disk."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        cfg = get_config()
        self._cache_dir = cache_dir or cfg.prices_cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, ticker: str) -> Path:
        return self._cache_dir / f"{ticker.upper()}.parquet"

    def get_history(
        self,
        ticker: str,
        *,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return the full daily history (DatetimeIndex) for a ticker.

        Columns include at least: Open, High, Low, Close, Adj Close, Volume.
        Returns an empty DataFrame if the ticker has no data.

        Cache is permanent — pass `force_refresh=True` (or delete the parquet)
        to re-fetch.
        """
        path = self._cache_file(ticker)
        if not force_refresh and path.exists():
            return pd.read_parquet(path)

        log.info("Fetching yfinance history for %s", ticker)
        t = yf.Ticker(ticker)
        df = t.history(period="max", auto_adjust=False, actions=False)
        if df is None or df.empty:
            log.warning("yfinance returned empty history for %s", ticker)
            empty = pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            )
            empty.to_parquet(path)
            return empty
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        df.to_parquet(path)
        return df

    def get_close_at(
        self,
        ticker: str,
        as_of: date,
        *,
        max_lookback_days: int = 10,
    ) -> float | None:
        """Return the as-traded closing price at or before `as_of`.

        Walks backward up to `max_lookback_days` to find the last trading day
        on or before `as_of` (handles weekends/holidays). Returns None if no
        data exists in that window.
        """
        df = self.get_history(ticker)
        if df.empty:
            return None
        ts = pd.Timestamp(as_of)
        sliced = df.loc[df.index <= ts]
        if sliced.empty:
            return None
        last_date = sliced.index.max()
        if (ts - last_date).days > max_lookback_days:
            return None
        return float(sliced["Close"].iloc[-1])

    def get_adj_close_at(
        self,
        ticker: str,
        as_of: date,
        *,
        max_lookback_days: int = 10,
    ) -> float | None:
        """Return the split/dividend-adjusted closing price at or before `as_of`."""
        df = self.get_history(ticker)
        if df.empty:
            return None
        ts = pd.Timestamp(as_of)
        sliced = df.loc[df.index <= ts]
        if sliced.empty:
            return None
        last_date = sliced.index.max()
        if (ts - last_date).days > max_lookback_days:
            return None
        return float(sliced["Adj Close"].iloc[-1])

    def forward_total_return_cagr(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> float | None:
        """Annualized total return between two dates, using Adj Close.

        Delisting handling: if the ticker has no data at `end_date`, the
        *last available* Adj Close is used as the exit price. This is the
        unbiased convention (acquisitions show up as merger price, bankruptcies
        as the near-zero final price). The annualization period is always the
        nominal `end_date - start_date`, never the actual data span — so a
        stock that delisted halfway through the window has its return spread
        over the full horizon, correctly penalizing it relative to survivors.

        Returns None if there's no data at or before `start_date`.
        """
        df = self.get_history(ticker)
        if df.empty:
            return None

        ts_start = pd.Timestamp(start_date)
        ts_end = pd.Timestamp(end_date)

        start_slice = df.loc[df.index <= ts_start]
        if start_slice.empty:
            return None
        end_slice = df.loc[df.index <= ts_end]
        if end_slice.empty:
            return None

        p0 = float(start_slice["Adj Close"].iloc[-1])
        p1 = float(end_slice["Adj Close"].iloc[-1])
        if p0 <= 0:
            return None

        years = (end_date - start_date).days / 365.25
        if years <= 0:
            return None

        return (p1 / p0) ** (1.0 / years) - 1.0

    def was_delisted_before(
        self,
        ticker: str,
        as_of: date,
        *,
        gap_days: int = 30,
    ) -> bool:
        """True if the ticker has a final price more than `gap_days` before `as_of`.

        Useful for flagging label rows where the forward CAGR was computed on
        partial data (the stock stopped trading before the label horizon).
        """
        df = self.get_history(ticker)
        if df.empty:
            return False
        last = df.index.max()
        return (pd.Timestamp(as_of) - last).days > gap_days

    def get_splits(
        self,
        ticker: str,
        *,
        force_refresh: bool = False,
    ) -> pd.Series:
        """Return a Series of stock-split ratios indexed by ex-date.

        Cached as a parquet alongside the price file. A "2.0" entry on a date
        means every 1 share before that date became 2 shares on that date.
        """
        path = self._cache_dir / f"{ticker.upper()}.splits.parquet"
        if not force_refresh and path.exists():
            df = pd.read_parquet(path)
            if df.empty:
                return pd.Series(dtype=float)
            return df.iloc[:, 0]

        log.info("Fetching yfinance splits for %s", ticker)
        t = yf.Ticker(ticker)
        splits = t.splits
        if splits is None or splits.empty:
            empty = pd.DataFrame({"ratio": []})
            empty.to_parquet(path)
            return pd.Series(dtype=float)
        splits = splits.copy()
        splits.index = pd.DatetimeIndex(splits.index).tz_localize(None).normalize()
        splits.name = "ratio"
        splits.to_frame().to_parquet(path)
        return splits

    def split_factor_since(self, ticker: str, since_date: date) -> float:
        """Cumulative split factor for splits with ex-date strictly after `since_date`.

        Use case: SEC reports per-share quantities (EPS, dividends/share, share
        count) in the share basis as of the filing date. yfinance returns
        prices retroactively split-adjusted to today's basis. To compare them
        meaningfully you must scale the SEC per-share quantity:

            eps_today_basis = eps_as_filed / split_factor_since(filed_date)

        Returns 1.0 if no splits occurred after since_date.
        """
        splits = self.get_splits(ticker)
        if splits.empty:
            return 1.0
        ts = pd.Timestamp(since_date)
        relevant = splits.loc[splits.index > ts]
        if relevant.empty:
            return 1.0
        return float(relevant.prod())

    def coverage(self, ticker: str) -> tuple[date | None, date | None]:
        """Return (first_date, last_date) of available price data, or (None, None)."""
        df = self.get_history(ticker)
        if df.empty:
            return None, None
        return df.index.min().date(), df.index.max().date()
