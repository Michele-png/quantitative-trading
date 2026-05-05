"""Historical S&P 500 constituent loader.

Data source: https://github.com/fja05680/sp500
File:        sp500_ticker_start_end.csv  (one row per contiguous membership period)
Format:      ticker, start_date, end_date  (end_date empty = currently a member)

A single ticker may have multiple rows if it was added, removed, then re-added
(e.g., AAL was a member 1996-1997, removed during bankruptcy, re-added 2015).
The loader handles this correctly.

This dataset is what kills survivorship bias in our backtest: at each trade
date T we use *the actual S&P 500 members on T*, not today's list.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from quantitative_trading.config import get_config

log = logging.getLogger(__name__)


SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "sp500_ticker_start_end.csv"
)


class UniverseError(RuntimeError):
    """Raised when the universe data can't be loaded or queried."""


class SP500Universe:
    """Historical S&P 500 constituents."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        source_url: str = SOURCE_URL,
    ) -> None:
        cfg = get_config()
        self._cache_dir = cache_dir or cfg.universe_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._source_url = source_url
        self._df: pd.DataFrame | None = None

    def _cache_file(self) -> Path:
        return self._cache_dir / "sp500_ticker_start_end.csv"

    def download(self, *, force_refresh: bool = False) -> Path:
        path = self._cache_file()
        if path.exists() and not force_refresh:
            return path
        log.info("Downloading S&P 500 historical constituents from %s", self._source_url)
        resp = requests.get(self._source_url, timeout=30)
        resp.raise_for_status()
        path.write_text(resp.text)
        return path

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        path = self.download()
        df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
        if not {"ticker", "start_date", "end_date"}.issubset(df.columns):
            raise UniverseError(
                f"Universe CSV missing expected columns; got {list(df.columns)}"
            )
        df["ticker"] = df["ticker"].str.upper().str.strip()
        self._df = df
        return df

    def get_members(self, target_date: date) -> set[str]:
        """Return the set of tickers that were S&P 500 members on `target_date`.

        Membership is inclusive at both endpoints: a ticker with
        `start_date <= target_date <= end_date` (or `end_date` null) is a member.
        """
        df = self._load()
        ts = pd.Timestamp(target_date)
        in_window = (df["start_date"] <= ts) & (
            df["end_date"].isna() | (df["end_date"] >= ts)
        )
        return set(df.loc[in_window, "ticker"].unique())

    def get_membership_periods(self, ticker: str) -> list[tuple[date, date | None]]:
        """Return all (start, end) periods when `ticker` was a member.

        end is None for currently-active periods.
        """
        df = self._load()
        rows = df[df["ticker"] == ticker.upper().strip()]
        out: list[tuple[date, date | None]] = []
        for _, r in rows.iterrows():
            start = r["start_date"].date()
            end = r["end_date"].date() if pd.notna(r["end_date"]) else None
            out.append((start, end))
        return out

    def is_member(self, ticker: str, target_date: date) -> bool:
        """True if `ticker` was an S&P 500 member on `target_date`."""
        for start, end in self.get_membership_periods(ticker):
            if start <= target_date and (end is None or end >= target_date):
                return True
        return False

    def member_count(self, target_date: date) -> int:
        """Number of S&P 500 members on `target_date` (sanity check: ~500)."""
        return len(self.get_members(target_date))

    def all_tickers_ever(self) -> set[str]:
        """All tickers that have ever been in the S&P 500 (since dataset start)."""
        df = self._load()
        return set(df["ticker"].unique())
