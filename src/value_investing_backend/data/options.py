"""yfinance options-chain wrapper with Black-Scholes fallback.

Mirrors `PriceClient` in style but for option chains. Two use cases:

    1. Live chain lookup at decision time — what strikes are available
       for ticker T at expiry E, and what's the mid premium?
    2. Pricing fallback — for far-OTM strikes or illiquid tickers the
       chain has no real bid/ask. Black-Scholes gives a reasonable
       theoretical premium from spot, IV proxy, time to expiry, strike.

Notes
-----
* Yahoo's public chain endpoint is unauthenticated and returns calls +
  puts for one expiration at a time. yfinance's `Ticker.option_chain`
  is a thin wrapper around it.
* This module does **not** cache to disk by default. Options data is
  intraday-volatile and the dashboard uses its own /api/yf proxy with
  short revalidation. The class accepts an optional in-process cache
  for callers that batch-fetch (e.g. ETL).
* Strikes are returned in the underlying's currency (USD for US
  tickers). The Black-Scholes formulas are currency-agnostic.

This module is intentionally framework-light: pandas DataFrames in,
floats / DataFrames out. No SQL, no network state. The dashboard does
not import this module directly (Vercel runs Node, not Python) — but
keeping parity with `PriceClient` makes the eventual ETL trivial.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import pandas as pd

log = logging.getLogger(__name__)

Kind = Literal["call", "put"]
Prefer = Literal["below", "above", "closest"]


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun 26.2.17 — same approximation
    the dashboard uses client-side so the two stay in lockstep)."""
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911
    sign = -1.0 if x < 0 else 1.0
    ax = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-ax * ax)
    return 0.5 * (1.0 + sign * y)


def bs_premium(
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    r: float,
    kind: Kind,
) -> float:
    """Black-Scholes premium for a European option.

    Parameters
    ----------
    spot : float       Underlying price.
    strike : float     Strike price.
    t_years : float    Time to expiry in years.
    iv : float         Annualized implied volatility (decimal, 0.32 = 32%).
    r : float          Risk-free rate (decimal, 0.04 = 4%).
    kind : "call" or "put"

    Returns 0.0 for any non-positive input.
    """
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return 0.0
    sig_sqrt_t = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / sig_sqrt_t
    d2 = d1 - sig_sqrt_t
    if kind == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
    return strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    r: float,
    kind: Kind,
) -> float:
    """Black-Scholes delta. Positive for calls (0..1), negative for puts
    (-1..0). ``abs(delta)`` is a decent prob-ITM approximation under BS."""
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return 0.0
    sig_sqrt_t = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / sig_sqrt_t
    if kind == "call":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


# Default risk-free rate used when the caller doesn't pass one. ~4% is a
# reasonable mid-2024..2026 short rate; premiums at 30-60 DTE move only
# marginally as r drifts ±100bp.
DEFAULT_R = 0.04


# ---------------------------------------------------------------------------
# Options chain wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainQuote:
    """Single contract from a chain. Mid is computed from bid/ask when
    both sides are >0; otherwise falls back to `last`."""

    kind: Kind
    expiry: date
    strike: float
    bid: float | None
    ask: float | None
    last: float | None
    mid: float | None
    iv: float | None
    volume: int | None
    open_interest: int | None


class OptionsError(RuntimeError):
    """Raised when an option chain cannot be fetched or interpreted."""


class OptionsClient:
    """yfinance-backed options client.

    Two methods do the heavy lifting:

        list_expiries(ticker) -> list[date]
        get_chain(ticker, expiry) -> pd.DataFrame   # one expiry, both kinds

    Plus convenience selectors (`nearest_strike`, `pick`) and a BS
    fallback (`fallback_premium`) for the case where the chain has no
    bid for the target strike.
    """

    def __init__(self) -> None:
        # Lazy import so unit tests can `monkeypatch.setattr` without
        # needing yfinance installed at import time.
        import yfinance as yf  # noqa: PLC0415  (intentional lazy import)

        self._yf = yf

    # ------------------------------------------------------------------
    # Chain fetchers
    # ------------------------------------------------------------------

    def list_expiries(self, ticker: str) -> list[date]:
        """Return all available expirations Yahoo lists for the ticker.

        Empty list when there are no listed options (very common for
        small-caps and non-US tickers).
        """
        t = self._yf.Ticker(ticker)
        raw = getattr(t, "options", None) or ()
        out: list[date] = []
        for s in raw:
            try:
                out.append(datetime.strptime(s, "%Y-%m-%d").date())
            except ValueError:
                log.warning("Skipping unparseable expiry %r for %s", s, ticker)
        return out

    def get_chain(
        self,
        ticker: str,
        expiry: date,
    ) -> pd.DataFrame:
        """Return the unified calls+puts DataFrame for one expiry.

        Columns: ``kind, expiry, strike, bid, ask, last, mid, iv,
        volume, open_interest``. Empty DataFrame if the chain is
        unavailable (yfinance raises) — caller decides whether to fall
        back to BS or surface "no data".
        """
        t = self._yf.Ticker(ticker)
        try:
            chain = t.option_chain(expiry.strftime("%Y-%m-%d"))
        except Exception as exc:  # noqa: BLE001 (yfinance throws many)
            log.warning("option_chain(%s, %s) failed: %s", ticker, expiry, exc)
            return _empty_chain_df()

        calls = _normalize(chain.calls, "call", expiry)
        puts = _normalize(chain.puts, "put", expiry)
        return pd.concat([calls, puts], ignore_index=True)

    # ------------------------------------------------------------------
    # Strike selection
    # ------------------------------------------------------------------

    @staticmethod
    def nearest_strike(
        strikes: list[float],
        target: float,
        *,
        prefer: Prefer = "closest",
    ) -> float | None:
        """Pick the strike from `strikes` closest to `target`.

        - ``prefer="below"`` constrains to strikes <= target (e.g.
          cash-secured put at strike ≤ MoS).
        - ``prefer="above"`` constrains to strikes >= target (e.g.
          covered call at strike ≥ Sticker).
        - ``prefer="closest"`` ignores side.

        Returns None if no strike satisfies the constraint.
        """
        if not strikes:
            return None
        if prefer == "below":
            eligible = [s for s in strikes if s <= target]
        elif prefer == "above":
            eligible = [s for s in strikes if s >= target]
        else:
            eligible = list(strikes)
        if not eligible:
            return None
        return min(eligible, key=lambda s: abs(s - target))

    def pick(
        self,
        chain: pd.DataFrame,
        *,
        kind: Kind,
        target_strike: float,
        prefer: Prefer = "closest",
    ) -> ChainQuote | None:
        """Pick one ChainQuote from a chain DataFrame.

        Returns the strike on the selected side nearest to
        ``target_strike``. Returns None if the chain has no rows of
        the requested kind.
        """
        side = chain[chain["kind"] == kind]
        if side.empty:
            return None
        strike = self.nearest_strike(
            side["strike"].tolist(),
            target_strike,
            prefer=prefer,
        )
        if strike is None:
            return None
        row = side.loc[side["strike"] == strike].iloc[0]
        expiry_val = row["expiry"]
        expiry_date = (
            expiry_val if isinstance(expiry_val, date) else _coerce_date(expiry_val)
        )
        return ChainQuote(
            kind=kind,
            expiry=expiry_date,
            strike=float(strike),
            bid=_maybe_float(row["bid"]),
            ask=_maybe_float(row["ask"]),
            last=_maybe_float(row["last"]),
            mid=_maybe_float(row["mid"]),
            iv=_maybe_float(row["iv"]),
            volume=_maybe_int(row["volume"]),
            open_interest=_maybe_int(row["open_interest"]),
        )

    # ------------------------------------------------------------------
    # BS fallback
    # ------------------------------------------------------------------

    @staticmethod
    def fallback_premium(
        spot: float,
        strike: float,
        as_of: date,
        expiry: date,
        iv: float,
        kind: Kind,
        *,
        r: float = DEFAULT_R,
    ) -> float:
        """Theoretical premium when no bid/ask is available.

        Computes years-to-expiry from (expiry - as_of) and dispatches
        to :func:`bs_premium`. Returns 0.0 if expiry is on/before
        ``as_of``.
        """
        days = (expiry - as_of).days
        if days <= 0:
            return 0.0
        t_years = days / 365.25
        return bs_premium(spot, strike, t_years, iv, r, kind)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(df: pd.DataFrame | None, kind: Kind, expiry: date) -> pd.DataFrame:
    """Coerce yfinance's chain DataFrame to the unified schema we expose.

    yfinance returns columns: contractSymbol, strike, lastPrice, bid,
    ask, change, percentChange, volume, openInterest, impliedVolatility,
    inTheMoney, contractSize, currency. We keep only what the dashboard
    consumes and compute ``mid`` here so callers don't have to repeat
    the (bid+ask)/2 + last-fallback logic.
    """
    if df is None or df.empty:
        return _empty_chain_df()

    out = pd.DataFrame(
        {
            "kind": kind,
            "expiry": expiry,
            "strike": df["strike"].astype(float),
            "bid": df.get("bid"),
            "ask": df.get("ask"),
            "last": df.get("lastPrice"),
            "iv": df.get("impliedVolatility"),
            "volume": df.get("volume"),
            "open_interest": df.get("openInterest"),
        }
    )
    out["mid"] = out.apply(_compute_mid, axis=1)
    return out[
        [
            "kind",
            "expiry",
            "strike",
            "bid",
            "ask",
            "last",
            "mid",
            "iv",
            "volume",
            "open_interest",
        ]
    ]


def _compute_mid(row: pd.Series) -> float | None:
    bid = row.get("bid")
    ask = row.get("ask")
    last = row.get("last")
    if (
        bid is not None
        and ask is not None
        and pd.notna(bid)
        and pd.notna(ask)
        and bid > 0
        and ask > 0
    ):
        return float((bid + ask) / 2)
    if last is not None and pd.notna(last) and last > 0:
        return float(last)
    return None


def _empty_chain_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "kind",
            "expiry",
            "strike",
            "bid",
            "ask",
            "last",
            "mid",
            "iv",
            "volume",
            "open_interest",
        ]
    )


def _maybe_float(x: object) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _maybe_int(x: object) -> int | None:
    f = _maybe_float(x)
    if f is None:
        return None
    return int(f)


def _coerce_date(x: object) -> date:
    if isinstance(x, date):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime().date()
    if isinstance(x, str):
        return datetime.strptime(x, "%Y-%m-%d").date()
    raise OptionsError(f"Cannot coerce {x!r} to date")