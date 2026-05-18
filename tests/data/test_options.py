"""Unit tests for OptionsClient + the Black-Scholes helpers.

Mocks `yfinance.Ticker.option_chain` with a tiny synthetic chain to verify:

  * `get_chain` returns the unified (calls + puts) DataFrame with `mid`
    computed from bid/ask (and a `last` fallback).
  * `list_expiries` parses Yahoo's ISO date strings.
  * `nearest_strike` honours the "below"/"above"/"closest" preference.
  * `pick` returns the right ChainQuote for cash-secured-put style
    selection (strike ≤ MoS) and covered-call style (strike ≥ Sticker).
  * `bs_premium` matches a known reference value at the money and
    converges to intrinsic value as expiry → 0.
  * `fallback_premium` returns 0 when the option has already expired.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from value_investing_backend.data.options import (
    DEFAULT_R,
    OptionsClient,
    bs_delta,
    bs_premium,
)

# ---------------------------------------------------------------------------
# Synthetic chain
# ---------------------------------------------------------------------------


def _synthetic_calls() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contractSymbol": [f"FAKE_C{int(s)}" for s in (90, 100, 110)],
            "strike": [90.0, 100.0, 110.0],
            "lastPrice": [12.0, 5.0, 1.5],
            "bid": [11.8, 4.9, 1.4],
            "ask": [12.2, 5.1, 1.6],
            "volume": [10, 200, 50],
            "openInterest": [100, 1500, 250],
            "impliedVolatility": [0.30, 0.28, 0.32],
        }
    )


def _synthetic_puts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contractSymbol": [f"FAKE_P{int(s)}" for s in (90, 100, 110)],
            "strike": [90.0, 100.0, 110.0],
            "lastPrice": [1.2, 3.5, 9.8],
            # Strike 90 has no real bid (illiquid far-OTM) — `mid` must
            # fall back to `last`.
            "bid": [0.0, 3.4, 9.6],
            "ask": [0.0, 3.6, 10.0],
            "volume": [5, 150, 20],
            "openInterest": [30, 1000, 200],
            "impliedVolatility": [0.34, 0.29, 0.28],
        }
    )


def _mock_yf_ticker(expiries: list[str]) -> MagicMock:
    mock = MagicMock()
    mock.options = tuple(expiries)
    chain_ns = MagicMock()
    chain_ns.calls = _synthetic_calls()
    chain_ns.puts = _synthetic_puts()
    mock.option_chain.return_value = chain_ns
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_expiries_parses_strings() -> None:
    with patch("yfinance.Ticker") as ticker_cls:
        ticker_cls.return_value = _mock_yf_ticker(
            ["2026-06-19", "2026-07-17", "not-a-date", "2026-08-21"]
        )
        client = OptionsClient()
        expiries = client.list_expiries("FAKE")
    assert expiries == [
        date(2026, 6, 19),
        date(2026, 7, 17),
        date(2026, 8, 21),
    ]


def test_get_chain_returns_unified_frame_with_mid() -> None:
    with patch("yfinance.Ticker") as ticker_cls:
        ticker_cls.return_value = _mock_yf_ticker(["2026-06-19"])
        client = OptionsClient()
        chain = client.get_chain("FAKE", date(2026, 6, 19))

    assert set(chain["kind"]) == {"call", "put"}
    # 3 calls + 3 puts
    assert len(chain) == 6

    # ATM call has bid 4.9 / ask 5.1 -> mid 5.0
    atm_call = chain[(chain["kind"] == "call") & (chain["strike"] == 100.0)].iloc[0]
    assert atm_call["mid"] == pytest.approx(5.0, abs=0.001)

    # Illiquid far-OTM put has bid=ask=0 -> mid falls back to last (1.2).
    far_otm_put = chain[(chain["kind"] == "put") & (chain["strike"] == 90.0)].iloc[0]
    assert far_otm_put["mid"] == pytest.approx(1.2, abs=0.001)


def test_get_chain_handles_yfinance_failure() -> None:
    with patch("yfinance.Ticker") as ticker_cls:
        mock = MagicMock()
        mock.option_chain.side_effect = RuntimeError("boom")
        ticker_cls.return_value = mock
        client = OptionsClient()
        chain = client.get_chain("FAKE", date(2026, 6, 19))
    assert chain.empty
    # Schema preserved so the dashboard doesn't crash on missing columns.
    assert set(chain.columns) >= {"kind", "strike", "mid"}


def test_nearest_strike_below_constrains() -> None:
    strikes = [85.0, 90.0, 95.0, 100.0, 105.0]
    # For a cash-secured put at MoS = 98, the seller wants strike ≤ MoS.
    assert OptionsClient.nearest_strike(strikes, 98.0, prefer="below") == 95.0
    # For a covered call at Sticker = 102, want strike ≥ Sticker.
    assert OptionsClient.nearest_strike(strikes, 102.0, prefer="above") == 105.0
    # Closest ignores side.
    assert OptionsClient.nearest_strike(strikes, 98.0, prefer="closest") == 100.0


def test_nearest_strike_returns_none_when_no_eligible_strike() -> None:
    # Target above every strike but constrained to "above": no eligible.
    assert OptionsClient.nearest_strike([90.0, 95.0], 100.0, prefer="above") is None
    # Empty list.
    assert OptionsClient.nearest_strike([], 100.0) is None


def test_pick_csp_at_mos() -> None:
    with patch("yfinance.Ticker") as ticker_cls:
        ticker_cls.return_value = _mock_yf_ticker(["2026-06-19"])
        client = OptionsClient()
        chain = client.get_chain("FAKE", date(2026, 6, 19))

    # MoS = 98 -> closest below = strike 90 (we only have 90/100/110 in
    # the synthetic). Real chains would have 95/97.5/100 etc.
    quote = client.pick(chain, kind="put", target_strike=98.0, prefer="below")
    assert quote is not None
    assert quote.kind == "put"
    assert quote.strike == 90.0
    # Mid for this strike fell back to `last` = 1.2.
    assert quote.mid == pytest.approx(1.2, abs=0.001)
    # IV was 0.34 in the synthetic.
    assert quote.iv == pytest.approx(0.34, abs=1e-6)


def test_pick_cc_at_sticker() -> None:
    with patch("yfinance.Ticker") as ticker_cls:
        ticker_cls.return_value = _mock_yf_ticker(["2026-06-19"])
        client = OptionsClient()
        chain = client.get_chain("FAKE", date(2026, 6, 19))

    # Sticker = 102 -> closest above = strike 110.
    quote = client.pick(chain, kind="call", target_strike=102.0, prefer="above")
    assert quote is not None
    assert quote.kind == "call"
    assert quote.strike == 110.0
    # Mid = (1.4 + 1.6) / 2 = 1.5.
    assert quote.mid == pytest.approx(1.5, abs=0.001)


def test_bs_premium_atm_matches_reference() -> None:
    # ATM, spot=100, strike=100, sigma=20%, T=1y, r=4%.
    # Closed-form:
    #   d1 = (0 + (0.04 + 0.02) * 1) / 0.20 = 0.30, d2 = 0.10
    #   N(0.30) ≈ 0.6179, N(0.10) ≈ 0.5398, e^(-0.04) ≈ 0.96079
    #   C = 100 * 0.6179 - 100 * 0.96079 * 0.5398 ≈ 9.93
    #   P = 100 * 0.96079 * 0.4602 - 100 * 0.3821 ≈ 6.00
    c = bs_premium(100, 100, 1.0, 0.20, 0.04, "call")
    p = bs_premium(100, 100, 1.0, 0.20, 0.04, "put")
    assert c == pytest.approx(9.93, abs=0.05)
    assert p == pytest.approx(6.00, abs=0.05)
    # Put-call parity: C - P = S - K*exp(-rT) ≈ 100 - 96.079 = 3.921
    assert (c - p) == pytest.approx(100 - 100 * np.exp(-0.04), abs=0.01)


def test_bs_premium_converges_to_intrinsic_at_expiry() -> None:
    # Tiny time to expiry: premium ≈ max(intrinsic, 0).
    call_itm = bs_premium(110, 100, 1e-6, 0.20, 0.04, "call")
    call_otm = bs_premium(90, 100, 1e-6, 0.20, 0.04, "call")
    put_itm = bs_premium(90, 100, 1e-6, 0.20, 0.04, "put")
    put_otm = bs_premium(110, 100, 1e-6, 0.20, 0.04, "put")
    assert call_itm == pytest.approx(10.0, abs=0.001)
    assert call_otm == pytest.approx(0.0, abs=0.001)
    assert put_itm == pytest.approx(10.0, abs=0.001)
    assert put_otm == pytest.approx(0.0, abs=0.001)


def test_bs_premium_handles_degenerate_inputs() -> None:
    assert bs_premium(0, 100, 1, 0.2, 0.04, "call") == 0.0
    assert bs_premium(100, 0, 1, 0.2, 0.04, "call") == 0.0
    assert bs_premium(100, 100, 0, 0.2, 0.04, "call") == 0.0
    assert bs_premium(100, 100, 1, 0, 0.04, "call") == 0.0


def test_bs_delta_signs() -> None:
    # ATM 1-year delta ≈ 0.55 call / -0.45 put under 20% IV, 4% r.
    dc = bs_delta(100, 100, 1.0, 0.20, 0.04, "call")
    dp = bs_delta(100, 100, 1.0, 0.20, 0.04, "put")
    assert 0.45 < dc < 0.7
    assert -0.55 < dp < -0.3
    # Put-call delta identity: dC - dP = 1.
    assert (dc - dp) == pytest.approx(1.0, abs=1e-6)


def test_fallback_premium_returns_zero_after_expiry() -> None:
    p = OptionsClient.fallback_premium(
        spot=100.0,
        strike=100.0,
        as_of=date(2026, 6, 20),
        expiry=date(2026, 6, 19),  # already past
        iv=0.20,
        kind="put",
    )
    assert p == 0.0


def test_fallback_premium_matches_direct_bs() -> None:
    as_of = date(2026, 1, 1)
    expiry = date(2027, 1, 1)
    direct = bs_premium(100.0, 100.0, 1.0, 0.20, DEFAULT_R, "call")
    via = OptionsClient.fallback_premium(
        spot=100.0,
        strike=100.0,
        as_of=as_of,
        expiry=expiry,
        iv=0.20,
        kind="call",
    )
    # 365/365.25 ≈ 1.000685 — they should match within a fraction of a cent.
    assert via == pytest.approx(direct, rel=0.001)
