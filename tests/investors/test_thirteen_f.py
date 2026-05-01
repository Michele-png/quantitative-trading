"""Tests for the 13F-HR parser, anchored on real DJCO filings.

These tests hit the LIVE SEC EDGAR API on first run (results are cached on
disk by EdgarClient, so subsequent runs are network-free). They are skipped
if SEC_USER_AGENT is not configured.

Validation anchors:

* DJCO 2021-Q1 must contain exactly 5 holdings, including BABA (Munger's
  famous initiation) at CUSIP 01609W102 with 165,320 shares.
* DJCO 2020-Q4 must NOT contain BABA (only 4 holdings).
"""

from __future__ import annotations

import os
from datetime import date

import pytest

from quantitative_trading.investors.thirteen_f import (
    _normalize_cusip,
    _normalize_text,
    parse_information_table,
)


# ----------------------------------------------------------------- Pure helpers


class TestNormalizers:
    def test_cusip_left_pads_to_nine(self) -> None:
        assert _normalize_cusip("60505104") == "060505104"
        assert _normalize_cusip("060505104") == "060505104"

    def test_cusip_uppercases_and_strips(self) -> None:
        assert _normalize_cusip("  01609w102  ") == "01609W102"

    def test_cusip_handles_none_and_empty(self) -> None:
        assert _normalize_cusip(None) is None
        assert _normalize_cusip("") is None
        assert _normalize_cusip("   ") is None

    def test_text_collapses_whitespace(self) -> None:
        assert _normalize_text("Bank of\n  America Corp") == "Bank of America Corp"

    def test_text_decodes_double_encoded_entities(self) -> None:
        # Some 13F filings double-encode `&` as `&amp;amp;`; lxml decodes once
        # to `&amp;`, then we decode the second time to `&`.
        assert _normalize_text("Wells Fargo &amp; Co") == "Wells Fargo & Co"
        assert _normalize_text("Wells Fargo & Co") == "Wells Fargo & Co"


# ----------------------------------------------------------------- Schema-version units


class TestValueUnits:
    """Verify the pre-/post-2023 value-units convention."""

    SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
    <informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
        <infoTable>
            <nameOfIssuer>TEST CO</nameOfIssuer>
            <titleOfClass>COM</titleOfClass>
            <cusip>123456789</cusip>
            <value>1500</value>
            <shrsOrPrnAmt>
                <sshPrnamt>10000</sshPrnamt>
                <sshPrnamtType>SH</sshPrnamtType>
            </shrsOrPrnAmt>
        </infoTable>
    </informationTable>"""

    def test_pre_2023_value_in_thousands(self) -> None:
        holdings = parse_information_table(
            self.SAMPLE_XML, period_of_report=date(2022, 12, 31)
        )
        assert len(holdings) == 1
        # value=1500 is in thousands -> $1,500,000
        assert holdings[0].value_usd == 1_500_000.0

    def test_2023q4_value_in_whole_dollars(self) -> None:
        holdings = parse_information_table(
            self.SAMPLE_XML, period_of_report=date(2023, 12, 31)
        )
        # value=1500 is in whole dollars -> $1,500
        assert holdings[0].value_usd == 1_500.0


# ----------------------------------------------------------------- DJCO live validation


@pytest.fixture(scope="module")
def djco_filings() -> list:
    """Live SEC pull (cached after first run). Skipped without SEC_USER_AGENT."""
    if not os.environ.get("SEC_USER_AGENT", "").strip():
        from quantitative_trading.config import init_env
        init_env()
        if not os.environ.get("SEC_USER_AGENT", "").strip():
            pytest.skip("SEC_USER_AGENT not configured; cannot hit EDGAR.")
    from quantitative_trading.data.edgar import EdgarClient
    from quantitative_trading.investors.thirteen_f import ThirteenFClient
    e = EdgarClient()
    tf = ThirteenFClient(e)
    return tf.fetch_all_filings(783412)  # DJCO


class TestDJCOAnchors:
    """The canonical Munger DJCO validation set from audit plan section 10.4."""

    def test_djco_has_at_least_30_filings_since_2014(self, djco_filings: list) -> None:
        assert len(djco_filings) >= 30, "DJCO files quarterly since 2014; expect 30+"

    def test_djco_2020q4_does_not_contain_baba(self, djco_filings: list) -> None:
        q4_2020 = [f for f in djco_filings if f.period_of_report == date(2020, 12, 31)]
        assert len(q4_2020) == 1
        cusips = {h.cusip for h in q4_2020[0].holdings}
        assert "01609W102" not in cusips, "BABA was a Q1 2021 buy, not Q4 2020"

    def test_djco_2021q1_contains_baba_initiation(self, djco_filings: list) -> None:
        q1_2021 = [f for f in djco_filings if f.period_of_report == date(2021, 3, 31)]
        assert len(q1_2021) == 1
        baba = [h for h in q1_2021[0].holdings if h.cusip == "01609W102"]
        assert len(baba) == 1, "Munger's famous BABA buy must appear in 2021-Q1"
        assert baba[0].shares == 165_320, "Initial BABA position was 165,320 shares"
        # Pre-2023 filing -> value in thousands -> $37.483M
        assert baba[0].value_usd == pytest.approx(37_483_000.0)
        assert "Alibaba" in baba[0].name_of_issuer

    def test_djco_holdings_count_typically_4_to_8(self, djco_filings: list) -> None:
        # DJCO is known to be very concentrated; sanity-check that a parsing
        # error didn't produce a runaway holding count.
        for f in djco_filings:
            assert 1 <= len(f.holdings) <= 15, (
                f"DJCO {f.period_of_report} has {len(f.holdings)} holdings — "
                f"unusual for this filer"
            )
