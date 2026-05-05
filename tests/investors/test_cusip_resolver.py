"""Tests for the dual-source CUSIP resolver.

Live tests hit OpenFIGI (free, no key) and SEC. Cached after first run.
Skipped if SEC_USER_AGENT is not configured.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module")
def resolver():
    if not os.environ.get("SEC_USER_AGENT", "").strip():
        from quantitative_trading.config import init_env
        init_env()
        if not os.environ.get("SEC_USER_AGENT", "").strip():
            pytest.skip("SEC_USER_AGENT not configured.")
    from quantitative_trading.data.edgar import EdgarClient
    from quantitative_trading.investors.cusip_resolver import CusipResolver
    return CusipResolver(EdgarClient())


class TestKnownCusips:
    """Spot-check on a small set of CUSIPs covering common, ADR, and bad cases."""

    def test_aapl(self, resolver) -> None:
        r = resolver.resolve("037833100")
        assert r.ticker == "AAPL"
        assert r.cik == 320193
        assert r.security_type == "Common Stock"
        assert r.source == "verified"

    def test_bac(self, resolver) -> None:
        r = resolver.resolve("060505104")
        assert r.ticker == "BAC"
        assert r.cik == 70858
        assert r.is_evaluable_security_type

    def test_baba_is_adr(self, resolver) -> None:
        r = resolver.resolve("01609W102")
        assert r.ticker == "BABA"
        assert r.security_type == "ADR"
        # ADRs are NOT evaluable per audit plan section 5 even when SEC has a CIK.
        assert not r.is_evaluable_security_type

    def test_alphabet_share_classes_share_cik(self, resolver) -> None:
        r1 = resolver.resolve("02079K305")
        r2 = resolver.resolve("02079K107")
        # Both Class A and Class C resolve to the same Alphabet CIK.
        assert r1.cik == r2.cik == 1652044
        # And both are common stock (not preferred / unit / warrant).
        assert r1.security_type == r2.security_type == "Common Stock"

    def test_unresolvable_cusip(self, resolver) -> None:
        r = resolver.resolve("999999999")
        assert not r.is_resolved
        assert r.source == "unresolved"

    def test_cusip_left_padding(self, resolver) -> None:
        # 8-character CUSIPs (leading zero stripped) should resolve to the
        # same security as the 9-character form.
        r1 = resolver.resolve("60505104")
        r2 = resolver.resolve("060505104")
        assert r1.cusip == "060505104"
        assert r2.cusip == "060505104"
        assert r1.ticker == r2.ticker == "BAC"


class TestEvaluabilityFlag:
    def test_common_stock_is_evaluable(self) -> None:
        from quantitative_trading.investors.cusip_resolver import CusipResolution
        r = CusipResolution(
            cusip="037833100", ticker="AAPL", cik=320193, issuer_name="APPLE INC",
            security_type="Common Stock", exchange="US", source="verified",
        )
        assert r.is_evaluable_security_type
        assert r.is_verified_against_sec

    def test_adr_is_not_evaluable(self) -> None:
        from quantitative_trading.investors.cusip_resolver import CusipResolution
        r = CusipResolution(
            cusip="01609W102", ticker="BABA", cik=1577552, issuer_name="ALIBABA",
            security_type="ADR", exchange="US", source="verified",
        )
        assert not r.is_evaluable_security_type

    def test_preferred_stock_is_not_evaluable(self) -> None:
        from quantitative_trading.investors.cusip_resolver import CusipResolution
        r = CusipResolution(
            cusip="123456789", ticker="ANY-PFD", cik=999, issuer_name="ANY",
            security_type="Preferred", exchange="US", source="verified",
        )
        assert not r.is_evaluable_security_type
