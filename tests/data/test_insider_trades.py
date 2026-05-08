"""Tests for the Form 4 parser and insider-alignment summary."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from quantitative_trading.config import get_config
from quantitative_trading.data.insider_trades import (
    InsiderTransaction,
    fetch_insider_history,
    parse_form4_xml,
    summarize_insider_alignment,
)


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


def _form4_xml(
    *,
    name: str = "Doe John A",
    is_director: bool = False,
    is_officer: bool = True,
    is_ten_pct: bool = False,
    officer_title: str = "Chief Executive Officer",
    transactions: list[dict] | None = None,
    file_footnote: str | None = None,
) -> str:
    txn_blocks = []
    for t in transactions or []:
        footnote_block = ""
        if t.get("plan_footnote"):
            footnote_block = f"<footnoteId id=\"{t['plan_footnote']}\"/>"
        txn_blocks.append(
            f"""
        <nonDerivativeTransaction>
            <transactionDate><value>{t['date']}</value></transactionDate>
            <transactionCoding>
                <transactionCode>{t['code']}</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>{t['shares']}</value></transactionShares>
                <transactionPricePerShare><value>{t['price']}</value></transactionPricePerShare>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction>
                    <value>{t.get('post', 1000)}</value>
                </sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            {footnote_block}
        </nonDerivativeTransaction>
            """
        )
    footnote = (
        f'<footnote id="F1">{file_footnote}</footnote>' if file_footnote else ""
    )
    return f"""<?xml version="1.0"?>
<ownershipDocument>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerName>{name}</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>{int(is_director)}</isDirector>
            <isOfficer>{int(is_officer)}</isOfficer>
            <isTenPercentOwner>{int(is_ten_pct)}</isTenPercentOwner>
            <officerTitle>{officer_title}</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    {''.join(txn_blocks)}
    <footnotes>{footnote}</footnotes>
</ownershipDocument>
"""


# --------------------------------------------------------------------------
# parse_form4_xml
# --------------------------------------------------------------------------


def test_parse_open_market_buy() -> None:
    xml = _form4_xml(
        name="Buffett Warren",
        officer_title="Chief Executive Officer",
        transactions=[{"date": "2024-05-01", "code": "P", "shares": 100, "price": 50}],
    )
    rows = parse_form4_xml(xml, accession="0001-1")
    assert len(rows) == 1
    t = rows[0]
    assert t.code == "P"
    assert t.role == "ceo"
    assert t.shares == 100
    assert t.price_per_share == 50
    assert t.value_usd == 5_000
    assert not t.is_10b5_1_plan


def test_parse_role_classification() -> None:
    cases = [
        ("Chief Executive Officer", True, False, False, "ceo"),
        ("Chief Financial Officer", True, False, False, "cfo"),
        ("VP Sales", True, False, False, "officer"),
        ("", False, True, False, "director"),
        ("", False, False, True, "ten_percent_owner"),
    ]
    for title, is_off, is_dir, is_ten, expected in cases:
        xml = _form4_xml(
            officer_title=title, is_officer=is_off,
            is_director=is_dir, is_ten_pct=is_ten,
            transactions=[{"date": "2024-01-15", "code": "P", "shares": 1, "price": 1}],
        )
        rows = parse_form4_xml(xml, accession="A")
        assert rows[0].role == expected, (title, expected, rows[0].role)


def test_parse_detects_10b5_1_plan_in_filing_footnote() -> None:
    xml = _form4_xml(
        transactions=[{"date": "2024-05-01", "code": "S", "shares": 1000, "price": 100,
                       "plan_footnote": "F1"}],
        file_footnote="Sale executed pursuant to a Rule 10b5-1 trading plan.",
    )
    rows = parse_form4_xml(xml, accession="A")
    assert rows[0].is_10b5_1_plan
    assert rows[0].code == "S"


def test_parse_handles_no_transactions() -> None:
    xml = _form4_xml(transactions=[])
    rows = parse_form4_xml(xml, accession="A")
    assert rows == []


def test_parse_handles_invalid_xml_gracefully() -> None:
    rows = parse_form4_xml("<broken>", accession="A")
    assert rows == []


# --------------------------------------------------------------------------
# summarize_insider_alignment
# --------------------------------------------------------------------------


def _txn(
    *, name: str, code: str, shares: float, price: float,
    on: date, role: str = "ceo", plan: bool = False,
) -> InsiderTransaction:
    return InsiderTransaction(
        accession=f"acc-{name}-{on.isoformat()}",
        filer_name=name, role=role,
        is_director=role == "director",
        is_officer=role in {"ceo", "cfo", "officer", "chairman", "president", "coo"},
        is_ten_percent_owner=role == "ten_percent_owner",
        transaction_date=on,
        code=code, shares=shares, price_per_share=price,
        value_usd=shares * price, post_holdings=None, is_10b5_1_plan=plan,
    )


def test_summarize_passes_with_net_buys_and_no_recent_sells() -> None:
    txns = [
        _txn(name="CEO", code="P", shares=100, price=50, on=date(2024, 1, 10)),
        _txn(name="CFO", code="P", shares=50, price=50, on=date(2024, 1, 20), role="cfo"),
        _txn(name="DIR", code="P", shares=20, price=50, on=date(2024, 2, 1),
             role="director"),
    ]
    res = summarize_insider_alignment(txns, as_of=date(2024, 6, 1))
    assert res.passes
    assert res.coordinated_buy
    assert res.coordinated_buy_count >= 3
    assert res.net_open_market_value_usd == pytest.approx(170 * 50)


def test_summarize_excludes_10b5_1_plan_sells_from_signal() -> None:
    txns = [
        _txn(name="CEO", code="S", shares=10_000, price=100, on=date(2024, 5, 15),
             plan=True),
    ]
    res = summarize_insider_alignment(txns, as_of=date(2024, 6, 1))
    assert res.open_market_sell_value_usd == 0  # plan sells excluded
    assert res.passes  # no real signal at all → passes (net = 0, no large recent sells)


def test_summarize_fails_on_large_recent_non_plan_sells() -> None:
    txns = [
        _txn(name="CEO", code="S", shares=100_000, price=50, on=date(2024, 5, 15)),
    ]
    res = summarize_insider_alignment(txns, as_of=date(2024, 6, 1))
    assert not res.passes
    assert res.has_large_recent_sells


def test_summarize_ignores_taxes_and_grants_and_gifts() -> None:
    txns = [
        _txn(name="CEO", code="A", shares=10_000, price=50, on=date(2024, 1, 10)),  # award
        _txn(name="CEO", code="F", shares=2_000, price=50, on=date(2024, 1, 11)),  # tax
        _txn(name="CEO", code="G", shares=500, price=50, on=date(2024, 1, 12)),    # gift
        _txn(name="CEO", code="M", shares=1_000, price=50, on=date(2024, 1, 13)),  # exercise
    ]
    res = summarize_insider_alignment(txns, as_of=date(2024, 6, 1))
    assert res.open_market_buy_value_usd == 0
    assert res.open_market_sell_value_usd == 0
    assert res.passes


def test_summarize_window_excludes_old_transactions() -> None:
    txns = [
        _txn(name="CEO", code="S", shares=200_000, price=50, on=date(2018, 1, 1)),
    ]
    res = summarize_insider_alignment(
        txns, as_of=date(2024, 6, 1), lookback_months=24,
    )
    assert res.n_transactions == 0
    assert res.passes


# --------------------------------------------------------------------------
# fetch_insider_history (mocked EDGAR)
# --------------------------------------------------------------------------


def test_fetch_insider_history_filters_by_form_and_date() -> None:
    edgar = MagicMock()
    edgar.list_filings.return_value = [
        {"accessionNumber": "A1", "filingDate": "2024-01-15", "form": "4",
         "primaryDocument": "wf-form4_x.xml"},
        {"accessionNumber": "A2", "filingDate": "2010-01-15", "form": "4",
         "primaryDocument": "wf-form4_x.xml"},  # outside window
    ]
    edgar.fetch_form4_xml.return_value = _form4_xml(
        transactions=[{"date": "2024-01-15", "code": "P", "shares": 1, "price": 1}],
    )
    out = fetch_insider_history(edgar, cik=999, start=date(2023, 1, 1),
                                end=date(2025, 1, 1))
    edgar.list_filings.assert_called_with(999, forms=("4", "4/A"))
    assert len(out) == 1
    assert out[0].accession == "A1"


def test_fetch_insider_history_tolerates_parse_errors() -> None:
    edgar = MagicMock()
    edgar.list_filings.return_value = [
        {"accessionNumber": "A1", "filingDate": "2024-01-15", "form": "4",
         "primaryDocument": "x.xml"},
    ]
    edgar.fetch_form4_xml.return_value = "<broken>"
    out = fetch_insider_history(edgar, cik=999, start=date(2023, 1, 1),
                                end=date(2025, 1, 1))
    assert out == []
