"""Unit tests for PointInTimeFacts using synthetic SEC-shaped fixtures.

The fixtures mirror the real SEC `companyfacts` payload structure, including
the `start` field for flow concepts and the gotcha that the `fy` field is the
FILING's year (not the data year).
"""

from __future__ import annotations

from datetime import date

import pytest

from quantitative_trading.data.pit_facts import PointInTimeFacts


def _flow(
    *,
    start: str,
    end: str,
    val: float,
    accn: str,
    fy_filing: int,
    form: str,
    filed: str,
) -> dict:
    """Flow-concept entry (income statement, cash flow)."""
    return {
        "start": start,
        "end": end,
        "val": val,
        "accn": accn,
        "fy": fy_filing,
        "fp": "FY",
        "form": form,
        "filed": filed,
    }


def _snap(
    *,
    end: str,
    val: float,
    accn: str,
    fy_filing: int,
    form: str,
    filed: str,
) -> dict:
    """Snapshot-concept entry (balance sheet)."""
    return {
        "end": end,
        "val": val,
        "accn": accn,
        "fy": fy_filing,
        "fp": "FY",
        "form": form,
        "filed": filed,
    }


def _facts(units_by_concept: dict[str, list[dict]]) -> dict:
    """Build a SEC-shaped facts payload."""
    facts: dict = {"facts": {"us-gaap": {}, "dei": {}}}
    for concept, entries in units_by_concept.items():
        if concept.startswith("dei:"):
            taxonomy = "dei"
            name = concept[len("dei:") :]
            unit = "shares"
        elif concept.startswith("EarningsPerShare"):
            taxonomy = "us-gaap"
            name = concept
            unit = "USD/shares"
        else:
            taxonomy = "us-gaap"
            name = concept
            unit = "USD"
        facts["facts"][taxonomy][name] = {
            "label": name,
            "units": {unit: entries},
        }
    return {"cik": 999, "entityName": "Test Co", **facts}


# --------------------------------------------------------------------------
# Basic happy path
# --------------------------------------------------------------------------


def test_flow_basic_fy_value() -> None:
    payload = _facts(
        {
            "Revenues": [
                _flow(start="2014-01-01", end="2014-12-31", val=100,
                      accn="A1", fy_filing=2014, form="10-K", filed="2015-02-01"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)
    fv = pit.get_annual("revenue", 2014, as_of=date(2016, 1, 1))
    assert fv is not None
    assert fv.val == 100
    assert fv.end == date(2014, 12, 31)
    assert fv.start == date(2014, 1, 1)
    assert fv.concept == "Revenues"


def test_snapshot_basic_fy_value() -> None:
    payload = _facts(
        {
            "StockholdersEquity": [
                _snap(end="2014-12-31", val=500, accn="A1",
                      fy_filing=2014, form="10-K", filed="2015-02-01"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)
    fv = pit.get_annual("stockholders_equity", 2014, as_of=date(2016, 1, 1))
    assert fv is not None
    assert fv.val == 500
    assert fv.start is None  # snapshot — no period start


def test_fiscal_year_end_resolves_to_latest_end_in_fy_filings() -> None:
    """fiscal_year_end(Y) returns the company's official period-end date for FY-Y."""
    payload = _facts(
        {
            "Revenues": [
                _flow(start="2013-01-01", end="2013-12-31", val=90,
                      accn="A2013", fy_filing=2013, form="10-K", filed="2014-02-01"),
                _flow(start="2014-01-01", end="2014-12-31", val=100,
                      accn="A2014", fy_filing=2014, form="10-K", filed="2015-02-01"),
                _flow(start="2013-01-01", end="2013-12-31", val=90,
                      accn="A2014", fy_filing=2014, form="10-K", filed="2015-02-01"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)
    assert pit.fiscal_year_end(2014) == date(2014, 12, 31)
    assert pit.fiscal_year_end(2013) == date(2013, 12, 31)
    assert pit.fiscal_year_end(2099) is None


# --------------------------------------------------------------------------
# PIT cutoff
# --------------------------------------------------------------------------


def test_pit_cutoff_hides_future_filings() -> None:
    payload = _facts(
        {
            "Revenues": [
                _flow(start="2014-01-01", end="2014-12-31", val=100,
                      accn="A1", fy_filing=2014, form="10-K", filed="2015-02-01"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)
    assert pit.get_annual("revenue", 2014, as_of=date(2015, 1, 31)) is None
    fv_after = pit.get_annual("revenue", 2014, as_of=date(2015, 2, 1))
    assert fv_after is not None and fv_after.val == 100


# --------------------------------------------------------------------------
# Restatement
# --------------------------------------------------------------------------


def test_restatement_visible_after_restate_date() -> None:
    """Original 10-K reports 100; an amended 10-K/A later restates to 95."""
    payload = _facts(
        {
            "Revenues": [
                _flow(start="2014-01-01", end="2014-12-31", val=100,
                      accn="A1", fy_filing=2014, form="10-K", filed="2015-02-01"),
                _flow(start="2014-01-01", end="2014-12-31", val=95,
                      accn="A2", fy_filing=2014, form="10-K/A", filed="2016-06-15"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)

    fv_before = pit.get_annual("revenue", 2014, as_of=date(2016, 6, 14))
    assert fv_before is not None and fv_before.val == 100
    assert fv_before.form == "10-K"

    fv_after = pit.get_annual("revenue", 2014, as_of=date(2016, 6, 15))
    assert fv_after is not None and fv_after.val == 95
    assert fv_after.form == "10-K/A"


def test_restatement_in_subsequent_year_filing_visible() -> None:
    """The FY-(Y+1) 10-K reports comparative FY-Y data with a restated value.
    The restatement should be visible after the (Y+1) filing date."""
    payload = _facts(
        {
            "NetIncomeLoss": [
                _flow(start="2017-01-01", end="2017-12-31", val=16_540,
                      accn="K2017", fy_filing=2017, form="10-K", filed="2018-02-21"),
                _flow(start="2017-01-01", end="2017-12-31", val=1_300,
                      accn="K2018", fy_filing=2018, form="10-K", filed="2019-02-20"),
                _flow(start="2018-01-01", end="2018-12-31", val=15_297,
                      accn="K2018", fy_filing=2018, form="10-K", filed="2019-02-20"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)

    fv_before = pit.get_annual("net_income", 2017, as_of=date(2019, 2, 19))
    assert fv_before is not None and fv_before.val == 16_540

    fv_after = pit.get_annual("net_income", 2017, as_of=date(2019, 2, 20))
    assert fv_after is not None and fv_after.val == 1_300


# --------------------------------------------------------------------------
# Comparative-data trap (the bug that motivated the refactor)
# --------------------------------------------------------------------------


def test_comparative_prior_year_data_in_same_filing_does_not_leak() -> None:
    """The FY2014 10-K reports comparative data for FY2012, FY2013, AND FY2014,
    all tagged with `fy=2014` (the filing year). Querying for FY2013 must
    return the FY2013 entry, not the FY2014 one.

    Each fiscal year being queried also needs its own 10-K filing in the
    fixture (so that `fiscal_year_end(Y)` can resolve the end date).
    """
    payload = _facts(
        {
            "NetCashProvidedByUsedInOperatingActivities": [
                # FY2012 10-K: current year only (single entry).
                _flow(start="2011-09-25", end="2012-09-29", val=50_860_000_000,
                      accn="K2012", fy_filing=2012, form="10-K", filed="2012-10-31"),
                # FY2013 10-K: current + comparative.
                _flow(start="2011-09-25", end="2012-09-29", val=50_860_000_000,
                      accn="K2013", fy_filing=2013, form="10-K", filed="2013-10-30"),
                _flow(start="2012-09-30", end="2013-09-28", val=53_670_000_000,
                      accn="K2013", fy_filing=2013, form="10-K", filed="2013-10-30"),
                # FY2014 10-K: current + 2 prior years comparative.
                _flow(start="2011-09-25", end="2012-09-29", val=50_860_000_000,
                      accn="K2014", fy_filing=2014, form="10-K", filed="2014-10-27"),
                _flow(start="2012-09-30", end="2013-09-28", val=53_670_000_000,
                      accn="K2014", fy_filing=2014, form="10-K", filed="2014-10-27"),
                _flow(start="2013-09-29", end="2014-09-27", val=59_710_000_000,
                      accn="K2014", fy_filing=2014, form="10-K", filed="2014-10-27"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)

    fv2014 = pit.get_annual("operating_cash_flow", 2014, as_of=date(2016, 1, 1))
    assert fv2014 is not None and fv2014.val == 59_710_000_000

    fv2013 = pit.get_annual("operating_cash_flow", 2013, as_of=date(2016, 1, 1))
    assert fv2013 is not None and fv2013.val == 53_670_000_000

    fv2012 = pit.get_annual("operating_cash_flow", 2012, as_of=date(2016, 1, 1))
    assert fv2012 is not None and fv2012.val == 50_860_000_000


def test_quarterly_subperiods_in_same_filing_excluded() -> None:
    payload = _facts(
        {
            "Revenues": [
                _flow(start="2014-01-01", end="2014-03-31", val=25,
                      accn="A1", fy_filing=2014, form="10-K", filed="2015-02-01"),
                _flow(start="2014-04-01", end="2014-06-30", val=26,
                      accn="A1", fy_filing=2014, form="10-K", filed="2015-02-01"),
                _flow(start="2014-07-01", end="2014-09-30", val=24,
                      accn="A1", fy_filing=2014, form="10-K", filed="2015-02-01"),
                _flow(start="2014-10-01", end="2014-12-31", val=25,
                      accn="A1", fy_filing=2014, form="10-K", filed="2015-02-01"),
                _flow(start="2014-01-01", end="2014-12-31", val=100,
                      accn="A1", fy_filing=2014, form="10-K", filed="2015-02-01"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)
    fv = pit.get_annual("revenue", 2014, as_of=date(2016, 1, 1))
    assert fv is not None and fv.val == 100


def test_balance_sheet_picks_correct_year_from_multi_year_filing() -> None:
    """The 10-K balance sheet has 2-3 years of comparative data with shared fy_filing
    but different ends. Each FY queried must have its own 10-K to anchor the end date."""
    payload = _facts(
        {
            "StockholdersEquity": [
                # FY2012 10-K: just current year snapshot.
                _snap(end="2012-09-29", val=118, accn="K2012",
                      fy_filing=2012, form="10-K", filed="2012-10-31"),
                # FY2013 10-K: current + comparative.
                _snap(end="2012-09-29", val=118, accn="K2013",
                      fy_filing=2013, form="10-K", filed="2013-10-30"),
                _snap(end="2013-09-28", val=123, accn="K2013",
                      fy_filing=2013, form="10-K", filed="2013-10-30"),
                # FY2014 10-K: current + comparative + extra historical anchor.
                _snap(end="2011-09-24", val=76, accn="K2014",
                      fy_filing=2014, form="10-K", filed="2014-10-27"),
                _snap(end="2012-09-29", val=118, accn="K2014",
                      fy_filing=2014, form="10-K", filed="2014-10-27"),
                _snap(end="2013-09-28", val=123, accn="K2014",
                      fy_filing=2014, form="10-K", filed="2014-10-27"),
                _snap(end="2014-09-27", val=111, accn="K2014",
                      fy_filing=2014, form="10-K", filed="2014-10-27"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)
    assert pit.get_annual("stockholders_equity", 2014,
                          as_of=date(2016, 1, 1)).val == 111
    assert pit.get_annual("stockholders_equity", 2013,
                          as_of=date(2016, 1, 1)).val == 123
    assert pit.get_annual("stockholders_equity", 2012,
                          as_of=date(2016, 1, 1)).val == 118


# --------------------------------------------------------------------------
# 52/53-week fiscal calendars crossing calendar year boundaries (the JNJ case)
# --------------------------------------------------------------------------


def test_53_week_fiscal_year_accepted() -> None:
    """Some retailers have occasional 53-week fiscal years (~371 days)."""
    payload = _facts(
        {
            "Revenues": [
                _flow(start="2017-01-29", end="2018-02-03", val=500,
                      accn="A1", fy_filing=2017, form="10-K", filed="2018-03-15"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)
    fv = pit.get_annual("revenue", 2017, as_of=date(2019, 1, 1))
    assert fv is not None and fv.val == 500


def test_jnj_style_disambiguation_when_fy_ends_cross_calendar_year() -> None:
    """JNJ FY2016 ends Jan 1, 2017; FY2017 ends Dec 31, 2017. Both have
    end.year == 2017, and querying FY2017 must NOT return the FY2016 value."""
    payload = _facts(
        {
            "NetIncomeLoss": [
                # FY2016 10-K (filed Feb 2017): reports period Jan 4, 2016 - Jan 1, 2017.
                _flow(start="2016-01-04", end="2017-01-01", val=16_540,
                      accn="K2016", fy_filing=2016, form="10-K", filed="2017-02-27"),
                # FY2017 10-K (filed Feb 2018): reports period Jan 2, 2017 - Dec 31, 2017,
                # with comparative FY2016 included as well.
                _flow(start="2016-01-04", end="2017-01-01", val=16_540,
                      accn="K2017", fy_filing=2017, form="10-K", filed="2018-02-21"),
                _flow(start="2017-01-02", end="2017-12-31", val=1_300,
                      accn="K2017", fy_filing=2017, form="10-K", filed="2018-02-21"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)

    # FY2016 (ends 2017-01-01).
    assert pit.fiscal_year_end(2016) == date(2017, 1, 1)
    fv_2016 = pit.get_annual("net_income", 2016, as_of=date(2019, 1, 1))
    assert fv_2016 is not None and fv_2016.val == 16_540
    assert fv_2016.end == date(2017, 1, 1)

    # FY2017 (ends 2017-12-31) — must not pick up the Jan 1, 2017 FY2016 value.
    assert pit.fiscal_year_end(2017) == date(2017, 12, 31)
    fv_2017 = pit.get_annual("net_income", 2017, as_of=date(2019, 1, 1))
    assert fv_2017 is not None and fv_2017.val == 1_300
    assert fv_2017.end == date(2017, 12, 31)


# --------------------------------------------------------------------------
# Concept fallback
# --------------------------------------------------------------------------


def test_concept_fallback_when_primary_missing() -> None:
    payload = _facts(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                _flow(start="2018-01-01", end="2018-12-31", val=200,
                      accn="B1", fy_filing=2018, form="10-K", filed="2019-02-15"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)
    fv = pit.get_annual("revenue", 2018, as_of=date(2020, 1, 1))
    assert fv is not None
    assert fv.val == 200
    assert fv.concept == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_concept_fallback_skips_concept_with_no_fy_match() -> None:
    """Apple-like case: Revenues exists but only for 2018; SalesRevenueNet for 2014."""
    payload = _facts(
        {
            "Revenues": [
                _flow(start="2018-01-01", end="2018-12-31", val=200,
                      accn="X", fy_filing=2018, form="10-K", filed="2019-02-15"),
            ],
            "SalesRevenueNet": [
                _flow(start="2014-01-01", end="2014-12-31", val=180,
                      accn="Y", fy_filing=2014, form="10-K", filed="2015-02-15"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)
    fv = pit.get_annual("revenue", 2014, as_of=date(2020, 1, 1))
    assert fv is not None
    assert fv.val == 180
    assert fv.concept == "SalesRevenueNet"


# --------------------------------------------------------------------------
# Form filtering
# --------------------------------------------------------------------------


def test_10q_full_year_values_not_used_for_fy_queries() -> None:
    payload = _facts(
        {
            "Revenues": [
                _flow(start="2014-01-01", end="2014-12-31", val=999,
                      accn="QQ", fy_filing=2014, form="10-Q", filed="2015-01-15"),
                _flow(start="2014-01-01", end="2014-12-31", val=100,
                      accn="A1", fy_filing=2014, form="10-K", filed="2015-02-01"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)
    fv = pit.get_annual("revenue", 2014, as_of=date(2015, 2, 1))
    assert fv is not None
    assert fv.val == 100
    assert fv.form == "10-K"


# --------------------------------------------------------------------------
# Helper queries
# --------------------------------------------------------------------------


def test_annual_series_returns_contiguous_dict() -> None:
    payload = _facts(
        {
            "Revenues": [
                _flow(start=f"{fy}-01-01", end=f"{fy}-12-31", val=100 + fy,
                      accn=f"A{fy}", fy_filing=fy, form="10-K", filed=f"{fy + 1}-02-01")
                for fy in [2010, 2011, 2013]  # 2012 missing on purpose
            ],
        }
    )
    pit = PointInTimeFacts(payload)
    series = pit.get_annual_series("revenue", last_fiscal_year=2013, n_years=4,
                                   as_of=date(2015, 1, 1))
    assert sorted(series.keys()) == [2010, 2011, 2012, 2013]
    assert series[2010].val == 2110
    assert series[2011].val == 2111
    assert series[2012] is None  # gap year
    assert series[2013].val == 2113


def test_latest_fiscal_year_with_data_respects_pit() -> None:
    payload = _facts(
        {
            "Revenues": [
                _flow(start="2014-01-01", end="2014-12-31", val=100,
                      accn="A1", fy_filing=2014, form="10-K", filed="2015-02-01"),
                _flow(start="2015-01-01", end="2015-12-31", val=110,
                      accn="A2", fy_filing=2015, form="10-K", filed="2016-02-01"),
            ],
        }
    )
    pit = PointInTimeFacts(payload)
    assert pit.latest_fiscal_year_with_data("revenue",
                                             as_of=date(2015, 6, 1)) == 2014
    assert pit.latest_fiscal_year_with_data("revenue",
                                             as_of=date(2016, 6, 1)) == 2015
    assert pit.latest_fiscal_year_with_data("revenue",
                                             as_of=date(2014, 1, 1)) is None


def test_unknown_concept_group_raises() -> None:
    pit = PointInTimeFacts(_facts({}))
    with pytest.raises(ValueError, match="Unknown concept group"):
        pit.get_annual("not_a_real_group", 2020, as_of=date(2021, 1, 1))
