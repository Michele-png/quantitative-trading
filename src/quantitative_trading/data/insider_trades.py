"""Form 4 insider-trading data layer.

Form 4 is the SEC filing officers, directors, and 10%-owners must submit when
they trade their company's securities. EDGAR exposes a structured XML primary
document alongside the human-readable HTML wrapper, which we parse here.

Phil Town's "skin in the game" check cares about ONE specific signal:
heavy, coordinated **open-market buying** (transaction code ``P``) by named
officers and directors. Routine activity to filter out:

    * ``S`` — open-market sells (often diversification or pre-scheduled)
    * ``A`` — grants / awards (compensation, not skin in the game)
    * ``M`` — exercise of derivative securities
    * ``F`` — payment of tax via share withholding
    * ``G`` — gifts (no economic decision)

Anything filed under a Rule 10b5-1 trading plan is also dropped from the
"signal" set: those orders are scheduled in advance to avoid insider-trading
liability and don't reflect a discretionary view at the trade date.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup

if TYPE_CHECKING:  # avoid runtime cycle through data.__init__
    from quantitative_trading.data.edgar import EdgarClient


log = logging.getLogger(__name__)


# Transaction codes per SEC Form 4 instructions (subset that matters here).
OPEN_MARKET_BUY = "P"
OPEN_MARKET_SELL = "S"
TAX_WITHHOLDING = "F"
GIFT = "G"
GRANT_OR_AWARD = "A"
DERIVATIVE_EXERCISE = "M"


@dataclass(frozen=True)
class InsiderTransaction:
    """One Form 4 transaction row, normalized for downstream analysis."""

    accession: str
    filer_name: str
    role: str  # canonical: "ceo" | "cfo" | "director" | "officer" | "ten_percent_owner" | "other"
    is_director: bool
    is_officer: bool
    is_ten_percent_owner: bool
    transaction_date: date
    code: str
    shares: float
    price_per_share: float
    value_usd: float  # shares * price
    post_holdings: float | None
    is_10b5_1_plan: bool


@dataclass(frozen=True)
class InsiderAlignmentResult:
    """Aggregated insider-alignment signal computed over a lookback window."""

    lookback_start: date
    lookback_end: date
    n_transactions: int
    open_market_buy_value_usd: float
    open_market_sell_value_usd: float  # excluding 10b5-1 plans + tax withholding
    net_open_market_value_usd: float
    coordinated_buy: bool
    coordinated_buy_count: int  # distinct insiders buying within 60 days
    has_large_recent_sells: bool  # > $1M of non-plan sells in last 90 days
    by_role_net_usd: dict[str, float]
    passes: bool
    rationale: str


# --------------------------------------------------------------------------
# Role classification
# --------------------------------------------------------------------------


def _normalize_title(officer_title: str | None) -> str:
    """Map a free-text officer title to a canonical role bucket."""
    if not officer_title:
        return "officer"
    t = officer_title.lower()
    # Order matters: "Chief Executive Officer" → ceo before "officer".
    if "chief executive" in t or t.strip() in {"ceo", "principal executive officer"}:
        return "ceo"
    if "chief financial" in t or t.strip() in {"cfo", "principal financial officer"}:
        return "cfo"
    if "chief operating" in t:
        return "coo"
    if "chairman" in t or "chair " in t or t.strip() == "chair":
        return "chairman"
    if "president" in t:
        return "president"
    return "officer"


def _classify_role(
    *,
    is_director: bool,
    is_officer: bool,
    is_ten_percent_owner: bool,
    officer_title: str | None,
) -> str:
    """Pick the most specific role label from the boolean flags + title text."""
    if is_officer:
        return _normalize_title(officer_title)
    if is_director:
        return "director"
    if is_ten_percent_owner:
        return "ten_percent_owner"
    return "other"


# --------------------------------------------------------------------------
# XML parsing
# --------------------------------------------------------------------------


def _to_bool(s: Any) -> bool:
    if s is None:
        return False
    text = str(s).strip().lower()
    return text in {"1", "true", "yes"}


def _to_float(s: Any) -> float:
    if s is None:
        return 0.0
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return 0.0


def _txt(node: Any, name: str) -> str | None:
    if node is None:
        return None
    found = node.find(name)
    if found is None:
        return None
    value = found.find("value")
    target = value if value is not None else found
    return (target.get_text() or "").strip() or None


def _is_10b5_1(footnotes_text: str) -> bool:
    """Detect Rule 10b5-1 plan disclosures in any free-text field of the filing."""
    if not footnotes_text:
        return False
    return bool(re.search(r"10b5\s*-?\s*1", footnotes_text, re.IGNORECASE))


def parse_form4_xml(xml_text: str, accession: str) -> list[InsiderTransaction]:
    """Parse a Form 4 XML primary document into InsiderTransaction rows.

    Form 4 has two transaction blocks: ``nonDerivativeTransaction`` (common
    stock / direct holdings) and ``derivativeTransaction`` (options, RSUs,
    warrants). Phil Town's open-market buying signal lives in the
    non-derivative block — derivative grants are compensation, not signal —
    so we read only the non-derivative block here.
    """
    soup = BeautifulSoup(xml_text, "lxml-xml")
    if soup.find("ownershipDocument") is None:
        # Fall back to the HTML parser; some filings ship malformed XML.
        soup = BeautifulSoup(xml_text, "lxml")

    reporter = soup.find("reportingOwner")
    if reporter is None:
        return []

    filer_name = _txt(reporter.find("reportingOwnerId"), "rptOwnerName") or "Unknown"
    relationship = reporter.find("reportingOwnerRelationship")
    is_director = _to_bool(_txt(relationship, "isDirector"))
    is_officer = _to_bool(_txt(relationship, "isOfficer"))
    is_ten_percent = _to_bool(_txt(relationship, "isTenPercentOwner"))
    officer_title = _txt(relationship, "officerTitle")
    role = _classify_role(
        is_director=is_director,
        is_officer=is_officer,
        is_ten_percent_owner=is_ten_percent,
        officer_title=officer_title,
    )

    # Aggregate every footnote / explanation field for 10b5-1 detection.
    footnotes_text = " ".join(node.get_text(" ") for node in soup.find_all("footnote"))

    out: list[InsiderTransaction] = []
    for txn in soup.find_all("nonDerivativeTransaction"):
        date_str = _txt(txn, "transactionDate")
        code = _txt(txn.find("transactionCoding"), "transactionCode") or ""
        amounts = txn.find("transactionAmounts")
        shares = _to_float(_txt(amounts, "transactionShares"))
        price = _to_float(_txt(amounts, "transactionPricePerShare"))
        post = txn.find("postTransactionAmounts")
        post_holdings_raw = _txt(post, "sharesOwnedFollowingTransaction") if post else None
        post_holdings = float(post_holdings_raw) if post_holdings_raw else None

        if date_str is None:
            continue
        try:
            txn_date = date.fromisoformat(date_str)
        except ValueError:
            continue

        # Per-transaction footnote text + the file-level footnotes are both
        # checked for the 10b5-1 marker.
        local_footnotes = " ".join(node.get_text(" ") for node in txn.find_all("footnoteId"))
        is_plan = _is_10b5_1(footnotes_text + " " + local_footnotes)

        out.append(
            InsiderTransaction(
                accession=accession,
                filer_name=filer_name,
                role=role,
                is_director=is_director,
                is_officer=is_officer,
                is_ten_percent_owner=is_ten_percent,
                transaction_date=txn_date,
                code=code,
                shares=shares,
                price_per_share=price,
                value_usd=shares * price,
                post_holdings=post_holdings,
                is_10b5_1_plan=is_plan,
            )
        )
    return out


# --------------------------------------------------------------------------
# Fetch + summarize
# --------------------------------------------------------------------------


def fetch_insider_history(
    edgar: EdgarClient,
    cik: int,
    start: date,
    end: date,
) -> list[InsiderTransaction]:
    """Pull Form 4 / 4/A filings for a CIK between two dates, parse, and flatten.

    Uses ``EdgarClient.list_filings`` with the ``forms`` filter and skips any
    filing without a parseable XML primary document (legacy paper filings).
    """
    filings = edgar.list_filings(cik, forms=("4", "4/A"))
    out: list[InsiderTransaction] = []
    for f in filings:
        try:
            filed = date.fromisoformat(str(f.get("filingDate", "")))
        except ValueError:
            continue
        if filed < start or filed > end:
            continue
        accession = str(f.get("accessionNumber", ""))
        if not accession:
            continue
        primary_doc = f.get("primaryDocument")
        try:
            xml_text = edgar.fetch_form4_xml(cik, accession, primary_doc)
        except Exception as exc:  # noqa: BLE001 - network/4xx tolerated
            log.debug("Form 4 fetch failed for %s: %s", accession, exc)
            continue
        try:
            out.extend(parse_form4_xml(xml_text, accession))
        except Exception as exc:  # noqa: BLE001 - parse failures tolerated
            log.debug("Form 4 parse failed for %s: %s", accession, exc)
    return out


def summarize_insider_alignment(
    transactions: list[InsiderTransaction],
    *,
    as_of: date,
    lookback_months: int = 24,
    coordinated_buy_window_days: int = 60,
    coordinated_buy_min_distinct_insiders: int = 3,
    large_recent_sell_threshold_usd: float = 1_000_000.0,
    large_recent_sell_window_days: int = 90,
) -> InsiderAlignmentResult:
    """Aggregate Form 4 transactions into a Phil-Town-style alignment signal.

    The pass criterion follows the plan:
        * net open-market buys (excluding plan sells, tax withholding, gifts,
          grants, derivative exercises) is non-negative, AND
        * no large coordinated insider sells in the past 90 days (more than
          ``large_recent_sell_threshold_usd`` of non-plan sells).

    A ``coordinated_buy`` flag is also returned: 3+ distinct insiders making
    open-market buys within a 60-day window — Phil Town's "massive green flag".
    """
    lookback_start = as_of - timedelta(days=lookback_months * 30)
    in_window = [
        t for t in transactions
        if lookback_start <= t.transaction_date <= as_of
    ]

    by_role_net: dict[str, float] = {}
    buy_value = 0.0
    sell_value = 0.0
    buy_dates_by_filer: dict[str, list[date]] = {}

    for t in in_window:
        if t.code == OPEN_MARKET_BUY:
            buy_value += t.value_usd
            by_role_net[t.role] = by_role_net.get(t.role, 0.0) + t.value_usd
            buy_dates_by_filer.setdefault(t.filer_name, []).append(t.transaction_date)
        elif t.code == OPEN_MARKET_SELL and not t.is_10b5_1_plan:
            sell_value += t.value_usd
            by_role_net[t.role] = by_role_net.get(t.role, 0.0) - t.value_usd
        # Other codes (A, F, G, M, ...) are intentionally ignored.

    net = buy_value - sell_value

    # Coordinated buys: any 60-day window containing buys from ≥3 distinct insiders.
    all_buy_events: list[tuple[date, str]] = sorted(
        (d, name) for name, ds in buy_dates_by_filer.items() for d in ds
    )
    coordinated_count = 0
    coordinated_buy = False
    for i, (d_i, _) in enumerate(all_buy_events):
        window_end = d_i + timedelta(days=coordinated_buy_window_days)
        names = {
            name for d_j, name in all_buy_events[i:]
            if d_j <= window_end
        }
        if len(names) >= coordinated_buy_min_distinct_insiders:
            coordinated_buy = True
            coordinated_count = max(coordinated_count, len(names))

    # Large recent sells (90-day window ending at as_of, non-plan).
    recent_sell_floor = as_of - timedelta(days=large_recent_sell_window_days)
    recent_sells = sum(
        t.value_usd for t in in_window
        if t.code == OPEN_MARKET_SELL
        and not t.is_10b5_1_plan
        and t.transaction_date >= recent_sell_floor
    )
    has_large_recent = recent_sells > large_recent_sell_threshold_usd

    passes = net >= 0.0 and not has_large_recent
    rationale = (
        f"Window {lookback_start.isoformat()}..{as_of.isoformat()}: "
        f"buys ${buy_value/1e3:,.0f}k, sells ${sell_value/1e3:,.0f}k "
        f"(non-plan), net ${net/1e3:,.0f}k. "
        f"Coordinated-buy={coordinated_buy} ({coordinated_count} insiders); "
        f"large recent sells (last {large_recent_sell_window_days}d)="
        f"{has_large_recent} (${recent_sells/1e3:,.0f}k)."
    )

    return InsiderAlignmentResult(
        lookback_start=lookback_start,
        lookback_end=as_of,
        n_transactions=len(in_window),
        open_market_buy_value_usd=buy_value,
        open_market_sell_value_usd=sell_value,
        net_open_market_value_usd=net,
        coordinated_buy=coordinated_buy,
        coordinated_buy_count=coordinated_count,
        has_large_recent_sells=has_large_recent,
        by_role_net_usd=by_role_net,
        passes=passes,
        rationale=rationale,
    )
