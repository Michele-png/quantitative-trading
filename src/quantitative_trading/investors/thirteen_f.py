"""SEC Form 13F-HR downloader and information-table parser.

Form 13F-HR is the quarterly long-equity holdings report filed by every
institutional manager with > $100M AUM. Each filing has two documents:

    1. **Primary cover page** (`primary_doc.xml`): metadata including
       `periodOfReport`, the filing manager's name, and a count of holdings.
    2. **Information table** (`informationtable.xml` or `*.xml` attachment):
       the actual list of holdings, one `<infoTable>` element per position.

Each holding has CUSIP, issuer name, title of class (e.g., "COM" for common
stock), value, and a share count.

Two value-units conventions exist:

    * **Pre-2023-Q4** (periodOfReport < 2023-12-31): `<value>` is reported in
      THOUSANDS of dollars.
    * **2023-Q4 and later**: `<value>` is in WHOLE DOLLARS. SEC changed this
      with the November 2023 amendment to Form 13F.

This module returns a normalized `value_usd` field in whole dollars regardless
of which convention the filing used.

Filename heuristic
------------------
The information-table XML attachment doesn't have a fixed filename across
filers. We fetch the filing's `index.json` to discover all attachments and
pick the XML file that is not the primary cover-page document.

Caching
-------
All HTTP responses are cached on disk by the underlying `EdgarClient`. Parsed
filings are NOT cached separately — re-parsing is cheap relative to the I/O.
"""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from lxml import etree

from quantitative_trading.data.edgar import EdgarClient, EdgarError

log = logging.getLogger(__name__)


# SEC reporting threshold change: periods on/after 2023-12-31 report value
# in whole dollars; earlier in thousands.
_VALUE_WHOLE_DOLLAR_FROM = date(2023, 12, 31)


@dataclass(frozen=True)
class ThirteenFHolding:
    """One position from a 13F-HR information table."""

    cusip: str                       # 9-character CUSIP, uppercase
    name_of_issuer: str
    title_of_class: str              # "COM", "CL A", "PUT", "CALL", etc.
    value_usd: float                 # always whole dollars (normalized)
    shares: int                      # share count from <sshPrnamt>
    shares_or_principal_type: str    # "SH" (shares) or "PRN" (principal $ for debt)
    put_call: str | None             # "Put", "Call", or None for stock


@dataclass(frozen=True)
class ThirteenFFiling:
    """One 13F-HR filing with all its holdings."""

    cik: int
    accession_number: str
    form: str                       # "13F-HR" or "13F-HR/A"
    filing_date: date
    period_of_report: date          # quarter-end date the filing covers
    holdings: tuple[ThirteenFHolding, ...]

    @property
    def is_amendment(self) -> bool:
        return self.form.endswith("/A")

    @property
    def quarter_end(self) -> date:
        """Alias for period_of_report (the standard 13F quarter-end)."""
        return self.period_of_report


# ----------------------------------------------------------------- Parser


def _strip_ns(tag: str) -> str:
    """Strip XML namespace from a tag: '{http://...}foo' -> 'foo'."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _normalize_text(s: str | None) -> str | None:
    """Collapse whitespace and decode HTML entities defensively.

    Some 13F filers double-encode XML special characters (`&` -> `&amp;amp;`),
    so a single decode-by-XML-parser leaves us with `&amp;` in the text. Decode
    once more to be robust. Whitespace is collapsed to single spaces because
    issuer names sometimes embed newlines from line-wrapped XML source.
    """
    if s is None:
        return None
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _normalize_cusip(s: str | None) -> str | None:
    """Normalize a CUSIP: uppercase, strip whitespace, left-pad to 9 chars.

    CUSIPs are formally 9-character alphanumeric (8-char issuer+issue + check
    digit). Some 13F filers strip leading zeros, so the same security appears
    as both `060505104` and `60505104` — we treat them as the same CUSIP by
    left-padding any sub-9-character string with `0`.
    """
    if s is None:
        return None
    cleaned = s.strip().upper().replace(" ", "")
    if not cleaned:
        return None
    # Pad to 9 chars; if longer than 9 (rare malformed entries), keep as-is.
    if len(cleaned) < 9:
        cleaned = cleaned.zfill(9)
    return cleaned


def _find_text(elem: Any, name: str) -> str | None:
    """Find a child element by local name (namespace-agnostic), return text."""
    for child in elem:
        if _strip_ns(child.tag) == name:
            return (child.text or "").strip() or None
    return None


def _find_child(elem: Any, name: str) -> Any | None:
    for child in elem:
        if _strip_ns(child.tag) == name:
            return child
    return None


def parse_information_table(xml_bytes: bytes, *, period_of_report: date) -> list[ThirteenFHolding]:
    """Parse a 13F information-table XML payload into normalized holdings.

    Schema-version-agnostic — uses local-name matching to handle namespace
    changes between the 2008 and 2023 schemas.
    """
    try:
        root = etree.fromstring(xml_bytes)  # noqa: S320 — SEC-trusted source
    except etree.XMLSyntaxError as exc:
        raise EdgarError(f"Failed to parse 13F XML: {exc}") from exc

    # Root may be <informationTable> directly, or wrapped in another element.
    if _strip_ns(root.tag) != "informationTable":
        # Look one level down.
        for child in root:
            if _strip_ns(child.tag) == "informationTable":
                root = child
                break

    value_multiplier = 1.0 if period_of_report >= _VALUE_WHOLE_DOLLAR_FROM else 1000.0

    out: list[ThirteenFHolding] = []
    for entry in root:
        if _strip_ns(entry.tag) != "infoTable":
            continue

        cusip = _normalize_cusip(_find_text(entry, "cusip"))
        name_of_issuer = _normalize_text(_find_text(entry, "nameOfIssuer"))
        title_of_class = _normalize_text(_find_text(entry, "titleOfClass"))
        raw_value = _find_text(entry, "value")
        put_call = _normalize_text(_find_text(entry, "putCall"))

        sh_block = _find_child(entry, "shrsOrPrnAmt")
        if sh_block is None:
            continue
        sh_amt = _find_text(sh_block, "sshPrnamt")
        sh_type = _normalize_text(_find_text(sh_block, "sshPrnamtType"))

        if cusip is None or name_of_issuer is None or raw_value is None or sh_amt is None:
            log.debug("Skipping incomplete <infoTable> in filing")
            continue

        try:
            value_raw = float(raw_value)
            shares = int(float(sh_amt))
        except ValueError:
            log.warning(
                "Non-numeric value/shares in 13F: value=%r shares=%r", raw_value, sh_amt
            )
            continue

        out.append(
            ThirteenFHolding(
                cusip=cusip,
                name_of_issuer=name_of_issuer,
                title_of_class=title_of_class or "",
                value_usd=value_raw * value_multiplier,
                shares=shares,
                shares_or_principal_type=(sh_type or "SH").upper(),
                put_call=put_call,
            )
        )
    return out


def parse_cover_period_of_report(xml_bytes: bytes) -> date | None:
    """Extract <periodOfReport> from a 13F cover-page XML (primary_doc.xml)."""
    try:
        root = etree.fromstring(xml_bytes)  # noqa: S320
    except etree.XMLSyntaxError:
        return None
    # Walk the tree looking for <periodOfReport>.
    for elem in root.iter():
        if _strip_ns(elem.tag) == "periodOfReport" and elem.text:
            txt = elem.text.strip()
            # SEC uses MM-DD-YYYY in cover pages.
            for fmt_re in (
                (r"(\d{2})-(\d{2})-(\d{4})", lambda m: date(int(m[3]), int(m[1]), int(m[2]))),
                (r"(\d{4})-(\d{2})-(\d{2})", lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
            ):
                m = re.match(fmt_re[0], txt)
                if m:
                    try:
                        return fmt_re[1](m)
                    except ValueError:
                        continue
    return None


# ----------------------------------------------------------------- Client


class ThirteenFClient:
    """Fetch and parse 13F-HR filings for a CIK."""

    def __init__(self, edgar_client: EdgarClient) -> None:
        self._edgar = edgar_client

    def _fetch_index_json(self, cik: int, accession: str) -> dict[str, Any]:
        """Fetch the EDGAR filing index (lists all attachments)."""
        index_text = self._edgar.fetch_filing_document(cik, accession, "index.json")
        return json.loads(index_text)

    def _identify_information_table_filename(
        self, index_payload: dict[str, Any]
    ) -> str | None:
        """Find the information-table XML attachment in a filing index.

        Heuristic: the cover page is `primary_doc.xml`. The information table
        is the OTHER `*.xml` file. If multiple non-cover XMLs exist, prefer
        the one whose name contains 'info' or 'table'.
        """
        items = index_payload.get("directory", {}).get("item", [])
        xml_files = [
            it["name"] for it in items
            if it.get("name", "").lower().endswith(".xml")
            and it.get("name", "").lower() != "primary_doc.xml"
        ]
        if not xml_files:
            return None
        if len(xml_files) == 1:
            return xml_files[0]
        # Prefer informational names if multiple exist.
        for preferred in ("infotable", "informationtable", "info_table"):
            for f in xml_files:
                if preferred in f.lower():
                    return f
        # Fallback: first non-cover XML alphabetically (deterministic).
        return sorted(xml_files)[0]

    def fetch_filing(
        self,
        cik: int,
        accession_number: str,
        *,
        filing_date: date,
        form: str,
    ) -> ThirteenFFiling | None:
        """Fetch and parse one 13F-HR filing.

        Returns None if the filing cannot be parsed (e.g., notice-only
        filings that have no information table).
        """
        # Find the information-table XML via the filing index.
        try:
            index_payload = self._fetch_index_json(cik, accession_number)
        except EdgarError as exc:
            log.warning("No index for %s/%s: %s", cik, accession_number, exc)
            return None

        info_table_name = self._identify_information_table_filename(index_payload)
        if info_table_name is None:
            # Notice-only 13F-NT or weirdly-structured filing — skip.
            log.debug("No information table in %s/%s", cik, accession_number)
            return None

        # Fetch cover page for periodOfReport.
        try:
            cover_text = self._edgar.fetch_filing_document(
                cik, accession_number, "primary_doc.xml"
            )
        except EdgarError:
            cover_text = ""
        period = parse_cover_period_of_report(cover_text.encode()) if cover_text else None
        if period is None:
            # Fall back to the filing date's quarter-end (this is approximate).
            log.warning(
                "No periodOfReport in cover for %s/%s; falling back to filing-date quarter",
                cik, accession_number,
            )
            period = _quarter_end_at_or_before(filing_date)

        # Fetch and parse the information table.
        try:
            table_text = self._edgar.fetch_filing_document(
                cik, accession_number, info_table_name
            )
        except EdgarError as exc:
            log.warning("No info-table %s in %s/%s: %s",
                        info_table_name, cik, accession_number, exc)
            return None

        holdings = parse_information_table(table_text.encode(), period_of_report=period)

        return ThirteenFFiling(
            cik=cik,
            accession_number=accession_number,
            form=form,
            filing_date=filing_date,
            period_of_report=period,
            holdings=tuple(holdings),
        )

    def fetch_all_filings(self, cik: int) -> list[ThirteenFFiling]:
        """Fetch ALL 13F-HR (and 13F-HR/A) filings for a CIK.

        Returns filings sorted by period_of_report ascending (oldest first),
        which is the natural order for first-ever-appearance detection.
        """
        meta = self._edgar.list_filings(cik, forms=("13F-HR", "13F-HR/A"))
        out: list[ThirteenFFiling] = []
        for f in meta:
            filing_date = date.fromisoformat(f["filingDate"])
            parsed = self.fetch_filing(
                cik=cik,
                accession_number=f["accessionNumber"],
                filing_date=filing_date,
                form=f["form"],
            )
            if parsed is not None:
                out.append(parsed)
        out.sort(key=lambda x: (x.period_of_report, x.filing_date))
        return out


# ----------------------------------------------------------------- Helpers


def _quarter_end_at_or_before(d: date) -> date:
    """Return the calendar quarter-end on or before `d`."""
    quarter_ends = [date(d.year, m, 31 if m in (3, 12) else 30) for m in (3, 6, 9, 12)]
    # Adjust for month days
    quarter_ends = [
        date(d.year, 3, 31),
        date(d.year, 6, 30),
        date(d.year, 9, 30),
        date(d.year, 12, 31),
    ]
    candidates = [q for q in quarter_ends if q <= d]
    if candidates:
        return max(candidates)
    return date(d.year - 1, 12, 31)
