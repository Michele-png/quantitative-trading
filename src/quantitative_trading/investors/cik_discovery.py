"""Discovery utility for investment-adviser CIKs.

Most of the audited entities (Pabrai/Dalal Street, Himalaya, Akre Capital, etc.)
are investment advisers, not public-company 10-K filers, so they do not appear
in the SEC `company_tickers.json` map that `EdgarClient.get_cik` uses. They
do, however, file 13F-HR through EDGAR and are searchable by entity name.

This module wraps EDGAR's company-search ATOM endpoint:

    https://www.sec.gov/cgi-bin/browse-edgar
        ?action=getcompany&company=<name>&type=13F-HR&output=atom

and returns the candidate `(cik, entity_name)` pairs that match a query.

Used once during setup to verify the static CIK map in `investor_universe.py`;
not part of the runtime audit pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from quantitative_trading.data.edgar import EdgarClient

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityMatch:
    """One candidate match from EDGAR company search."""

    cik: int
    entity_name: str


# Single-entity-resolution path: <company-info><cik>...</cik><conformed-name>...</conformed-name>
_SINGLE_CIK_RE = re.compile(
    r'<company-info>.*?<cik>(\d+)</cik>.*?<conformed-name>([^<]+)</conformed-name>',
    re.DOTALL | re.IGNORECASE,
)
# Multi-entity path: each <entry> contains a CIK reference inside an href
_MULTI_CIK_RE = re.compile(r'CIK=(\d{10})', re.IGNORECASE)
_MULTI_TITLE_RE = re.compile(r'<title>([^<]+)</title>', re.IGNORECASE)
_ENTRY_RE = re.compile(r'<entry>(.*?)</entry>', re.DOTALL | re.IGNORECASE)


def search_filers_by_name(
    edgar: EdgarClient,
    name_query: str,
    *,
    form: str = "13F-HR",
    max_results: int = 20,
) -> list[EntityMatch]:
    """Search EDGAR for filers whose entity name matches `name_query`.

    Uses the public ATOM endpoint at /cgi-bin/browse-edgar. The endpoint has
    two response shapes:

    * **Single-entity resolution**: when `name_query` matches exactly one
      filer prefix, EDGAR returns that filer's filings page with a top-level
      `<company-info><cik>...</cik><conformed-name>...</conformed-name></company-info>`
      block. The `<entry>` blocks below are individual filings, not entities.

    * **Multi-entity resolution**: when multiple filers match, each `<entry>`
      represents one entity match with its CIK in a query-string href.

    This function returns the entity matches in either case (length 1 in the
    common single-resolution case). Pass the most distinctive prefix possible
    to avoid mid-list noise.
    """
    url = (
        f"{edgar.BASE_WWW}/cgi-bin/browse-edgar"
        f"?action=getcompany&company={name_query.replace(' ', '+')}"
        f"&type={form}&dateb=&owner=include&count={max_results}&output=atom"
    )
    resp = edgar._get(url)  # noqa: SLF001 — internal reuse of rate-limited HTTP
    if resp.status_code != 200:
        log.warning("EDGAR search returned %d for %r", resp.status_code, name_query)
        return []
    body = resp.text

    # Path 1: single-entity resolution.
    single = _SINGLE_CIK_RE.search(body)
    if single is not None:
        cik = int(single.group(1))
        name = single.group(2).strip()
        return [EntityMatch(cik=cik, entity_name=name)]

    # Path 2: multi-entity resolution. Deduplicate by CIK (entries can repeat
    # within paginated filings views).
    seen: dict[int, str] = {}
    for entry in _ENTRY_RE.findall(body):
        cik_match = _MULTI_CIK_RE.search(entry)
        name_match = _MULTI_TITLE_RE.search(entry)
        if cik_match is None or name_match is None:
            continue
        cik = int(cik_match.group(1))
        if cik in seen:
            continue
        title = re.sub(r"\s*CIK#\d+.*$", "", name_match.group(1).strip()).strip()
        seen[cik] = title
    return [EntityMatch(cik=c, entity_name=n) for c, n in seen.items()]
