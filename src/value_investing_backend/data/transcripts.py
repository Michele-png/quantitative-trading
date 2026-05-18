"""Earnings-call transcript providers.

Earnings call transcripts are NOT a first-class SEC API. Two realistic sources:

1. **Financial Modeling Prep** — paid commercial API with structured transcripts
   and full history. The free tier rate-limits aggressively (~250 req/day) and
   only covers recent quarters, but it's enough for ad-hoc screening of a few
   firms. Used here as the primary provider when ``FMP_API_KEY`` is set.

2. **SEC 8-K Exhibit 99.x** — companies often attach the prepared remarks +
   Q&A as Exhibit 99.1 or 99.2 to their 8-K "Item 2.02 Results of Operations"
   filings. Coverage is patchy (40-60% of S&P 500, worse pre-2015) and the
   text is unstructured (no speaker tags), but it's free and reliably reachable
   through the existing ``EdgarClient``. Used as the fallback.

The ``TranscriptProvider`` protocol abstracts the difference so the management
pipeline can swap providers (FMP, SEC, Quartr, API Ninjas, ...) without
touching the agent. ``DefaultTranscriptProvider`` chains FMP → SEC.

Caching: each successful fetch is written to disk under
``cfg.transcripts_cache_dir`` keyed by ``(provider, ticker, year, quarter)``,
so repeated calls during a weekly refresh cost zero API requests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import requests
from bs4 import BeautifulSoup

from value_investing_backend.config import get_config

if TYPE_CHECKING:
    from value_investing_backend.data.edgar import EdgarClient


log = logging.getLogger(__name__)


# Heuristic length floor for "this looks like a transcript, not a press release".
_TRANSCRIPT_MIN_CHARS = 5_000


@dataclass(frozen=True)
class EarningsTranscript:
    """One earnings call transcript, normalized across providers."""

    ticker: str
    fiscal_year: int
    fiscal_quarter: int  # 1..4
    call_date: date | None
    source: str  # "fmp" | "sec_8k" | ...
    text: str

    def __len__(self) -> int:
        return len(self.text)


class TranscriptProvider(Protocol):
    """Pluggable source for earnings-call transcripts."""

    name: str

    def get_transcripts(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> list[EarningsTranscript]:
        ...


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------


def _cache_path(cache_dir: Path, provider: str, ticker: str, year: int, quarter: int) -> Path:
    key = f"{provider}|{ticker.upper()}|{year}Q{quarter}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return cache_dir / f"{provider}_{ticker.upper()}_{year}Q{quarter}_{digest}.json"


def _save_to_cache(cache_dir: Path, t: EarningsTranscript) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, t.source, t.ticker, t.fiscal_year, t.fiscal_quarter)
    payload = asdict(t)
    payload["call_date"] = t.call_date.isoformat() if t.call_date else None
    path.write_text(json.dumps(payload))


def _load_from_cache(
    cache_dir: Path, provider: str, ticker: str, year: int, quarter: int
) -> EarningsTranscript | None:
    path = _cache_path(cache_dir, provider, ticker, year, quarter)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return EarningsTranscript(
        ticker=raw["ticker"],
        fiscal_year=raw["fiscal_year"],
        fiscal_quarter=raw["fiscal_quarter"],
        call_date=date.fromisoformat(raw["call_date"]) if raw["call_date"] else None,
        source=raw["source"],
        text=raw["text"],
    )


def _quarters_in_window(start: date, end: date) -> list[tuple[int, int]]:
    """Enumerate (year, quarter) pairs whose quarter-end falls in [start, end]."""
    out: list[tuple[int, int]] = []
    for y in range(start.year, end.year + 1):
        for q in (1, 2, 3, 4):
            q_end_month = q * 3
            q_end = date(y, q_end_month, 28)  # 28th avoids month-length issues
            if start <= q_end <= end:
                out.append((y, q))
    return out


# --------------------------------------------------------------------------
# Financial Modeling Prep provider
# --------------------------------------------------------------------------


class FmpTranscriptProvider:
    """Earnings transcripts via Financial Modeling Prep.

    Endpoint: ``GET /api/v3/earning_call_transcript/{ticker}?quarter=Q&year=Y&apikey=...``

    Failure modes are handled gracefully — the management pipeline runs on
    whatever transcripts it gets, even if some quarters are missing. We:

        * cache hits to disk so repeated calls cost zero requests,
        * log and continue on 402 (free-tier exceeded) or 429 (rate limited),
        * return ``[]`` rather than raise on network errors.
    """

    name = "fmp"
    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(
        self,
        api_key: str,
        cache_dir: Path | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("FmpTranscriptProvider requires a non-empty api_key.")
        self._api_key = api_key
        self._cache_dir = cache_dir or get_config().transcripts_cache_dir
        self._timeout = timeout
        self._session = session or requests.Session()
        self._rate_limited = False  # latches True after first 402/429

    def get_transcripts(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> list[EarningsTranscript]:
        out: list[EarningsTranscript] = []
        for year, quarter in _quarters_in_window(start, end):
            cached = _load_from_cache(self._cache_dir, self.name, ticker, year, quarter)
            if cached is not None:
                out.append(cached)
                continue
            if self._rate_limited:
                continue
            t = self._fetch_one(ticker, year, quarter)
            if t is not None:
                _save_to_cache(self._cache_dir, t)
                out.append(t)
        return out

    def _fetch_one(self, ticker: str, year: int, quarter: int) -> EarningsTranscript | None:
        url = f"{self.BASE_URL}/earning_call_transcript/{ticker.upper()}"
        params = {"quarter": quarter, "year": year, "apikey": self._api_key}
        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
        except requests.RequestException as exc:
            log.warning("FMP transcript fetch failed for %s %sQ%s: %s",
                        ticker, year, quarter, exc)
            return None

        if response.status_code in (402, 429):
            log.warning(
                "FMP rate-limit hit (status %s) on %s %sQ%s — skipping remaining requests.",
                response.status_code, ticker, year, quarter,
            )
            self._rate_limited = True
            return None
        if response.status_code != 200:
            log.warning("FMP returned %s for %s %sQ%s",
                        response.status_code, ticker, year, quarter)
            return None

        try:
            payload = response.json()
        except ValueError:
            return None

        # FMP returns a list. Empty list = no transcript for that quarter.
        if not isinstance(payload, list) or not payload:
            return None
        item = payload[0]
        text = (item.get("content") or "").strip()
        if not text:
            return None
        call_date_str = item.get("date")
        try:
            call_date = (
                date.fromisoformat(call_date_str.split(" ")[0])
                if call_date_str else None
            )
        except (AttributeError, ValueError):
            call_date = None
        return EarningsTranscript(
            ticker=ticker.upper(),
            fiscal_year=int(item.get("year", year)),
            fiscal_quarter=int(item.get("quarter", quarter)),
            call_date=call_date,
            source=self.name,
            text=text,
        )


# --------------------------------------------------------------------------
# SEC 8-K Exhibit 99.x fallback
# --------------------------------------------------------------------------


_EXHIBIT_99_PATTERN = re.compile(r"^ex-?99[._-]?\d*\.htm$|^ex99[._-]?\d*\.htm$", re.IGNORECASE)


class Sec8KTranscriptProvider:
    """Best-effort earnings transcripts via SEC 8-K Exhibit 99.x.

    Finds 8-K filings in the date window, downloads the filing index, picks
    Exhibit 99.x attachments, and returns whichever look long enough to
    plausibly be a transcript. The text is unstructured (no speaker tags),
    so this is a strict fallback when no commercial provider is configured.

    Quarter assignment is heuristic: we map each 8-K's ``filingDate`` to the
    most recently completed fiscal quarter (filings about quarterly results
    appear shortly after the quarter close). Good enough for "last N quarters"
    style requests.
    """

    name = "sec_8k"

    def __init__(
        self,
        edgar_client: EdgarClient,
        cache_dir: Path | None = None,
    ) -> None:
        self._edgar = edgar_client
        self._cache_dir = cache_dir or get_config().transcripts_cache_dir

    def get_transcripts(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> list[EarningsTranscript]:
        try:
            cik = self._edgar.get_cik(ticker)
        except KeyError:
            log.debug("SEC 8-K provider: no CIK for %s", ticker)
            return []
        try:
            filings = self._edgar.list_filings(cik, forms=("8-K", "8-K/A"))
        except Exception as exc:  # noqa: BLE001
            log.warning("SEC 8-K filings list failed for %s: %s", ticker, exc)
            return []

        out: list[EarningsTranscript] = []
        for f in filings:
            try:
                filed = date.fromisoformat(str(f.get("filingDate", "")))
            except ValueError:
                continue
            if filed < start or filed > end:
                continue
            year, quarter = _quarter_for_filing_date(filed)
            cached = _load_from_cache(self._cache_dir, self.name, ticker, year, quarter)
            if cached is not None:
                out.append(cached)
                continue
            text = self._fetch_exhibit_99(cik, str(f.get("accessionNumber", "")))
            if not text or len(text) < _TRANSCRIPT_MIN_CHARS:
                continue
            t = EarningsTranscript(
                ticker=ticker.upper(),
                fiscal_year=year,
                fiscal_quarter=quarter,
                call_date=filed,
                source=self.name,
                text=text,
            )
            _save_to_cache(self._cache_dir, t)
            out.append(t)
        return out

    def _fetch_exhibit_99(self, cik: int, accession: str) -> str | None:
        if not accession:
            return None
        clean = accession.replace("-", "")
        index_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include"
        )
        # We can avoid the index entirely by using the filing-summary JSON.
        try:
            summary = requests.get(
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{clean}/index.json",
                headers={
                    "User-Agent": get_config().sec_user_agent,
                    "Accept-Encoding": "gzip, deflate",
                },
                timeout=30,
            )
            summary.raise_for_status()
        except requests.RequestException as exc:
            log.debug("8-K index fetch failed (%s) for %s: %s", index_url, accession, exc)
            return None

        try:
            items = summary.json().get("directory", {}).get("item", [])
        except ValueError:
            return None

        for item in items:
            name = (item.get("name") or "").lower()
            if not _EXHIBIT_99_PATTERN.match(name):
                continue
            try:
                html = self._edgar.fetch_filing_document(cik, accession, item["name"])
            except Exception as exc:  # noqa: BLE001
                log.debug("8-K exhibit fetch failed for %s/%s: %s",
                          accession, item["name"], exc)
                continue
            text = _strip_html(html)
            if len(text) >= _TRANSCRIPT_MIN_CHARS:
                return text
        return None


def _strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def _quarter_for_filing_date(filed: date) -> tuple[int, int]:
    """Map an 8-K filing date to the most-recently-completed calendar quarter.

    Earnings 8-Ks land within ~6 weeks of quarter close. So an 8-K filed in
    May 2024 most likely discusses Q1 2024 results.
    """
    if filed.month <= 2:
        return filed.year - 1, 4
    if filed.month <= 5:
        return filed.year, 1
    if filed.month <= 8:
        return filed.year, 2
    if filed.month <= 11:
        return filed.year, 3
    return filed.year, 4


# --------------------------------------------------------------------------
# Default provider chain
# --------------------------------------------------------------------------


class DefaultTranscriptProvider:
    """Try FMP first, fall back to SEC 8-K, deduplicate by (year, quarter).

    If the FMP key is absent we skip straight to SEC. If both providers fail
    we return an empty list — the management pipeline will run with whatever
    transcripts it got (possibly none) rather than crash.
    """

    name = "chain"

    def __init__(
        self,
        edgar_client: EdgarClient,
        fmp_api_key: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        cfg = get_config()
        self._cache_dir = cache_dir or cfg.transcripts_cache_dir
        self._providers: list[TranscriptProvider] = []
        key = fmp_api_key if fmp_api_key is not None else cfg.fmp_api_key
        if key:
            self._providers.append(FmpTranscriptProvider(key, cache_dir=self._cache_dir))
        self._providers.append(
            Sec8KTranscriptProvider(edgar_client, cache_dir=self._cache_dir)
        )

    def get_transcripts(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> list[EarningsTranscript]:
        seen: set[tuple[int, int]] = set()
        out: list[EarningsTranscript] = []
        for provider in self._providers:
            for t in provider.get_transcripts(ticker, start, end):
                key = (t.fiscal_year, t.fiscal_quarter)
                if key in seen:
                    continue
                seen.add(key)
                out.append(t)
        out.sort(key=lambda x: (x.fiscal_year, x.fiscal_quarter))
        return out
