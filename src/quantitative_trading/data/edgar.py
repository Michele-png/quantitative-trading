"""SEC EDGAR HTTP client with rate limiting, retries, and disk cache.

All SEC bulk endpoints are cached as JSON on disk. Subsequent reads are
local-only. Cache invalidation is manual (delete the file) — appropriate for a
historical-research workload where freshness < days does not matter.

References:
    * SEC fair access policy:   https://www.sec.gov/os/accessing-edgar-data
    * Submissions endpoint:     https://data.sec.gov/submissions/CIK{cik:010d}.json
    * Company facts endpoint:   https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json
    * Company tickers index:    https://www.sec.gov/files/company_tickers.json
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from quantitative_trading.config import get_config

log = logging.getLogger(__name__)


class EdgarError(RuntimeError):
    """Raised when EDGAR returns an error or unexpected payload."""


class _RateLimiter:
    """Simple token-bucket rate limiter, thread-safe.

    SEC's published cap is 10 requests/second; we stay under at 8 r/s by default.
    """

    def __init__(self, max_per_sec: int) -> None:
        self._max = max_per_sec
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque(maxlen=max_per_sec)

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if len(self._timestamps) < self._max:
                self._timestamps.append(now)
                return
            oldest = self._timestamps[0]
            elapsed = now - oldest
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
            self._timestamps.append(time.monotonic())


class EdgarClient:
    """SEC EDGAR client. Caches all responses to disk under cfg.edgar_cache_dir."""

    BASE_DATA = "https://data.sec.gov"
    BASE_WWW = "https://www.sec.gov"
    DEFAULT_RATE = 8

    def __init__(
        self,
        user_agent: str | None = None,
        cache_dir: Path | None = None,
        rate_limit_per_sec: int = DEFAULT_RATE,
    ) -> None:
        cfg = get_config()
        self._user_agent = user_agent or cfg.sec_user_agent
        self._cache_dir = cache_dir or cfg.edgar_cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        (self._cache_dir / "submissions").mkdir(exist_ok=True)
        (self._cache_dir / "facts").mkdir(exist_ok=True)
        (self._cache_dir / "filings").mkdir(exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self._user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self._rate = _RateLimiter(rate_limit_per_sec)

    @retry(
        retry=retry_if_exception_type((requests.RequestException, EdgarError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _get(self, url: str, *, headers: dict[str, str] | None = None) -> requests.Response:
        self._rate.acquire()
        log.debug("GET %s", url)
        resp = self._session.get(url, headers=headers, timeout=30)
        if resp.status_code == 429:
            raise EdgarError(f"Rate limited (429) on {url}")
        if resp.status_code >= 500:
            raise EdgarError(f"Server error {resp.status_code} on {url}")
        if resp.status_code == 404:
            return resp
        resp.raise_for_status()
        return resp

    def get_company_tickers(self, *, force_refresh: bool = False) -> dict[str, int]:
        """Return {TICKER -> CIK} mapping (uppercase tickers).

        SEC's master ticker list maps every active filer's primary ticker to its CIK.
        Cached locally; refresh weekly in practice.
        """
        cache_file = self._cache_dir / "company_tickers.json"
        if cache_file.exists() and not force_refresh:
            data = json.loads(cache_file.read_text())
        else:
            url = f"{self.BASE_WWW}/files/company_tickers.json"
            resp = self._get(url)
            data = resp.json()
            cache_file.write_text(json.dumps(data))
        out: dict[str, int] = {}
        for entry in data.values():
            ticker = str(entry["ticker"]).upper()
            cik = int(entry["cik_str"])
            out[ticker] = cik
        return out

    def get_company_tickers_with_names(
        self, *, force_refresh: bool = False
    ) -> list[dict[str, Any]]:
        """Return the raw SEC company_tickers entries with `ticker`, `cik_str`, `title`.

        Used by the CUSIP resolver's name-match fallback to look up tickers
        for issuers that OpenFIGI returns without a US-primary listing
        (a known gap for some smaller-cap US tickers).
        """
        cache_file = self._cache_dir / "company_tickers.json"
        if cache_file.exists() and not force_refresh:
            data = json.loads(cache_file.read_text())
        else:
            url = f"{self.BASE_WWW}/files/company_tickers.json"
            resp = self._get(url)
            data = resp.json()
            cache_file.write_text(json.dumps(data))
        return list(data.values())

    def get_cik(self, ticker: str) -> int:
        """Look up CIK for a ticker. Raises EdgarError if not found."""
        mapping = self.get_company_tickers()
        cik = mapping.get(ticker.upper())
        if cik is None:
            raise EdgarError(f"Ticker {ticker!r} not found in SEC company_tickers index.")
        return cik

    def get_submissions(self, cik: int, *, force_refresh: bool = False) -> dict[str, Any]:
        """Fetch raw submissions/CIK{cik:010d}.json (filings history)."""
        cik_str = f"{cik:010d}"
        cache_file = self._cache_dir / "submissions" / f"{cik_str}.json"
        if cache_file.exists() and not force_refresh:
            return json.loads(cache_file.read_text())
        url = f"{self.BASE_DATA}/submissions/CIK{cik_str}.json"
        resp = self._get(url)
        if resp.status_code == 404:
            raise EdgarError(f"No submissions found for CIK {cik}.")
        data = resp.json()
        cache_file.write_text(json.dumps(data))
        return data

    def get_company_facts(self, cik: int, *, force_refresh: bool = False) -> dict[str, Any]:
        """Fetch raw companyfacts/CIK{cik:010d}.json (full XBRL fact set)."""
        cik_str = f"{cik:010d}"
        cache_file = self._cache_dir / "facts" / f"{cik_str}.json"
        if cache_file.exists() and not force_refresh:
            return json.loads(cache_file.read_text())
        url = f"{self.BASE_DATA}/api/xbrl/companyfacts/CIK{cik_str}.json"
        resp = self._get(url)
        if resp.status_code == 404:
            raise EdgarError(f"No XBRL facts available for CIK {cik}.")
        data = resp.json()
        cache_file.write_text(json.dumps(data))
        return data

    def get_additional_submissions(
        self, cik: int, file_name: str, *, force_refresh: bool = False
    ) -> dict[str, Any]:
        """Fetch a paginated submissions file (filings.files[*].name)."""
        cache_file = self._cache_dir / "submissions" / file_name
        if cache_file.exists() and not force_refresh:
            return json.loads(cache_file.read_text())
        url = f"{self.BASE_DATA}/submissions/{file_name}"
        resp = self._get(url)
        if resp.status_code == 404:
            return {}
        data = resp.json()
        cache_file.write_text(json.dumps(data))
        return data

    def list_filings(
        self,
        cik: int,
        forms: Iterable[str] = ("10-K", "10-Q"),
        *,
        include_archived: bool = True,
    ) -> list[dict[str, Any]]:
        """Return a list of filings for the given forms, newest first.

        SEC's submissions endpoint returns only the most recent ~1000 filings
        in the "recent" block; older filings live in additional paginated files
        referenced by name in `filings.files`. Set `include_archived=False` to
        skip those (faster but misses older filings — needed for early backtests).

        Each entry has: accessionNumber, filingDate, reportDate, form, primaryDocument.
        """
        subs = self.get_submissions(cik)
        forms_set = {f.upper() for f in forms}

        def _extract(block: dict[str, Any]) -> list[dict[str, Any]]:
            accessions = block.get("accessionNumber", [])
            filing_dates = block.get("filingDate", [])
            report_dates = block.get("reportDate", [])
            forms_list = block.get("form", [])
            primary_docs = block.get("primaryDocument", [])
            out: list[dict[str, Any]] = []
            for accn, fd, rd, form, doc in zip(
                accessions, filing_dates, report_dates,
                forms_list, primary_docs, strict=False,
            ):
                if form.upper() not in forms_set:
                    continue
                out.append({
                    "accessionNumber": accn,
                    "filingDate": fd,
                    "reportDate": rd,
                    "form": form,
                    "primaryDocument": doc,
                    "cik": cik,
                })
            return out

        results = _extract(subs.get("filings", {}).get("recent", {}))

        if include_archived:
            for archived_file in subs.get("filings", {}).get("files", []):
                name = archived_file.get("name")
                if not name:
                    continue
                archived = self.get_additional_submissions(cik, name)
                # Older paginated files are flat dicts of arrays (no "recent" wrapper)
                results.extend(_extract(archived))

        # Newest first
        results.sort(key=lambda f: f["filingDate"], reverse=True)
        return results

    def fetch_filing_document(
        self,
        cik: int,
        accession: str,
        primary_document: str,
    ) -> str:
        """Download a filing's primary document (HTML or text). Cached on disk."""
        accn_clean = accession.replace("-", "")
        cache_file = (
            self._cache_dir / "filings" / f"{cik:010d}_{accn_clean}_{primary_document}"
        )
        if cache_file.exists():
            return cache_file.read_text(errors="replace")
        url = f"{self.BASE_WWW}/Archives/edgar/data/{cik}/{accn_clean}/{primary_document}"
        resp = self._get(url)
        if resp.status_code == 404:
            raise EdgarError(f"Filing document not found: {url}")
        text = resp.text
        cache_file.write_text(text, errors="replace")
        return text
