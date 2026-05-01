"""Minimal SEC EDGAR client."""

from __future__ import annotations

from typing import Any

import requests

from quantitative_trading.config import get_config


class EdgarClient:
    """Fetch CIKs, company facts, filings, and filing documents from SEC EDGAR."""

    BASE_URL = "https://data.sec.gov"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

    def __init__(self) -> None:
        cfg = get_config()
        self._headers = {"User-Agent": cfg.sec_user_agent, "Accept-Encoding": "gzip, deflate"}
        self._ticker_map: dict[str, int] | None = None

    def get_cik(self, ticker: str) -> int:
        """Return the CIK for a ticker."""
        ticker_map = self._load_ticker_map()
        symbol = ticker.upper()
        if symbol not in ticker_map:
            raise KeyError(f"No CIK found for ticker {ticker!r}.")
        return ticker_map[symbol]

    def get_company_facts(self, cik: int) -> dict[str, Any]:
        """Fetch SEC company-facts JSON for a CIK."""
        response = requests.get(
            f"{self.BASE_URL}/api/xbrl/companyfacts/CIK{cik:010d}.json",
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def list_filings(self, cik: int) -> list[dict[str, Any]]:
        """List recent submissions for a CIK as dictionaries."""
        response = requests.get(
            f"{self.BASE_URL}/submissions/CIK{cik:010d}.json",
            headers=self._headers,
            timeout=30,
        )
        response.raise_for_status()
        recent = response.json().get("filings", {}).get("recent", {})
        keys = list(recent)
        n_rows = len(recent.get(keys[0], [])) if keys else 0
        return [{key: recent[key][idx] for key in keys} for idx in range(n_rows)]

    def fetch_filing_document(self, cik: int, accession: str, primary_document: str) -> str:
        """Fetch the primary filing document text."""
        clean_accession = accession.replace("-", "")
        url = f"{self.ARCHIVES_URL}/{cik}/{clean_accession}/{primary_document}"
        response = requests.get(url, headers=self._headers, timeout=30)
        response.raise_for_status()
        return response.text

    def _load_ticker_map(self) -> dict[str, int]:
        if self._ticker_map is None:
            response = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=self._headers,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            self._ticker_map = {
                str(item["ticker"]).upper(): int(item["cik_str"])
                for item in payload.values()
            }
        return self._ticker_map
