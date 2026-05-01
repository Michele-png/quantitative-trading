"""CUSIP -> (ticker, CIK, security type) resolver with dual-source verification.

Per audit plan section 10 item 5, every CUSIP appearing in an elite 13F is resolved
through TWO sources and disagreements are flagged for manual review. Rationale:
typical CUSIP-mapping error rate at 10y is 1-3% (mergers, spin-offs, share-class
splits), and on n ~ 30-60 evaluable buys, a single misrouted ticker swings
per-criterion pass-rates by 3-7 ppt.

Sources
-------
1. **Primary: OpenFIGI** (https://www.openfigi.com) - industry-standard
   identifier mapping. Free tier without API key allows 25 batch requests
   per minute (10 CUSIPs per batch). Returns ticker, FIGI, security type
   (Common Stock / ADR / Preferred / Warrant / etc.), and exchange code.

2. **Cross-check: SEC company_tickers.json** - the canonical SEC map of
   public-issuer ticker -> CIK. We verify that OpenFIGI's ticker is in this
   map; if so, we get the CIK for free (used by the existing PIT data layer).

Resolution outcomes
-------------------
* `verified`        - OpenFIGI ticker is in SEC company_tickers; high confidence.
* `openfigi_only`   - OpenFIGI returned a ticker but it's not in SEC's map.
                      Usually means ADR, foreign listing, or de-listed entity.
                      Routed to `non_evaluable` per audit plan section 5.
* `unresolved`      - OpenFIGI returned no match. Manual review required.
* `manual_override` - Resolution comes from `cusip_overrides.json`.

Caching
-------
Each successful resolution is written to a single JSON file at
`data/investors/cusip_cache.json`. Cache is permanent; delete the file to
force re-resolution.

Disagreement log
----------------
Cases flagged as `openfigi_only` or `unresolved` are also written to
`data/investors/cusip_disagreements.csv` for human review. The audit pipeline
reads `cusip_overrides.json` (if present) to apply manual decisions.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests

from quantitative_trading.config import get_config
from quantitative_trading.data.edgar import EdgarClient

log = logging.getLogger(__name__)


OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
OPENFIGI_BATCH_SIZE = 10        # Free-tier max per request
OPENFIGI_REQUESTS_PER_MIN = 20  # Stay under the 25/min free-tier cap


# Security types that are evaluable for the Big 5 audit. Anything else
# (ADRs, preferred stock, warrants, units, debt) routes to non_evaluable.
EVALUABLE_SECURITY_TYPES: frozenset[str] = frozenset({
    "Common Stock",
})


@dataclass(frozen=True)
class CusipResolution:
    """One CUSIP's resolution with provenance."""

    cusip: str
    ticker: str | None
    cik: int | None
    issuer_name: str | None
    security_type: str | None     # OpenFIGI: "Common Stock", "ADR", etc.
    exchange: str | None          # OpenFIGI: "US", "UN", "UW", etc.
    source: str                   # "verified" | "openfigi_only" | "unresolved" | "manual_override"

    @property
    def is_resolved(self) -> bool:
        return self.ticker is not None

    @property
    def is_verified_against_sec(self) -> bool:
        """OpenFIGI's ticker is also in SEC company_tickers (we have a CIK)."""
        return self.source in ("verified", "manual_override") and self.cik is not None

    @property
    def is_evaluable_security_type(self) -> bool:
        """Routes to evaluable per audit plan section 5 (Common Stock only by default)."""
        return self.security_type in EVALUABLE_SECURITY_TYPES


# -------------------------------------------------------- Rate-limited OpenFIGI


class _OpenFigiRateLimiter:
    """Token-bucket: at most N requests per 60 seconds, thread-safe."""

    def __init__(self, max_per_min: int) -> None:
        self._max = max_per_min
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque(maxlen=max_per_min)

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if len(self._timestamps) < self._max:
                self._timestamps.append(now)
                return
            oldest = self._timestamps[0]
            elapsed = now - oldest
            if elapsed < 60.0:
                time.sleep(60.0 - elapsed)
            self._timestamps.append(time.monotonic())


# -------------------------------------------------------- Resolver


class CusipResolver:
    """Dual-source CUSIP resolver with on-disk cache and manual-override support."""

    def __init__(
        self,
        edgar_client: EdgarClient,
        *,
        cache_dir: Path | None = None,
        openfigi_api_key: str | None = None,
    ) -> None:
        cfg = get_config()
        self._edgar = edgar_client
        base = cache_dir or (cfg.data_dir / "investors")
        base.mkdir(parents=True, exist_ok=True)
        self._cache_file = base / "cusip_cache.json"
        self._disagreements_file = base / "cusip_disagreements.csv"
        self._overrides_file = base / "cusip_overrides.json"

        self._cache: dict[str, CusipResolution] = self._load_cache()
        self._overrides: dict[str, dict] = self._load_overrides()
        self._rate = _OpenFigiRateLimiter(OPENFIGI_REQUESTS_PER_MIN)
        self._session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if openfigi_api_key:
            headers["X-OPENFIGI-APIKEY"] = openfigi_api_key
        self._session.headers.update(headers)

        # CIK lookup map, populated lazily.
        self._sec_ticker_to_cik: dict[str, int] | None = None
        # Normalized-issuer-name -> (ticker, cik) for the name-match fallback.
        self._sec_name_to_ticker_cik: dict[str, tuple[str, int]] | None = None

    # --------------------------------------------------- Cache I/O

    def _load_cache(self) -> dict[str, CusipResolution]:
        if not self._cache_file.exists():
            return {}
        raw = json.loads(self._cache_file.read_text())
        return {k: CusipResolution(**v) for k, v in raw.items()}

    def _save_cache(self) -> None:
        serializable = {k: asdict(v) for k, v in self._cache.items()}
        self._cache_file.write_text(json.dumps(serializable, indent=2, sort_keys=True))

    def _load_overrides(self) -> dict[str, dict]:
        if not self._overrides_file.exists():
            return {}
        return json.loads(self._overrides_file.read_text())

    # --------------------------------------------------- SEC ticker map

    def _ticker_to_cik(self, ticker: str) -> int | None:
        if self._sec_ticker_to_cik is None:
            self._sec_ticker_to_cik = self._edgar.get_company_tickers()
        return self._sec_ticker_to_cik.get(ticker.upper())

    @staticmethod
    def _normalize_issuer_name(name: str) -> str:
        """Normalize issuer name for SEC name-match fallback.

        Strips ONLY legal-form suffixes (CORP, INC, LTD, LP, LLC, PLC, CO,
        COMPANY) and the trailing "NEW" / "OLD" relisting markers. Keeps
        industry/scope terms (ENERGY, RESOURCES, INTERNATIONAL, etc.) because
        stripping them causes too many name collisions in SEC's index.
        Expands common 13F abbreviations.
        """
        s = name.upper()
        # Common 13F abbreviation expansions.
        s = re.sub(r"\bINTL\b", "INTERNATIONAL", s)
        s = re.sub(r"\bMGMT\b", "MANAGEMENT", s)
        s = re.sub(r"\bASSOC\b", "ASSOCIATES", s)
        s = re.sub(r"\bPPTYS\b", "PROPERTIES", s)
        s = re.sub(r"\bMANAGMT\b", "MANAGEMENT", s)
        # Strip relisting markers (e.g., "CONSOL ENERGY INC NEW" -> "CONSOL ENERGY INC")
        s = re.sub(r"\b(NEW|OLD|HLDG|HLDGS)\b", "", s)
        # Strip ONLY legal-form suffixes (industry words like ENERGY/RESOURCES kept).
        for suffix in (
            "CORPORATION", "CORP", "INCORPORATED", "INC",
            "LIMITED", "LTD", "LP", "LLC", "PLC",
            "COMPANY", "CO",
        ):
            s = re.sub(rf"\b{suffix}\b", "", s)
        # Strip non-alphanumerics and collapse whitespace.
        s = re.sub(r"[^A-Z0-9 ]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _name_to_ticker_cik(self, issuer_name: str) -> tuple[str, int] | None:
        """SEC name-match fallback: try to find a unique SEC ticker for `issuer_name`.

        Returns (ticker, cik) on a unique match, else None. The match is
        deliberately strict (exact normalized-name equality) — fuzzy matching
        risks routing the wrong CIK.
        """
        if self._sec_name_to_ticker_cik is None:
            entries = self._edgar.get_company_tickers_with_names()
            mapping: dict[str, tuple[str, int]] = {}
            collisions: set[str] = set()
            for entry in entries:
                title = entry.get("title", "")
                if not title:
                    continue
                key = self._normalize_issuer_name(title)
                ticker = str(entry.get("ticker", "")).upper()
                cik = int(entry.get("cik_str", 0))
                if not ticker or not cik or not key:
                    continue
                if key in mapping and mapping[key] != (ticker, cik):
                    collisions.add(key)
                else:
                    mapping[key] = (ticker, cik)
            # Drop ambiguous keys (multiple tickers map to the same normalized name).
            for k in collisions:
                mapping.pop(k, None)
            self._sec_name_to_ticker_cik = mapping
            log.info(
                "Built SEC name-match index: %d unique normalized names "
                "(%d collisions dropped)",
                len(mapping), len(collisions),
            )
        key = self._normalize_issuer_name(issuer_name)
        return self._sec_name_to_ticker_cik.get(key)

    # --------------------------------------------------- OpenFIGI

    def _query_openfigi_batch(self, cusips: list[str]) -> dict[str, dict | None]:
        """POST one batch (<= 10 CUSIPs) to OpenFIGI and return CUSIP -> first match."""
        if not cusips:
            return {}
        payload = [
            {"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in cusips
        ]
        self._rate.acquire()
        for attempt in range(3):
            resp = self._session.post(OPENFIGI_URL, json=payload, timeout=30)
            if resp.status_code == 429:
                # Free tier exhausted within the minute; wait and retry.
                wait = 30.0 * (attempt + 1)
                log.warning("OpenFIGI 429; sleeping %.1fs", wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                log.warning("OpenFIGI %d; retrying", resp.status_code)
                time.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            results = resp.json()
            assert len(results) == len(cusips)
            out: dict[str, dict | None] = {}
            for cusip, entry in zip(cusips, results, strict=True):
                if "data" in entry and entry["data"]:
                    # Prefer Common Stock; otherwise first entry.
                    for d in entry["data"]:
                        if d.get("securityType") == "Common Stock":
                            out[cusip] = d
                            break
                    else:
                        out[cusip] = entry["data"][0]
                else:
                    out[cusip] = None
            return out
        log.error("OpenFIGI batch failed after retries; returning empties")
        return {c: None for c in cusips}

    # --------------------------------------------------- Resolution

    def _resolve_one(self, cusip: str) -> CusipResolution:
        # 1. Manual override takes precedence.
        if cusip in self._overrides:
            ov = self._overrides[cusip]
            ticker = ov.get("ticker")
            cik = ov.get("cik")
            if cik is None and ticker is not None:
                cik = self._ticker_to_cik(ticker)
            return CusipResolution(
                cusip=cusip,
                ticker=ticker,
                cik=cik,
                issuer_name=ov.get("issuer_name"),
                security_type=ov.get("security_type", "Common Stock"),
                exchange=ov.get("exchange", "US"),
                source="manual_override",
            )

        # 2. Cache hit (and not unresolved-stale)?
        if cusip in self._cache and self._cache[cusip].source != "unresolved":
            return self._cache[cusip]

        # 3. Query OpenFIGI (single-CUSIP fallback path; batches handled in bulk_resolve).
        match = self._query_openfigi_batch([cusip]).get(cusip)
        if match is None:
            return CusipResolution(
                cusip=cusip, ticker=None, cik=None, issuer_name=None,
                security_type=None, exchange=None, source="unresolved",
            )

        ticker = match.get("ticker")
        cik = self._ticker_to_cik(ticker) if ticker else None
        source = "verified" if cik is not None else "openfigi_only"
        return CusipResolution(
            cusip=cusip,
            ticker=ticker,
            cik=cik,
            issuer_name=match.get("name"),
            security_type=match.get("securityType"),
            exchange=match.get("exchCode"),
            source=source,
        )

    def resolve(self, cusip: str, *, issuer_name_hint: str | None = None) -> CusipResolution:
        """Resolve a single CUSIP. Cached after first call.

        `issuer_name_hint` (typically the 13F's `nameOfIssuer`) is used by
        the SEC name-match fallback when OpenFIGI cannot find a US-primary
        listing for this CUSIP.
        """
        normalized = cusip.upper().strip().zfill(9)
        if normalized in self._cache and self._cache[normalized].source != "unresolved":
            return self._cache[normalized]
        result = self._resolve_one(normalized)
        # Fallback: if OpenFIGI didn't find a US listing, try SEC name match.
        if (
            result.source == "unresolved"
            and issuer_name_hint
            and self._name_to_ticker_cik(issuer_name_hint) is not None
        ):
            ticker, cik = self._name_to_ticker_cik(issuer_name_hint)
            result = CusipResolution(
                cusip=normalized,
                ticker=ticker,
                cik=cik,
                issuer_name=issuer_name_hint,
                security_type="Common Stock",
                exchange="US",
                source="sec_name_match",
            )
        self._cache[normalized] = result
        self._save_cache()
        return result

    def bulk_resolve(
        self,
        cusips: Iterable[str],
        *,
        issuer_name_hints: dict[str, str] | None = None,
    ) -> dict[str, CusipResolution]:
        """Resolve many CUSIPs efficiently using OpenFIGI batch requests.

        `issuer_name_hints` maps CUSIP -> 13F nameOfIssuer; used by the SEC
        name-match fallback for CUSIPs OpenFIGI can't resolve to a US-primary
        listing (a known gap for some smaller/mid-cap US tickers).

        Returns CUSIP -> CusipResolution. Updates the cache and writes the
        disagreements CSV for any unresolved or non-verified entries.
        """
        hints = issuer_name_hints or {}
        normalized = sorted({c.upper().strip().zfill(9) for c in cusips if c})

        # Skip already-cached & resolved entries.
        to_query = [
            c for c in normalized
            if c not in self._overrides
            and (c not in self._cache or self._cache[c].source == "unresolved")
        ]

        log.info(
            "Resolving %d CUSIPs (%d cached, %d via overrides)",
            len(to_query),
            sum(1 for c in normalized if c in self._cache and self._cache[c].source != "unresolved"),
            sum(1 for c in normalized if c in self._overrides),
        )

        # Batch query OpenFIGI.
        for i in range(0, len(to_query), OPENFIGI_BATCH_SIZE):
            batch = to_query[i: i + OPENFIGI_BATCH_SIZE]
            matches = self._query_openfigi_batch(batch)
            for cusip in batch:
                match = matches.get(cusip)
                if match is None:
                    # OpenFIGI miss -> try SEC name-match fallback.
                    fallback = None
                    hint = hints.get(cusip)
                    if hint:
                        fallback = self._name_to_ticker_cik(hint)
                    if fallback is not None:
                        ticker, cik = fallback
                        self._cache[cusip] = CusipResolution(
                            cusip=cusip,
                            ticker=ticker,
                            cik=cik,
                            issuer_name=hint,
                            security_type="Common Stock",
                            exchange="US",
                            source="sec_name_match",
                        )
                    else:
                        self._cache[cusip] = CusipResolution(
                            cusip=cusip, ticker=None, cik=None, issuer_name=hint,
                            security_type=None, exchange=None, source="unresolved",
                        )
                    continue
                ticker = match.get("ticker")
                cik = self._ticker_to_cik(ticker) if ticker else None
                source = "verified" if cik is not None else "openfigi_only"
                self._cache[cusip] = CusipResolution(
                    cusip=cusip,
                    ticker=ticker,
                    cik=cik,
                    issuer_name=match.get("name"),
                    security_type=match.get("securityType"),
                    exchange=match.get("exchCode"),
                    source=source,
                )
        if to_query:
            self._save_cache()

        # Compose results from cache (for normalized inputs).
        out: dict[str, CusipResolution] = {}
        for c in normalized:
            if c in self._overrides:
                out[c] = self._resolve_one(c)
            else:
                out[c] = self._cache.get(
                    c,
                    CusipResolution(
                        cusip=c, ticker=None, cik=None, issuer_name=hints.get(c),
                        security_type=None, exchange=None, source="unresolved",
                    ),
                )

        self._write_disagreements_log(out)
        return out

    # --------------------------------------------------- Disagreements log

    def _write_disagreements_log(self, results: dict[str, CusipResolution]) -> None:
        problematic = [
            r for r in results.values()
            if r.source in ("openfigi_only", "unresolved")
        ]
        if not problematic:
            return
        with self._disagreements_file.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "cusip", "ticker", "cik", "issuer_name",
                    "security_type", "exchange", "source",
                ],
            )
            w.writeheader()
            for r in sorted(problematic, key=lambda x: x.cusip):
                w.writerow(asdict(r))
        log.info(
            "Wrote %d entries to %s for manual review",
            len(problematic),
            self._disagreements_file,
        )
