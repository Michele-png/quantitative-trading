"""Archived management-source document access and ingestion helpers.

The Management LLM pipeline should consume immutable, provenance-rich source
documents whenever they exist. This module keeps the archive boundary narrow:

* archive readers expose validated text by document type;
* discovery workers may propose URLs, but never write storage directly;
* deterministic fetch/validation/store code owns downloading, hashing, and
  provenance recording.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup

from value_investing_backend.data.transcripts import EarningsTranscript

log = logging.getLogger(__name__)

MANAGEMENT_DOCUMENTS_BUCKET = "management-source-documents"

ManagementDocumentType = Literal[
    "earnings_transcript",
    "shareholder_letter",
    "proxy_compensation",
    "proxy_letter",
    "annual_report",
    "ir_material",
    "ten_k_mda",
    "ten_k_item1_fallback",
]

RetrievalStatus = Literal[
    "candidate",
    "downloaded",
    "validated",
    "rejected",
    "error",
    "manual_review",
]

TEXTUAL_MIME_TYPES = {
    "text/plain",
    "text/html",
    "application/json",
    "application/xhtml+xml",
}

DEFAULT_MIN_CHARS_BY_TYPE: dict[ManagementDocumentType, int] = {
    "earnings_transcript": 5_000,
    "shareholder_letter": 2_000,
    "proxy_compensation": 4_000,
    "proxy_letter": 1_000,
    "annual_report": 10_000,
    "ir_material": 1_500,
    "ten_k_mda": 4_000,
    "ten_k_item1_fallback": 2_000,
}

DEFAULT_ALLOWED_DOMAINS = (
    "sec.gov",
    "annualreports.com",
    "financialmodelingprep.com",
    "stocklight.com",
)

# Doc types whose content drives the most explicit LLM evidence. For these we
# refuse to fall back to manual_review on identity/domain warnings; weak identity
# means the candidate is rejected outright. Lower-stakes types are still allowed
# to land in manual_review for a human to promote.
STRICT_IDENTITY_DOC_TYPES: tuple[ManagementDocumentType, ...] = (
    "earnings_transcript",
    "annual_report",
    "ten_k_mda",
    "ten_k_item1_fallback",
    "proxy_compensation",
)

# Per-doc-type signature phrases. A document that fails ALL of its signatures
# is almost certainly mis-typed (e.g. an IR landing page being claimed as a
# proxy_compensation document). The validator routes mis-typed strict docs to
# manual_review so the strict-policy rejection then drops them outright.
DOC_TYPE_SIGNATURES: dict[ManagementDocumentType, tuple[str, ...]] = {
    "earnings_transcript": (
        "earnings call",
        "prepared remarks",
        "question-and-answer",
        "thank you, operator",
        "operator:",
        "conference call",
    ),
    "annual_report": (
        "annual report",
        "form 10-k",
        "fiscal year",
        "letter to shareholders",
        "letter to our shareholders",
        "letter to our stockholders",
    ),
    "ten_k_mda": (
        "management's discussion and analysis",
        "results of operations",
        "liquidity and capital resources",
    ),
    "proxy_compensation": (
        "compensation discussion",
        "named executive officers",
        "summary compensation table",
        "pay ratio",
        "executive compensation",
        "compensation committee",
    ),
    "proxy_letter": (
        "letter to shareholders",
        "letter to our shareholders",
        "letter to our stockholders",
        "to our shareholders",
        "to our stockholders",
        "dear shareholders",
        "dear stockholders",
    ),
    "shareholder_letter": (
        "letter to shareholders",
        "letter to our shareholders",
        "letter to our stockholders",
        "to our shareholders",
        "to our stockholders",
        "dear shareholders",
        "dear stockholders",
    ),
    # Intentionally unset (no required signature):
    #   - ir_material: too heterogeneous to require a single phrase.
    #   - ten_k_item1_fallback: phrases are tautological (e.g. "business").
}

# Suffixes stripped from company-name tokens before identity matching. We want
# multi-token matches (e.g. "Apple" + "AAPL") rather than a "Inc." substring
# matching every other public company.
COMPANY_NAME_SUFFIX_TOKENS: frozenset[str] = frozenset({
    "inc",
    "inc.",
    "incorporated",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "company",
    "ltd",
    "ltd.",
    "limited",
    "plc",
    "llc",
    "lp",
    "holdings",
    "group",
    "the",
    "&",
    "and",
})


@dataclass(frozen=True)
class ArchivedManagementDocument:
    """One validated management source document loaded from the archive."""

    ticker: str
    as_of: date
    doc_type: ManagementDocumentType
    text: str
    source_url: str
    storage_path: str
    content_hash: str
    provider: str
    published_date: date | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    confidence: float | None = None
    validation_notes: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentCandidate:
    """Structured URL proposal from a deterministic search or Cursor worker."""

    ticker: str
    company_name: str | None
    doc_type: ManagementDocumentType
    source_url: str
    as_of: date
    provider: str
    published_date: date | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    confidence: float | None = None
    quoted_evidence: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IRSourcesEntry:
    """Per-ticker investor-relations source seed used by deterministic discovery."""

    ticker: str
    ir_root: str | None = None
    annual_reports_url: str | None = None
    transcripts_url: str | None = None
    shareholder_letter_urls: tuple[str, ...] = ()
    extra_urls: tuple[tuple[ManagementDocumentType, str], ...] = ()
    allowed_domains: tuple[str, ...] = ()


@dataclass(frozen=True)
class IRSourcesConfig:
    """Collection of ``IRSourcesEntry`` rows, indexed by ticker."""

    entries: Mapping[str, IRSourcesEntry] = field(default_factory=dict)

    def get(self, ticker: str) -> IRSourcesEntry | None:
        return self.entries.get(ticker.upper())

    def domains_for(self, ticker: str) -> tuple[str, ...]:
        entry = self.get(ticker)
        return entry.allowed_domains if entry else ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> IRSourcesConfig:
        """Build from a {ticker: {ir_root, annual_reports_url, ...}} mapping.

        Keys are case-insensitive; values follow the ``IRSourcesEntry`` shape.
        Unknown keys are ignored so external configs can carry extra metadata.
        """

        entries: dict[str, IRSourcesEntry] = {}
        for ticker, payload in (raw or {}).items():
            if not isinstance(payload, Mapping):
                continue
            extra_raw = payload.get("extra_urls") or []
            extras: list[tuple[ManagementDocumentType, str]] = []
            for row in extra_raw:
                if not isinstance(row, Mapping):
                    continue
                doc_type = row.get("doc_type")
                url = row.get("url")
                if not doc_type or not url:
                    continue
                extras.append((doc_type, str(url)))
            letters_raw = payload.get("shareholder_letter_urls") or []
            entries[str(ticker).upper()] = IRSourcesEntry(
                ticker=str(ticker).upper(),
                ir_root=_clean_str(payload.get("ir_root")),
                annual_reports_url=_clean_str(payload.get("annual_reports_url")),
                transcripts_url=_clean_str(payload.get("transcripts_url")),
                shareholder_letter_urls=tuple(str(u) for u in letters_raw if u),
                extra_urls=tuple(extras),
                allowed_domains=tuple(
                    str(d).lower().strip()
                    for d in (payload.get("allowed_domains") or ())
                    if str(d).strip()
                ),
            )
        return cls(entries=entries)


@dataclass(frozen=True)
class DocumentValidationResult:
    """Outcome of validating a downloaded candidate before storage."""

    passed: bool
    notes: tuple[str, ...]
    manual_review_required: bool = False


@dataclass(frozen=True)
class FetchedManagementDocument:
    """Downloaded, normalized, hashed management document ready to store."""

    candidate: DocumentCandidate
    content_text: str
    raw_content: bytes
    content_hash: str
    mime_type: str
    validation: DocumentValidationResult

    @property
    def retrieval_status(self) -> RetrievalStatus:
        if self.validation.passed and not self.validation.manual_review_required:
            return "validated"
        if self.validation.passed:
            return "manual_review"
        return "rejected"


class EdgarDiscoverer:
    """Emit ``DocumentCandidate``s from SEC EDGAR for SEC-derived doc types.

    EDGAR is authoritative for proxy and 10-K filings. This discoverer maps
    those filings to the management-document types the LLM pipeline consumes:

    * ``proxy_compensation`` / ``proxy_letter`` -> latest DEF 14A primary document.
    * ``annual_report`` / ``ten_k_mda`` / ``ten_k_item1_fallback`` -> latest 10-K.

    Earnings transcripts and shareholder letters are not produced here because
    they are not native EDGAR artifacts; ``IRArchiveDiscoverer`` and the Cursor
    fallback handle them.
    """

    name = "edgar_deterministic"

    _DEF14A_DOC_TYPES: tuple[ManagementDocumentType, ...] = (
        "proxy_compensation",
        "proxy_letter",
    )
    _TENK_DOC_TYPES: tuple[ManagementDocumentType, ...] = (
        "annual_report",
        "ten_k_mda",
        "ten_k_item1_fallback",
    )

    def __init__(self, edgar_client: Any) -> None:
        self._edgar = edgar_client

    def discover(
        self,
        *,
        ticker: str,
        company_name: str | None,
        as_of: date,
        doc_types: Iterable[ManagementDocumentType],
    ) -> list[DocumentCandidate]:
        wanted = tuple(doc_types)
        forms = self._forms_needed_for(wanted)
        if not wanted or not forms:
            return []

        try:
            cik = self._edgar.get_cik(ticker)
        except Exception as exc:  # noqa: BLE001
            log.warning("EDGAR CIK lookup failed for %s: %s", ticker, exc)
            return []

        try:
            filings = self._edgar.list_filings(cik, forms=forms)
        except Exception as exc:  # noqa: BLE001
            log.warning("EDGAR list_filings failed for %s: %s", ticker, exc)
            return []

        latest_by_form = self._select_latest_per_form(filings, as_of=as_of)
        out: list[DocumentCandidate] = []
        for doc_type in wanted:
            filing = self._filing_for_doc_type(doc_type, latest_by_form)
            if filing is None:
                continue
            candidate = self._candidate_from_filing(
                filing=filing,
                doc_type=doc_type,
                ticker=ticker,
                company_name=company_name,
                as_of=as_of,
            )
            if candidate is not None:
                out.append(candidate)
        return out

    def _forms_needed_for(
        self, wanted: tuple[ManagementDocumentType, ...]
    ) -> list[str]:
        forms: list[str] = []
        if any(dt in self._DEF14A_DOC_TYPES for dt in wanted):
            forms.append("DEF 14A")
        if any(dt in self._TENK_DOC_TYPES for dt in wanted):
            forms.append("10-K")
        return forms

    @staticmethod
    def _select_latest_per_form(
        filings: Iterable[Mapping[str, Any]],
        *,
        as_of: date,
    ) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for filing in filings:
            form = str(filing.get("form", "")).upper()
            try:
                filing_date = _parse_date(filing.get("filingDate"))
            except Exception:  # noqa: BLE001
                continue
            if filing_date > as_of or form in latest:
                continue
            latest[form] = dict(filing)
        return latest

    def _candidate_from_filing(
        self,
        *,
        filing: Mapping[str, Any],
        doc_type: ManagementDocumentType,
        ticker: str,
        company_name: str | None,
        as_of: date,
    ) -> DocumentCandidate | None:
        url = _edgar_primary_doc_url(filing)
        if url is None:
            return None
        return DocumentCandidate(
            ticker=ticker.upper(),
            company_name=company_name,
            doc_type=doc_type,
            source_url=url,
            as_of=as_of,
            provider=self.name,
            published_date=_parse_optional_date(filing.get("filingDate")),
            fiscal_year=_year_for_filing(filing),
            fiscal_quarter=None,
            confidence=0.95,
            quoted_evidence=(
                f"SEC EDGAR {filing.get('form')} {filing.get('accessionNumber')} "
                f"filed {filing.get('filingDate')}"
            ),
            evidence={
                "discoverer": self.name,
                "form": filing.get("form"),
                "accession_number": filing.get("accessionNumber"),
                "primary_document": filing.get("primaryDocument"),
                "filing_date": str(filing.get("filingDate") or ""),
                "report_date": str(filing.get("reportDate") or ""),
            },
        )

    def _filing_for_doc_type(
        self,
        doc_type: ManagementDocumentType,
        latest_by_form: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        if doc_type in self._DEF14A_DOC_TYPES:
            return latest_by_form.get("DEF 14A")
        if doc_type in self._TENK_DOC_TYPES:
            return latest_by_form.get("10-K")
        return None


class IRArchiveDiscoverer:
    """Emit candidates from per-ticker investor-relations seeds.

    Seeds are user-curated and stored in ``ir_sources.yml`` (loaded by ETL). They
    are intentionally URL-direct: every URL must already be a stable, public
    archive entry. The fetcher and validator decide whether the content is
    actually usable; this discoverer only proposes locations.
    """

    name = "ir_archive_seed"

    def __init__(self, config: IRSourcesConfig) -> None:
        self._config = config

    def discover(
        self,
        *,
        ticker: str,
        company_name: str | None,
        as_of: date,
        doc_types: Iterable[ManagementDocumentType],
    ) -> list[DocumentCandidate]:
        entry = self._config.get(ticker)
        if entry is None:
            return []
        wanted = set(doc_types)
        if not wanted:
            return []

        out: list[DocumentCandidate] = []

        if "shareholder_letter" in wanted:
            for url in entry.shareholder_letter_urls:
                out.append(self._make_candidate(
                    ticker=ticker,
                    company_name=company_name,
                    doc_type="shareholder_letter",
                    url=url,
                    as_of=as_of,
                ))

        for doc_type, url in entry.extra_urls:
            if doc_type not in wanted:
                continue
            out.append(self._make_candidate(
                ticker=ticker,
                company_name=company_name,
                doc_type=doc_type,
                url=url,
                as_of=as_of,
            ))

        if "annual_report" in wanted and entry.annual_reports_url:
            out.append(self._make_candidate(
                ticker=ticker,
                company_name=company_name,
                doc_type="annual_report",
                url=entry.annual_reports_url,
                as_of=as_of,
            ))

        if "earnings_transcript" in wanted and entry.transcripts_url:
            out.append(self._make_candidate(
                ticker=ticker,
                company_name=company_name,
                doc_type="earnings_transcript",
                url=entry.transcripts_url,
                as_of=as_of,
            ))

        if "ir_material" in wanted and entry.ir_root:
            out.append(self._make_candidate(
                ticker=ticker,
                company_name=company_name,
                doc_type="ir_material",
                url=entry.ir_root,
                as_of=as_of,
            ))

        return out

    def _make_candidate(
        self,
        *,
        ticker: str,
        company_name: str | None,
        doc_type: ManagementDocumentType,
        url: str,
        as_of: date,
    ) -> DocumentCandidate:
        return DocumentCandidate(
            ticker=ticker.upper(),
            company_name=company_name,
            doc_type=doc_type,
            source_url=url,
            as_of=as_of,
            provider=self.name,
            published_date=None,
            fiscal_year=None,
            fiscal_quarter=None,
            confidence=0.7,
            quoted_evidence=None,
            evidence={
                "discoverer": self.name,
                "seed": url,
            },
        )


class DeterministicDiscoverer:
    """Run EDGAR + IR-archive discovery and dedupe by (doc_type, source_url)."""

    name = "deterministic"

    def __init__(
        self,
        *,
        edgar: EdgarDiscoverer | None = None,
        ir_archive: IRArchiveDiscoverer | None = None,
    ) -> None:
        self._edgar = edgar
        self._ir = ir_archive

    def discover(
        self,
        *,
        ticker: str,
        company_name: str | None,
        as_of: date,
        doc_types: Iterable[ManagementDocumentType],
    ) -> list[DocumentCandidate]:
        wanted = tuple(doc_types)
        seen: set[tuple[str, str]] = set()
        out: list[DocumentCandidate] = []

        for discoverer in (self._edgar, self._ir):
            if discoverer is None:
                continue
            try:
                candidates = discoverer.discover(
                    ticker=ticker,
                    company_name=company_name,
                    as_of=as_of,
                    doc_types=wanted,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("%s discovery failed for %s: %s", discoverer.name, ticker, exc)
                continue
            for candidate in candidates:
                key = (candidate.doc_type, candidate.source_url)
                if key in seen:
                    continue
                seen.add(key)
                out.append(candidate)
        return out


class ManagementDocumentArchive(Protocol):
    """Read-only archive lookup interface consumed by Management evaluators."""

    name: str

    def get_documents(
        self,
        ticker: str,
        as_of: date,
        *,
        doc_type: ManagementDocumentType | None = None,
        limit: int = 20,
    ) -> list[ArchivedManagementDocument]:
        ...


class ManagementDocumentStore(Protocol):
    """Write-side archive interface used by ETL/backfill jobs."""

    def store(self, document: FetchedManagementDocument) -> ArchivedManagementDocument:
        ...


class ArchiveBackedManagementProvider:
    """Expose archived documents in the shapes the Management bundler needs."""

    name = "management_archive"

    def __init__(self, archive: ManagementDocumentArchive) -> None:
        self._archive = archive

    def get_documents(
        self,
        ticker: str,
        as_of: date,
        *,
        doc_type: ManagementDocumentType | None = None,
        limit: int = 20,
    ) -> list[ArchivedManagementDocument]:
        return self._archive.get_documents(
            ticker=ticker,
            as_of=as_of,
            doc_type=doc_type,
            limit=limit,
        )

    def get_transcripts(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> list[EarningsTranscript]:
        docs = self._archive.get_documents(
            ticker=ticker,
            as_of=end,
            doc_type="earnings_transcript",
            limit=32,
        )
        out: list[EarningsTranscript] = []
        for doc in docs:
            doc_date = doc.published_date or doc.as_of
            if doc_date < start or doc_date > end:
                continue
            if doc.fiscal_year is None or doc.fiscal_quarter is None:
                continue
            out.append(
                EarningsTranscript(
                    ticker=doc.ticker,
                    fiscal_year=doc.fiscal_year,
                    fiscal_quarter=doc.fiscal_quarter,
                    call_date=doc_date,
                    source=doc.provider,
                    text=doc.text,
                )
            )
        out.sort(key=lambda item: (item.fiscal_year, item.fiscal_quarter))
        return _dedupe_transcripts(out)

    def get_shareholder_letter(
        self,
        ticker: str,
        as_of: date,
    ) -> ArchivedManagementDocument | None:
        return self._latest_by_type(
            ticker,
            as_of,
            ("shareholder_letter", "annual_report", "ir_material"),
        )

    def get_proxy_compensation(
        self,
        ticker: str,
        as_of: date,
    ) -> ArchivedManagementDocument | None:
        return self._latest_by_type(ticker, as_of, ("proxy_compensation",))

    def get_proxy_letter(
        self,
        ticker: str,
        as_of: date,
    ) -> ArchivedManagementDocument | None:
        return self._latest_by_type(ticker, as_of, ("proxy_letter",))

    def get_ir_documents(
        self,
        ticker: str,
        as_of: date,
        *,
        limit: int = 5,
    ) -> list[ArchivedManagementDocument]:
        return self._archive.get_documents(
            ticker=ticker,
            as_of=as_of,
            doc_type="ir_material",
            limit=limit,
        )

    def _latest_by_type(
        self,
        ticker: str,
        as_of: date,
        doc_types: tuple[ManagementDocumentType, ...],
    ) -> ArchivedManagementDocument | None:
        candidates: list[ArchivedManagementDocument] = []
        for doc_type in doc_types:
            candidates.extend(
                self._archive.get_documents(
                    ticker=ticker,
                    as_of=as_of,
                    doc_type=doc_type,
                    limit=5,
                )
            )
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda doc: (doc.published_date or date.min, doc.as_of, len(doc.text)),
        )


class LocalManagementDocumentArchive:
    """Archive reader backed by a JSONL file or directory of JSON rows.

    This is intentionally simple and test-friendly. Production ETL should use
    ``SupabaseManagementDocumentArchive`` so storage and metadata stay in sync.
    """

    name = "local_management_documents"

    def __init__(self, path: Path) -> None:
        self._path = path

    def get_documents(
        self,
        ticker: str,
        as_of: date,
        *,
        doc_type: ManagementDocumentType | None = None,
        limit: int = 20,
    ) -> list[ArchivedManagementDocument]:
        rows = list(self._iter_rows())
        docs = [
            _document_from_row(row, text=row.get("text", ""))
            for row in rows
            if _row_matches(row, ticker=ticker, as_of=as_of, doc_type=doc_type)
        ]
        docs = [doc for doc in docs if doc.text]
        docs.sort(key=lambda doc: (doc.published_date or date.min, doc.as_of), reverse=True)
        return docs[:limit]

    def _iter_rows(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        if self._path.is_dir():
            rows: list[dict[str, Any]] = []
            for path in sorted(self._path.glob("*.json")):
                rows.append(json.loads(path.read_text()))
            return rows
        rows = []
        for line in self._path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows


class SupabaseManagementDocumentArchive:
    """Read validated management documents from Supabase table + Storage."""

    name = "supabase_management_documents"

    def __init__(
        self,
        *,
        supabase_url: str,
        service_role_key: str,
        bucket: str = MANAGEMENT_DOCUMENTS_BUCKET,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self._url = supabase_url.rstrip("/")
        self._key = service_role_key
        self._bucket = bucket
        self._timeout = timeout
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls) -> SupabaseManagementDocumentArchive | None:
        url = os.environ.get("SUPABASE_URL", "").strip()
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if not url or not key:
            return None
        return cls(supabase_url=url, service_role_key=key)

    def get_documents(
        self,
        ticker: str,
        as_of: date,
        *,
        doc_type: ManagementDocumentType | None = None,
        limit: int = 20,
    ) -> list[ArchivedManagementDocument]:
        params: dict[str, str] = {
            "select": "*",
            "ticker": f"eq.{ticker.upper()}",
            "as_of": f"lte.{as_of.isoformat()}",
            "retrieval_status": "eq.validated",
            "manual_review_required": "eq.false",
            "order": "as_of.desc,published_date.desc,inserted_at.desc",
            "limit": str(limit),
        }
        if doc_type is not None:
            params["doc_type"] = f"eq.{doc_type}"

        response = self._session.get(
            f"{self._url}/rest/v1/management_documents",
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()

        out: list[ArchivedManagementDocument] = []
        for row in response.json():
            text = self._download_text(
                bucket=str(row.get("storage_bucket") or self._bucket),
                storage_path=str(row["storage_path"]),
            )
            out.append(_document_from_row(row, text=text))
        return out

    def _download_text(self, *, bucket: str, storage_path: str) -> str:
        object_path = quote(storage_path, safe="/")
        response = self._session.get(
            f"{self._url}/storage/v1/object/{bucket}/{object_path}",
            headers=self._headers(),
            timeout=self._timeout,
        )
        response.raise_for_status()
        return _bytes_to_text(response.content, response.headers.get("content-type", ""))

    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
        }


class DocumentValidationPolicy:
    """Validate candidate source content before it can reach the LLM.

    The policy gates documents on four properties:

    * length floor per ``doc_type``;
    * point-in-time: ``published_date`` must be non-null and ``<= as_of``;
    * company identity: the body must reference the ticker AND at least one
      non-generic company-name token (excluding ``inc``, ``corp``, ``the``...);
    * domain allowlist: the host must be in ``DEFAULT_ALLOWED_DOMAINS`` or in
      one of the per-ticker ``extra_allowed_domains`` (typically IR domains
      from ``ir_sources.yml``).

    A failing identity or domain check sets ``manual_review_required=True``.
    For high-stakes doc types in ``STRICT_IDENTITY_DOC_TYPES``, the policy
    refuses to mark ``passed=True`` while ``manual_review_required`` is set --
    those candidates must be rejected outright instead of routed through the
    manual-review pile.
    """

    def __init__(
        self,
        *,
        min_chars_by_type: dict[ManagementDocumentType, int] | None = None,
        allowed_domains: tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS,
        extra_allowed_domains: tuple[str, ...] = (),
        strict_identity_doc_types: tuple[ManagementDocumentType, ...] = (
            STRICT_IDENTITY_DOC_TYPES
        ),
        require_published_date: bool = True,
    ) -> None:
        self._min_chars_by_type = min_chars_by_type or DEFAULT_MIN_CHARS_BY_TYPE
        merged_domains = list(allowed_domains) + list(extra_allowed_domains)
        seen: set[str] = set()
        unique_domains: list[str] = []
        for domain in merged_domains:
            normalized = domain.lower().strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_domains.append(normalized)
        self._allowed_domains = tuple(unique_domains)
        self._strict_doc_types = strict_identity_doc_types
        self._require_published_date = require_published_date

    @property
    def allowed_domains(self) -> tuple[str, ...]:
        return self._allowed_domains

    def validate(self, candidate: DocumentCandidate, text: str) -> DocumentValidationResult:
        notes: list[str] = []
        hard_fail = False
        manual_review_required = False

        normalized = re.sub(r"\s+", " ", text).strip()
        min_chars = self._min_chars_by_type[candidate.doc_type]
        if len(normalized) < min_chars:
            hard_fail = True
            notes.append(
                f"text length {len(normalized)} below {candidate.doc_type} floor {min_chars}"
            )

        if candidate.published_date is None:
            if self._require_published_date:
                manual_review_required = True
                notes.append(
                    "published_date is missing; PIT eligibility cannot be confirmed"
                )
        elif candidate.published_date > candidate.as_of:
            hard_fail = True
            notes.append(
                f"published_date {candidate.published_date.isoformat()} after as_of "
                f"{candidate.as_of.isoformat()}"
            )

        identity_match = _identity_in_text(candidate, normalized)
        if not identity_match.matched:
            manual_review_required = True
            notes.append(
                "company identity not clearly present in document text"
                f" (matched tokens={list(identity_match.matched_tokens)})"
            )

        signature_match = _doc_type_signature_match(candidate.doc_type, normalized)
        if signature_match.required and not signature_match.matched:
            manual_review_required = True
            notes.append(
                f"document body does not contain any {candidate.doc_type} signature "
                f"phrase (checked={list(signature_match.expected)})"
            )

        domain = _hostname(candidate.source_url)
        if domain and not _domain_allowed(domain, self._allowed_domains):
            manual_review_required = True
            notes.append(f"domain {domain} is not allowlisted")

        if (
            manual_review_required
            and candidate.doc_type in self._strict_doc_types
        ):
            hard_fail = True
            notes.append(
                f"strict policy: {candidate.doc_type} cannot enter manual_review; "
                "candidate rejected outright"
            )

        return DocumentValidationResult(
            passed=not hard_fail,
            notes=tuple(notes),
            manual_review_required=manual_review_required and not hard_fail,
        )


class ManagementDocumentFetcher:
    """Deterministically download, normalize, validate, and hash candidates."""

    def __init__(
        self,
        *,
        validation_policy: DocumentValidationPolicy | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self._policy = validation_policy or DocumentValidationPolicy()
        self._timeout = timeout
        self._session = session or requests.Session()

    def fetch(self, candidate: DocumentCandidate) -> FetchedManagementDocument:
        response = self._session.get(
            candidate.source_url,
            headers={"User-Agent": _user_agent()},
            timeout=self._timeout,
        )
        response.raise_for_status()
        raw = response.content
        mime_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        text = _bytes_to_text(raw, mime_type)
        validation = self._policy.validate(candidate, text)
        return FetchedManagementDocument(
            candidate=candidate,
            content_text=text,
            raw_content=raw,
            content_hash=hashlib.sha256(raw).hexdigest(),
            mime_type=mime_type or "application/octet-stream",
            validation=validation,
        )


class SupabaseManagementDocumentStore:
    """Store validated documents in Supabase Storage and metadata index."""

    def __init__(
        self,
        *,
        supabase_url: str,
        service_role_key: str,
        bucket: str = MANAGEMENT_DOCUMENTS_BUCKET,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self._url = supabase_url.rstrip("/")
        self._key = service_role_key
        self._bucket = bucket
        self._timeout = timeout
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls) -> SupabaseManagementDocumentStore:
        url = os.environ.get("SUPABASE_URL", "").strip()
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required to store "
                "management documents."
            )
        return cls(supabase_url=url, service_role_key=key)

    def store(self, document: FetchedManagementDocument) -> ArchivedManagementDocument:
        storage_path = self._storage_path(document)
        self._upload(storage_path, document.raw_content, document.mime_type)
        row = self._upsert_metadata(document, storage_path)
        return _document_from_row(row, text=document.content_text)

    def _storage_path(self, document: FetchedManagementDocument) -> str:
        candidate = document.candidate
        published = candidate.published_date or candidate.as_of
        suffix = _suffix_for_mime(document.mime_type)
        return (
            f"{candidate.ticker.upper()}/{candidate.doc_type}/"
            f"{published.isoformat()}_{document.content_hash[:16]}{suffix}"
        )

    def _upload(self, storage_path: str, raw: bytes, mime_type: str) -> None:
        object_path = quote(storage_path, safe="/")
        response = self._session.post(
            f"{self._url}/storage/v1/object/{self._bucket}/{object_path}",
            headers={
                **self._headers(),
                "Content-Type": mime_type or "application/octet-stream",
                "x-upsert": "false",
            },
            data=raw,
            timeout=self._timeout,
        )
        if response.status_code == 409:
            return
        response.raise_for_status()

    def _upsert_metadata(
        self,
        document: FetchedManagementDocument,
        storage_path: str,
    ) -> dict[str, Any]:
        candidate = document.candidate
        payload = {
            "ticker": candidate.ticker.upper(),
            "as_of": candidate.as_of.isoformat(),
            "fiscal_year": candidate.fiscal_year,
            "fiscal_quarter": candidate.fiscal_quarter,
            "doc_type": candidate.doc_type,
            "source_url": candidate.source_url,
            "storage_bucket": self._bucket,
            "storage_path": storage_path,
            "content_hash": document.content_hash,
            "published_date": (
                candidate.published_date.isoformat() if candidate.published_date else None
            ),
            "provider": candidate.provider,
            "retrieval_status": document.retrieval_status,
            "confidence": candidate.confidence,
            "manual_review_required": document.validation.manual_review_required,
            "validation_notes": list(document.validation.notes),
            "evidence": {
                **candidate.evidence,
                "quoted_evidence": candidate.quoted_evidence,
                "mime_type": document.mime_type,
                "normalized_text_chars": len(document.content_text),
            },
        }
        response = self._session.post(
            f"{self._url}/rest/v1/management_documents",
            headers={
                **self._headers(),
                "Prefer": "resolution=merge-duplicates,return=representation",
            },
            params={"on_conflict": "storage_path"},
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            raise RuntimeError("Supabase returned no management_documents row")
        return data[0]

    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
        }


def default_management_archive_provider() -> ArchiveBackedManagementProvider | None:
    """Build the configured archive provider, if archive access is available."""

    local_path = os.environ.get("MANAGEMENT_DOCUMENTS_ARCHIVE_PATH", "").strip()
    if local_path:
        return ArchiveBackedManagementProvider(LocalManagementDocumentArchive(Path(local_path)))

    if os.environ.get("MANAGEMENT_DOCUMENT_ARCHIVE_DISABLED", "").strip() == "1":
        return None

    supabase = SupabaseManagementDocumentArchive.from_env()
    if supabase is None:
        return None
    return ArchiveBackedManagementProvider(supabase)


def _document_from_row(row: dict[str, Any], *, text: str) -> ArchivedManagementDocument:
    return ArchivedManagementDocument(
        ticker=str(row["ticker"]).upper(),
        as_of=_parse_date(row["as_of"]),
        doc_type=row["doc_type"],
        text=text,
        source_url=str(row["source_url"]),
        storage_path=str(row.get("storage_path", "")),
        content_hash=str(row.get("content_hash", "")),
        provider=str(row.get("provider", "management_archive")),
        published_date=_parse_optional_date(row.get("published_date")),
        fiscal_year=row.get("fiscal_year"),
        fiscal_quarter=row.get("fiscal_quarter"),
        confidence=row.get("confidence"),
        validation_notes=tuple(row.get("validation_notes") or ()),
        evidence=row.get("evidence") or {},
    )


def _row_matches(
    row: dict[str, Any],
    *,
    ticker: str,
    as_of: date,
    doc_type: ManagementDocumentType | None,
) -> bool:
    if str(row.get("ticker", "")).upper() != ticker.upper():
        return False
    if row.get("retrieval_status", "validated") != "validated":
        return False
    if bool(row.get("manual_review_required", False)):
        return False
    if doc_type is not None and row.get("doc_type") != doc_type:
        return False
    row_as_of = _parse_date(row["as_of"])
    if row_as_of > as_of:
        return False
    published = _parse_optional_date(row.get("published_date"))
    return published is None or published <= as_of


def _dedupe_transcripts(transcripts: list[EarningsTranscript]) -> list[EarningsTranscript]:
    seen: set[tuple[int, int]] = set()
    out: list[EarningsTranscript] = []
    for transcript in transcripts:
        key = (transcript.fiscal_year, transcript.fiscal_quarter)
        if key in seen:
            continue
        seen.add(key)
        out.append(transcript)
    return out


def _bytes_to_text(raw: bytes, mime_type: str) -> str:
    if mime_type in TEXTUAL_MIME_TYPES or mime_type.startswith("text/"):
        text = raw.decode("utf-8", errors="ignore")
        if "html" in mime_type:
            return _strip_html(text)
        return re.sub(r"\s+", " ", text).strip()
    if mime_type == "application/pdf" or _looks_like_pdf(raw):
        return _pdf_bytes_to_text(raw)
    # Unknown binary -- return empty so validation rejects on length rather
    # than feeding raw bytes into an LLM. Storing the raw_content is still safe
    # because the hash is computed over the original bytes.
    return ""


def bytes_to_text(raw: bytes, mime_type: str) -> str:
    """Public bytes->text extractor used by the validation pipeline.

    Exposed so out-of-tree tools (e.g. the dashboard's manual-review promotion
    CLI) can re-derive the same text representation that the validator sees,
    without re-implementing PDF / HTML stripping.
    """
    return _bytes_to_text(raw, mime_type)


def _looks_like_pdf(raw: bytes) -> bool:
    return raw[:5] == b"%PDF-"


def _pdf_bytes_to_text(raw: bytes) -> str:
    """Extract text from a PDF.

    Failure modes (corrupt, encrypted, scanned-image PDFs without OCR) return
    an empty string. ``DocumentValidationPolicy`` then rejects the candidate
    on the length floor, which is safer than producing gibberish that could
    pass the floor but mean nothing to the LLM.
    """
    try:
        from pypdf import PdfReader  # noqa: PLC0415  (lazy import keeps cold start fast)
    except ImportError as exc:
        log.warning("pypdf not installed; PDF text extraction disabled: %s", exc)
        return ""

    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        log.info("PDF parse failed: %s", exc)
        return ""

    if getattr(reader, "is_encrypted", False):
        try:
            if reader.decrypt("") == 0:
                log.info("PDF is encrypted; refusing to extract text")
                return ""
        except Exception:  # noqa: BLE001
            return ""

    chunks: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            log.debug("PDF page extract failed: %s", exc)
            continue
        if page_text:
            chunks.append(page_text)
    if not chunks:
        return ""
    return re.sub(r"\s+", " ", "\n".join(chunks)).strip()


def _strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _parse_optional_date(value: Any) -> date | None:
    if not value:
        return None
    return _parse_date(value)


@dataclass(frozen=True)
class _IdentityMatchResult:
    matched: bool
    matched_tokens: tuple[str, ...]


@dataclass(frozen=True)
class _DocTypeSignatureResult:
    required: bool
    matched: bool
    expected: tuple[str, ...]


def _doc_type_signature_match(
    doc_type: ManagementDocumentType, normalized_text: str
) -> _DocTypeSignatureResult:
    """Confirm the document body looks like the doc_type it claims to be.

    For doc types listed in ``DOC_TYPE_SIGNATURES``, the body must contain at
    least one signature phrase. This is the cheap defense against a Cursor
    fallback that returns a generic IR landing page for a ``proxy_compensation``
    request -- the page mentions the company but contains none of "compensation
    discussion / named executive officers / pay ratio / ...".
    """

    expected = DOC_TYPE_SIGNATURES.get(doc_type, ())
    if not expected:
        return _DocTypeSignatureResult(required=False, matched=True, expected=())
    haystack = normalized_text[:80_000].lower()
    matched = any(phrase in haystack for phrase in expected)
    return _DocTypeSignatureResult(required=True, matched=matched, expected=expected)


@dataclass(frozen=True)
class _SignatureMatchResult:
    required: bool
    matched: bool
    matched_phrases: tuple[str, ...]
    expected: tuple[str, ...]


def _doc_type_signature_match(
    doc_type: ManagementDocumentType, normalized_text: str
) -> _SignatureMatchResult:
    """Check the body for at least one signature phrase tied to ``doc_type``.

    Doc types in ``DOC_TYPE_SIGNATURES`` carry phrases that any genuine
    document of that kind almost always contains (e.g. the word "operator:"
    in a transcript, "compensation discussion" in a DEF 14A CD&A). A document
    that fails ALL of them is almost certainly mislabelled by upstream
    discovery -- typically because a Cursor agent picked a generic IR landing
    page when asked for a proxy compensation document.
    """

    expected = DOC_TYPE_SIGNATURES.get(doc_type, ())
    if not expected:
        return _SignatureMatchResult(
            required=False, matched=True, matched_phrases=(), expected=()
        )
    haystack = normalized_text[:80_000].lower()
    matched = tuple(phrase for phrase in expected if phrase in haystack)
    return _SignatureMatchResult(
        required=True,
        matched=bool(matched),
        matched_phrases=matched,
        expected=expected,
    )


def _identity_tokens(candidate: DocumentCandidate) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (ticker_tokens, name_tokens) used for identity matching.

    Generic corporate suffixes (Inc., Corp., Co., Ltd., ...) are stripped so a
    document only mentioning ``"Inc."`` doesn't trick the matcher into passing
    a totally unrelated company.
    """

    ticker_tokens = tuple(
        token.lower() for token in re.findall(r"[A-Za-z0-9]+", candidate.ticker or "") if token
    )
    raw_name = (candidate.company_name or "").lower()
    name_tokens = tuple(
        token
        for token in re.findall(r"[a-z0-9]+", raw_name)
        if token and token not in COMPANY_NAME_SUFFIX_TOKENS and len(token) > 1
    )
    return ticker_tokens, name_tokens


def _identity_in_text(candidate: DocumentCandidate, normalized_text: str) -> _IdentityMatchResult:
    """Match identity using non-generic tokens with multi-signal requirements.

    Trust hardening: a single brand-word hit ("apple") is NOT enough -- a
    competitor's 10-K, an unrelated press release, or a news scrape can all
    contain that one token. When the candidate carries both a ticker and at
    least one non-generic company-name token, the body MUST contain at least
    one of each independently. Single-brand fallbacks (only-name or
    only-ticker) still match on a single hit because we have no second axis
    to cross-check against.
    """

    ticker_tokens, name_tokens = _identity_tokens(candidate)
    haystack = normalized_text[:40_000].lower()
    if not ticker_tokens and not name_tokens:
        return _IdentityMatchResult(matched=True, matched_tokens=())

    ticker_hits = tuple(
        token for token in ticker_tokens
        if re.search(rf"\b{re.escape(token)}\b", haystack)
    )
    name_hits = tuple(
        token for token in name_tokens
        if re.search(rf"\b{re.escape(token)}\b", haystack)
    )

    if ticker_tokens and name_tokens:
        ok = bool(ticker_hits) and bool(name_hits)
    elif ticker_tokens:
        ok = bool(ticker_hits)
    else:
        ok = bool(name_hits)
    return _IdentityMatchResult(matched=ok, matched_tokens=ticker_hits + name_hits)


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _edgar_primary_doc_url(filing: Mapping[str, Any]) -> str | None:
    cik = filing.get("cik")
    accession = filing.get("accessionNumber")
    primary = filing.get("primaryDocument")
    if cik is None or not accession or not primary:
        return None
    accn_clean = str(accession).replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_clean}/{primary}"
    )


def _year_for_filing(filing: Mapping[str, Any]) -> int | None:
    report = filing.get("reportDate") or filing.get("filingDate")
    if not report:
        return None
    try:
        return _parse_date(report).year
    except Exception:  # noqa: BLE001
        return None


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _domain_allowed(domain: str, allowed_domains: tuple[str, ...]) -> bool:
    return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in allowed_domains)


def _suffix_for_mime(mime_type: str) -> str:
    if mime_type == "text/html":
        return ".html"
    if mime_type == "application/json":
        return ".json"
    if mime_type == "application/pdf":
        return ".pdf"
    return ".txt"


def _user_agent() -> str:
    return os.environ.get(
        "SEC_USER_AGENT",
        "value-investing-backend-management-documents contact@example.com",
    )


def fetched_document_to_json(document: FetchedManagementDocument) -> dict[str, Any]:
    """Serialize a fetched document for logs or dry-run backfill reports."""

    payload = asdict(document)
    payload["candidate"]["as_of"] = document.candidate.as_of.isoformat()
    if document.candidate.published_date is not None:
        payload["candidate"]["published_date"] = document.candidate.published_date.isoformat()
    payload["raw_content"] = f"<{len(document.raw_content)} bytes>"
    return payload
