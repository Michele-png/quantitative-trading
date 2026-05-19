"""Tests for management source archive providers and deterministic ingestion."""

from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from value_investing_backend.data.management_documents import (
    ArchiveBackedManagementProvider,
    DeterministicDiscoverer,
    DocumentCandidate,
    DocumentValidationPolicy,
    EdgarDiscoverer,
    IRArchiveDiscoverer,
    IRSourcesConfig,
    LocalManagementDocumentArchive,
    ManagementDocumentFetcher,
    SupabaseManagementDocumentStore,
    _bytes_to_text,
    _identity_in_text,
)


def _write_archive(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows))


def test_local_archive_filters_validated_point_in_time_rows(tmp_path: Path) -> None:
    archive_path = tmp_path / "management_documents.jsonl"
    _write_archive(
        archive_path,
        [
            {
                "ticker": "AAPL",
                "as_of": "2024-01-01",
                "doc_type": "shareholder_letter",
                "text": "Apple shareholder letter",
                "source_url": "https://investor.apple.com/letter",
                "storage_path": "AAPL/shareholder_letter/doc.txt",
                "content_hash": "a" * 64,
                "provider": "fixture",
                "published_date": "2023-10-31",
                "retrieval_status": "validated",
                "manual_review_required": False,
            },
            {
                "ticker": "AAPL",
                "as_of": "2026-01-01",
                "doc_type": "shareholder_letter",
                "text": "future text",
                "source_url": "https://investor.apple.com/future",
                "storage_path": "AAPL/shareholder_letter/future.txt",
                "content_hash": "b" * 64,
                "provider": "fixture",
                "published_date": "2025-10-31",
                "retrieval_status": "validated",
                "manual_review_required": False,
            },
        ],
    )

    archive = LocalManagementDocumentArchive(archive_path)
    docs = archive.get_documents(
        "AAPL",
        date(2024, 6, 1),
        doc_type="shareholder_letter",
    )

    assert len(docs) == 1
    assert docs[0].source_url == "https://investor.apple.com/letter"


def test_archive_provider_exposes_transcripts(tmp_path: Path) -> None:
    archive_path = tmp_path / "management_documents.jsonl"
    _write_archive(
        archive_path,
        [
            {
                "ticker": "MSFT",
                "as_of": "2024-04-30",
                "doc_type": "earnings_transcript",
                "text": "Microsoft earnings call transcript " * 300,
                "source_url": "https://example.com/msft-q1",
                "storage_path": "MSFT/earnings_transcript/doc.txt",
                "content_hash": "c" * 64,
                "provider": "archive_fixture",
                "published_date": "2024-04-25",
                "fiscal_year": 2024,
                "fiscal_quarter": 1,
                "retrieval_status": "validated",
                "manual_review_required": False,
            },
        ],
    )
    provider = ArchiveBackedManagementProvider(LocalManagementDocumentArchive(archive_path))

    transcripts = provider.get_transcripts(
        "MSFT",
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
    )

    assert len(transcripts) == 1
    assert transcripts[0].source == "archive_fixture"
    assert transcripts[0].fiscal_quarter == 1


def test_fetcher_hashes_and_rejects_too_short_content() -> None:
    session = MagicMock()
    response = MagicMock()
    response.content = b"Apple short"
    response.headers = {"content-type": "text/plain"}
    response.raise_for_status.return_value = None
    session.get.return_value = response

    fetcher = ManagementDocumentFetcher(
        validation_policy=DocumentValidationPolicy(),
        session=session,
    )
    doc = fetcher.fetch(
        DocumentCandidate(
            ticker="AAPL",
            company_name="Apple Inc.",
            doc_type="shareholder_letter",
            source_url="https://investor.apple.com/letter",
            as_of=date(2024, 1, 1),
            provider="fixture",
            published_date=date(2023, 10, 31),
        )
    )

    assert doc.content_hash
    assert not doc.validation.passed
    assert doc.retrieval_status == "rejected"
    assert "below shareholder_letter floor" in doc.validation.notes[0]


def test_supabase_store_treats_duplicate_storage_upload_as_idempotent() -> None:
    session = MagicMock()
    response = MagicMock()
    response.status_code = 400
    response.json.return_value = {
        "statusCode": "23505",
        "error": "Duplicate",
        "message": "The resource already exists",
    }
    response.text = "The resource already exists"
    session.post.return_value = response
    store = SupabaseManagementDocumentStore(
        supabase_url="https://example.supabase.co",
        service_role_key="service-role",
        session=session,
    )

    store._upload("MSFT/proxy_letter/doc.html", b"<html></html>", "text/html")

    response.raise_for_status.assert_not_called()


def test_supabase_store_raises_for_non_duplicate_storage_upload_error() -> None:
    session = MagicMock()
    response = MagicMock()
    response.status_code = 400
    response.json.return_value = {"message": "invalid bucket"}
    response.text = "invalid bucket"
    response.raise_for_status.side_effect = RuntimeError("bad request")
    session.post.return_value = response
    store = SupabaseManagementDocumentStore(
        supabase_url="https://example.supabase.co",
        service_role_key="service-role",
        session=session,
    )

    with pytest.raises(RuntimeError, match="bad request"):
        store._upload("MSFT/proxy_letter/doc.html", b"<html></html>", "text/html")


# --------------------------------------------------------------------------
# Validation policy hardening
# --------------------------------------------------------------------------


def _candidate(**overrides) -> DocumentCandidate:
    defaults = dict(
        ticker="AAPL",
        company_name="Apple Inc.",
        doc_type="shareholder_letter",
        source_url="https://investor.apple.com/letter",
        as_of=date(2024, 6, 1),
        provider="fixture",
        published_date=date(2024, 3, 1),
    )
    defaults.update(overrides)
    return DocumentCandidate(**defaults)


def test_validation_rejects_missing_published_date_for_strict_doc_type() -> None:
    """Strict doc types (e.g. annual_report) must have a known publication date.

    Without it, PIT eligibility cannot be confirmed, so the candidate is
    rejected outright rather than landing in manual_review."""
    policy = DocumentValidationPolicy(extra_allowed_domains=("apple.com",))
    cand = _candidate(doc_type="annual_report", published_date=None,
                       source_url="https://investor.apple.com/ar")
    result = policy.validate(cand, "Apple Inc. annual report " * 1500)
    assert not result.passed
    assert any("published_date is missing" in note for note in result.notes)


def test_validation_routes_letter_without_pub_date_to_manual_review() -> None:
    """Shareholder letters are lower-stakes: missing date is manual_review, not reject."""
    policy = DocumentValidationPolicy(extra_allowed_domains=("apple.com",))
    cand = _candidate(published_date=None)
    result = policy.validate(cand, "Apple Inc. shareholder letter " * 400)
    assert result.passed
    assert result.manual_review_required


def test_validation_identity_ignores_generic_suffixes() -> None:
    """A document that only contains 'Inc.' must not pass identity for Apple Inc."""
    cand = _candidate(company_name="Apple Inc.")
    match = _identity_in_text(cand, "Generic boilerplate, Inc. text " * 100)
    assert not match.matched


def test_validation_identity_requires_ticker_and_name_when_both_available() -> None:
    """Brand mention alone is not enough when the candidate also has a ticker.

    A competitor's 10-K, an analyst note, or a news scrape can all contain the
    word "apple". Identity matching must require the ticker AND a name token to
    independently appear when both are available on the candidate.
    """
    cand = _candidate(company_name="Apple Inc.")
    brand_only = _identity_in_text(cand, "Apple announced strong iPhone sales " * 50)
    assert not brand_only.matched

    both = _identity_in_text(cand, "Apple Inc. (NASDAQ: AAPL) reported results " * 30)
    assert both.matched
    assert "apple" in both.matched_tokens
    assert "aapl" in both.matched_tokens


def test_validation_identity_passes_when_only_name_token_available() -> None:
    """If the candidate has no ticker (e.g. private filing context), a single
    distinguishing name-token hit is the strongest signal we have."""
    cand = _candidate(ticker="", company_name="Apple Inc.")
    match = _identity_in_text(cand, "Apple announced strong iPhone sales " * 50)
    assert match.matched


def test_validation_strict_doc_type_rejects_on_unallowed_domain() -> None:
    policy = DocumentValidationPolicy()
    cand = _candidate(
        doc_type="proxy_compensation",
        source_url="https://random-aggregator.example/file",
    )
    body = (
        "Apple Inc. (NASDAQ: AAPL) Compensation Discussion and Analysis. "
        "Named Executive Officers. Summary Compensation Table. " * 200
    )
    result = policy.validate(cand, body)
    assert not result.passed
    assert any("not allowlisted" in note for note in result.notes)


def test_validation_proxy_landing_page_is_not_proxy_compensation() -> None:
    """A generic IR landing page that mentions the company must NOT pass as
    proxy_compensation. The doc-type signature check is the gate."""
    policy = DocumentValidationPolicy(extra_allowed_domains=("apple.com",))
    cand = _candidate(
        doc_type="proxy_compensation",
        source_url="https://investor.apple.com/",
    )
    body = (
        "Welcome to Apple Inc. (NASDAQ: AAPL) investor relations. "
        "Latest news, events, and quarterly updates. " * 300
    )
    result = policy.validate(cand, body)
    assert not result.passed
    assert any("does not contain any proxy_compensation signature" in note for note in result.notes)


def test_validation_real_proxy_compensation_passes() -> None:
    """A document that mentions company identity AND proxy-compensation signature
    phrases on an allowlisted domain should pass cleanly."""
    policy = DocumentValidationPolicy(extra_allowed_domains=("apple.com",))
    cand = _candidate(
        doc_type="proxy_compensation",
        source_url="https://www.sec.gov/Archives/edgar/data/320193/000032019324000001/aapl-def14a.htm",
    )
    body = (
        "Apple Inc. (NASDAQ: AAPL) DEF 14A. "
        "Compensation Discussion and Analysis. "
        "Named Executive Officers. Summary Compensation Table. "
        "Pay Ratio Disclosure. Compensation Committee Report. " * 100
    )
    result = policy.validate(cand, body)
    assert result.passed
    assert not result.manual_review_required


# --------------------------------------------------------------------------
# PDF parsing
# --------------------------------------------------------------------------


_PYPDF = pytest.importorskip("pypdf")


def _crafted_pdf(text: str) -> bytes:
    """Build a minimal in-memory PDF whose text layer is ``text``.

    Uses pypdf's writer so the resulting PDF round-trips through ``PdfReader``.
    """
    from pypdf import PdfWriter  # noqa: PLC0415
    from pypdf.generic import (  # noqa: PLC0415
        ContentStream,
        DictionaryObject,
        FloatObject,
        NameObject,
        NumberObject,
        TextStringObject,
    )

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    content = ContentStream(None, writer)
    content.operations = [
        ([], b"BT"),
        ([NameObject("/F1"), FloatObject(12)], b"Tf"),
        ([NumberObject(72), NumberObject(720)], b"Td"),
        ([TextStringObject(text)], b"Tj"),
        ([], b"ET"),
    ]
    page[NameObject("/Contents")] = content

    resources = DictionaryObject()
    fonts = DictionaryObject()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_indirect = writer._add_object(font)
    fonts[NameObject("/F1")] = font_indirect
    resources[NameObject("/Font")] = fonts
    page[NameObject("/Resources")] = resources

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_bytes_to_text_extracts_pdf_payload() -> None:
    pdf_bytes = _crafted_pdf("Apple Inc. annual report 2024 shareholder letter")
    text = _bytes_to_text(pdf_bytes, "application/pdf")
    assert "Apple" in text
    assert "shareholder letter" in text.lower()


def test_bytes_to_text_detects_pdf_without_content_type() -> None:
    """Even if the server returns the wrong content-type, the PDF magic is used."""
    pdf_bytes = _crafted_pdf("Apple Inc. proxy compensation discussion")
    text = _bytes_to_text(pdf_bytes, "application/octet-stream")
    assert "Apple" in text


def test_bytes_to_text_returns_empty_on_unknown_binary() -> None:
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 256
    text = _bytes_to_text(raw, "image/png")
    assert text == ""


# --------------------------------------------------------------------------
# DeterministicDiscoverer
# --------------------------------------------------------------------------


def _fake_edgar_client(filings: list[dict]) -> MagicMock:
    client = MagicMock()
    client.get_cik.return_value = 320193
    client.list_filings.return_value = filings
    return client


def test_edgar_discoverer_emits_def14a_and_10k_candidates() -> None:
    filings = [
        {
            "accessionNumber": "0000320193-24-000001",
            "filingDate": "2024-01-12",
            "reportDate": "2023-09-30",
            "form": "10-K",
            "primaryDocument": "aapl-20230930.htm",
            "cik": 320193,
        },
        {
            "accessionNumber": "0000320193-24-000002",
            "filingDate": "2024-03-04",
            "reportDate": "2023-12-31",
            "form": "DEF 14A",
            "primaryDocument": "aapl-def14a-2024.htm",
            "cik": 320193,
        },
        # Future filing must be skipped (after as_of).
        {
            "accessionNumber": "0000320193-25-000005",
            "filingDate": "2025-12-01",
            "reportDate": "2025-09-30",
            "form": "10-K",
            "primaryDocument": "aapl-20250930.htm",
            "cik": 320193,
        },
    ]
    discoverer = EdgarDiscoverer(_fake_edgar_client(filings))
    candidates = discoverer.discover(
        ticker="AAPL",
        company_name="Apple Inc.",
        as_of=date(2024, 6, 1),
        doc_types=(
            "proxy_compensation",
            "proxy_letter",
            "annual_report",
            "ten_k_mda",
        ),
    )
    by_type = {c.doc_type: c for c in candidates}
    assert set(by_type) == {
        "proxy_compensation",
        "proxy_letter",
        "annual_report",
        "ten_k_mda",
    }
    assert "aapl-20230930" in by_type["annual_report"].source_url
    assert by_type["proxy_compensation"].published_date == date(2024, 3, 4)
    assert all(c.provider == "edgar_deterministic" for c in candidates)


def test_ir_archive_discoverer_emits_from_seeds() -> None:
    config = IRSourcesConfig.from_mapping({
        "AAPL": {
            "ir_root": "https://investor.apple.com/",
            "annual_reports_url": "https://investor.apple.com/sec-filings/",
            "shareholder_letter_urls": [
                "https://investor.apple.com/letters/2023.pdf",
            ],
            "extra_urls": [
                {"doc_type": "ir_material", "url": "https://investor.apple.com/faq/"},
            ],
            "allowed_domains": ["apple.com"],
        }
    })
    discoverer = IRArchiveDiscoverer(config)
    candidates = discoverer.discover(
        ticker="AAPL",
        company_name="Apple Inc.",
        as_of=date(2024, 6, 1),
        doc_types=("shareholder_letter", "annual_report", "ir_material"),
    )
    doc_types = {c.doc_type for c in candidates}
    assert {"shareholder_letter", "annual_report", "ir_material"} <= doc_types
    assert config.domains_for("AAPL") == ("apple.com",)


def test_deterministic_discoverer_dedupes_across_sources() -> None:
    edgar = EdgarDiscoverer(_fake_edgar_client([
        {
            "accessionNumber": "0000320193-24-000001",
            "filingDate": "2024-01-12",
            "reportDate": "2023-09-30",
            "form": "10-K",
            "primaryDocument": "aapl-20230930.htm",
            "cik": 320193,
        },
    ]))
    ir = IRArchiveDiscoverer(IRSourcesConfig.from_mapping({
        "AAPL": {
            "annual_reports_url":
                "https://www.sec.gov/Archives/edgar/data/320193/000032019324000001/aapl-20230930.htm",
        }
    }))
    discoverer = DeterministicDiscoverer(edgar=edgar, ir_archive=ir)
    candidates = discoverer.discover(
        ticker="AAPL",
        company_name="Apple Inc.",
        as_of=date(2024, 6, 1),
        doc_types=("annual_report",),
    )
    assert len(candidates) == 1
    assert candidates[0].provider == "edgar_deterministic"
