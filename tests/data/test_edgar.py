"""Unit tests for EdgarClient ticker lookup and SEC-form normalization.

Regression coverage for the class-share ticker bug: SEC's
``company_tickers.json`` stores class-share tickers in dash form (``BRK-B``,
``BRK-A``, ``BF-B``) while the rest of the pipeline — ``tickers.yml``,
ScreenedRecord, the Supabase dashboard — uses the dot form (``BRK.B``). A naive
direct lookup of the dot form against the SEC dict raises EdgarError. The fix
normalizes ``.`` → ``-`` at the SEC boundary only.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from value_investing_backend.config import get_config
from value_investing_backend.data.edgar import (
    EdgarClient,
    EdgarError,
    _normalize_ticker_for_sec,
)


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


# Realistic shape of SEC's company_tickers.json: a dict whose values have
# ticker / cik_str / title. Class-share rows use the dash form.
_SEC_INDEX_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "2": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
    "3": {"cik_str": 1067983, "ticker": "BRK-A", "title": "BERKSHIRE HATHAWAY INC"},
    "4": {"cik_str": 14693, "ticker": "BF-B", "title": "BROWN FORMAN CORP"},
}


@pytest.fixture
def edgar_client(tmp_path: Path) -> EdgarClient:
    """An EdgarClient pre-seeded with a SEC index cache on disk.

    Writing the payload directly to the cache file skips the HTTP request
    path entirely; ``get_company_tickers`` reads it on first call.
    """
    cache_dir = tmp_path / "edgar"
    cache_dir.mkdir(parents=True)
    (cache_dir / "company_tickers.json").write_text(json.dumps(_SEC_INDEX_PAYLOAD))
    return EdgarClient(cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# _normalize_ticker_for_sec — direct helper tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_ticker", "expected"),
    [
        ("AAPL", "AAPL"),
        ("aapl", "AAPL"),
        ("BRK.B", "BRK-B"),
        ("brk.b", "BRK-B"),
        ("BRK-B", "BRK-B"),
        ("BRK.A", "BRK-A"),
        ("BF.B", "BF-B"),
        ("  BRK.B  ", "BRK-B"),
    ],
)
def test_normalize_ticker_for_sec(input_ticker: str, expected: str) -> None:
    assert _normalize_ticker_for_sec(input_ticker) == expected


def test_normalize_is_idempotent() -> None:
    """Normalizing an already-normalized SEC ticker is a no-op."""
    for t in ("AAPL", "BRK-B", "BF-B"):
        assert _normalize_ticker_for_sec(_normalize_ticker_for_sec(t)) == t


# ---------------------------------------------------------------------------
# EdgarClient.get_cik — end-to-end lookup with the on-disk cache
# ---------------------------------------------------------------------------


def test_get_cik_resolves_plain_ticker(edgar_client: EdgarClient) -> None:
    assert edgar_client.get_cik("AAPL") == 320193


def test_get_cik_accepts_lowercase(edgar_client: EdgarClient) -> None:
    assert edgar_client.get_cik("aapl") == 320193


def test_get_cik_resolves_class_b_dot_form(edgar_client: EdgarClient) -> None:
    """The regression: BRK.B (dot form) must resolve via SEC's dash entry."""
    assert edgar_client.get_cik("BRK.B") == 1067983


def test_get_cik_resolves_class_b_dash_form(edgar_client: EdgarClient) -> None:
    """Passing the SEC native dash form must also work."""
    assert edgar_client.get_cik("BRK-B") == 1067983


def test_get_cik_resolves_class_a_and_bf_b(edgar_client: EdgarClient) -> None:
    assert edgar_client.get_cik("BRK.A") == 1067983
    assert edgar_client.get_cik("BF.B") == 14693


def test_get_cik_raises_for_unknown_ticker(edgar_client: EdgarClient) -> None:
    with pytest.raises(EdgarError, match="not found in SEC company_tickers index"):
        edgar_client.get_cik("NOPE")


# ---------------------------------------------------------------------------
# get_cik with a stubbed HTTP fetch — proves the lookup path works end-to-end
# even when the cache file is absent and a fresh SEC fetch is required.
# ---------------------------------------------------------------------------


def test_get_cik_brk_b_with_mocked_http_fetch(tmp_path: Path) -> None:
    """Stub the SEC HTTP call and confirm BRK.B resolves through the fresh-fetch path."""
    cache_dir = tmp_path / "edgar_fresh"
    cache_dir.mkdir(parents=True)
    client = EdgarClient(cache_dir=cache_dir)

    fake_response = MagicMock()
    fake_response.json.return_value = _SEC_INDEX_PAYLOAD
    fake_response.status_code = 200

    client._get = MagicMock(return_value=fake_response)  # type: ignore[method-assign]

    cik = client.get_cik("BRK.B")

    assert cik == 1067983
    client._get.assert_called_once()
    fetched_url = client._get.call_args.args[0]
    assert fetched_url.endswith("/files/company_tickers.json")
    # The fetch should have populated the disk cache for subsequent reads.
    assert (cache_dir / "company_tickers.json").exists()
