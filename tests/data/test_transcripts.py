"""Tests for the earnings-call transcript providers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from value_investing_backend.config import get_config
from value_investing_backend.data.transcripts import (
    DefaultTranscriptProvider,
    EarningsTranscript,
    FmpTranscriptProvider,
    Sec8KTranscriptProvider,
    _quarter_for_filing_date,
    _quarters_in_window,
)


@pytest.fixture(autouse=True)
def _isolate_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_config.cache_clear()


def _mock_fmp_response(
    *, status: int, payload: list | dict | None = None,
) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload if payload is not None else []
    return response


# --------------------------------------------------------------------------
# Quarter math
# --------------------------------------------------------------------------


def test_quarters_in_window_returns_quarter_ends() -> None:
    qs = _quarters_in_window(date(2023, 1, 1), date(2023, 12, 31))
    assert qs == [(2023, 1), (2023, 2), (2023, 3), (2023, 4)]


def test_quarter_for_8k_filing_date_maps_to_recent_quarter() -> None:
    assert _quarter_for_filing_date(date(2024, 5, 1)) == (2024, 1)
    assert _quarter_for_filing_date(date(2024, 8, 1)) == (2024, 2)
    assert _quarter_for_filing_date(date(2024, 11, 1)) == (2024, 3)
    assert _quarter_for_filing_date(date(2024, 2, 1)) == (2023, 4)


# --------------------------------------------------------------------------
# FmpTranscriptProvider
# --------------------------------------------------------------------------


def test_fmp_provider_fetches_caches_and_replays(tmp_path: Path) -> None:
    session = MagicMock()
    session.get.return_value = _mock_fmp_response(
        status=200,
        payload=[{
            "year": 2024, "quarter": 1, "date": "2024-04-25 17:00:00",
            "content": "x" * 6000,
        }],
    )
    provider = FmpTranscriptProvider(
        api_key="key", cache_dir=tmp_path / "cache", session=session,
    )
    out1 = provider.get_transcripts(
        "AAPL", start=date(2024, 1, 1), end=date(2024, 3, 31),
    )
    out2 = provider.get_transcripts(
        "AAPL", start=date(2024, 1, 1), end=date(2024, 3, 31),
    )
    assert len(out1) == 1
    assert out1[0].source == "fmp"
    assert out1[0].fiscal_year == 2024
    assert out2[0].text == out1[0].text
    # Second call served entirely from disk cache.
    assert session.get.call_count == 1


def test_fmp_provider_skips_remaining_after_402_rate_limit(tmp_path: Path) -> None:
    session = MagicMock()
    session.get.return_value = _mock_fmp_response(status=402)
    provider = FmpTranscriptProvider(
        api_key="key", cache_dir=tmp_path / "cache", session=session,
    )
    out = provider.get_transcripts(
        "AAPL", start=date(2024, 1, 1), end=date(2024, 12, 31),
    )
    assert out == []
    # First request triggers latch; subsequent quarters are not requested.
    assert session.get.call_count == 1


def test_fmp_provider_returns_empty_on_network_error(tmp_path: Path) -> None:
    session = MagicMock()
    session.get.side_effect = requests.ConnectionError()
    provider = FmpTranscriptProvider(
        api_key="key", cache_dir=tmp_path / "cache", session=session,
    )
    out = provider.get_transcripts(
        "X", start=date(2024, 1, 1), end=date(2024, 3, 31),
    )
    assert out == []


def test_fmp_provider_requires_api_key() -> None:
    with pytest.raises(ValueError):
        FmpTranscriptProvider(api_key="")


def test_fmp_provider_skips_empty_payload(tmp_path: Path) -> None:
    session = MagicMock()
    session.get.return_value = _mock_fmp_response(status=200, payload=[])
    provider = FmpTranscriptProvider(
        api_key="key", cache_dir=tmp_path / "cache", session=session,
    )
    out = provider.get_transcripts(
        "X", start=date(2024, 1, 1), end=date(2024, 3, 31),
    )
    assert out == []


# --------------------------------------------------------------------------
# DefaultTranscriptProvider
# --------------------------------------------------------------------------


def test_default_provider_uses_sec_only_when_no_fmp_key(tmp_path: Path) -> None:
    edgar = MagicMock()
    chain = DefaultTranscriptProvider(
        edgar_client=edgar, fmp_api_key=None, cache_dir=tmp_path / "cache",
    )
    assert len(chain._providers) == 1  # SEC only
    assert isinstance(chain._providers[0], Sec8KTranscriptProvider)


def test_default_provider_chains_fmp_then_sec(tmp_path: Path) -> None:
    edgar = MagicMock()
    chain = DefaultTranscriptProvider(
        edgar_client=edgar, fmp_api_key="some_key", cache_dir=tmp_path / "cache",
    )
    assert len(chain._providers) == 2
    assert chain._providers[0].name == "fmp"
    assert chain._providers[1].name == "sec_8k"


def test_default_provider_dedups_by_year_quarter(tmp_path: Path) -> None:
    edgar = MagicMock()
    chain = DefaultTranscriptProvider(
        edgar_client=edgar, fmp_api_key=None, cache_dir=tmp_path / "cache",
    )
    fake = MagicMock()
    fake.name = "fake"
    fake.get_transcripts.return_value = [
        EarningsTranscript("X", 2024, 1, date(2024, 4, 25), "fake", "a"),
        EarningsTranscript("X", 2024, 1, date(2024, 4, 25), "fake", "duplicate"),
        EarningsTranscript("X", 2024, 2, date(2024, 7, 25), "fake", "b"),
    ]
    chain._providers = [fake]
    out = chain.get_transcripts("X", date(2024, 1, 1), date(2024, 12, 31))
    assert [t.fiscal_quarter for t in out] == [1, 2]
