"""Tests for the config loader.

These never invoke `init_env()`; the test environment is set entirely via
`monkeypatch.setenv`, ensuring the project's real `.env` cannot leak in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantitative_trading.config import ConfigError, get_config


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear config cache and any inherited Anthropic/SEC env vars before each test."""
    get_config.cache_clear()
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "SEC_USER_AGENT", "DATA_DIR"):
        monkeypatch.delenv(var, raising=False)


def test_config_loads_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    cfg = get_config()

    assert cfg.anthropic_api_key == "sk-ant-test"
    assert cfg.sec_user_agent == "Test User test@example.com"
    assert cfg.data_dir == tmp_path / "data"
    assert cfg.edgar_cache_dir.exists()
    assert cfg.prices_cache_dir.exists()
    assert cfg.llm_cache_dir.exists()


def test_missing_anthropic_key_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        get_config()


def test_placeholder_sec_user_agent_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Your Name your-email@example.com")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(ConfigError, match="SEC_USER_AGENT"):
        get_config()


def test_missing_at_in_sec_user_agent_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "no email here")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(ConfigError, match="SEC_USER_AGENT"):
        get_config()


def test_default_data_dir_is_relative_to_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")

    cfg = get_config()

    assert cfg.data_dir.is_absolute()
    assert cfg.data_dir == cfg.project_root / "data"
