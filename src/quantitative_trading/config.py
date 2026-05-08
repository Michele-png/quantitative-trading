"""Centralized configuration loaded from environment / .env file.

Single source of truth for credentials, paths, and tunables shared across the
package.

Loading semantics:
    * `init_env()` reads the project's `.env` file into `os.environ` (without
      overriding values already set). CLI entry points should call this once at
      startup. Library code should assume the environment is already populated.
    * `get_config()` reads from `os.environ` and validates. Cached.

Tests should never call `init_env()`; they manipulate `os.environ` directly via
`monkeypatch` and call `get_config.cache_clear()` between cases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or invalid."""


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str
    anthropic_model: str
    sec_user_agent: str
    data_dir: Path
    project_root: Path
    fmp_api_key: str | None
    anthropic_thinking_budget_tokens: int
    anthropic_max_input_chars: int

    @property
    def edgar_cache_dir(self) -> Path:
        return self.data_dir / "edgar"

    @property
    def prices_cache_dir(self) -> Path:
        return self.data_dir / "prices"

    @property
    def llm_cache_dir(self) -> Path:
        return self.data_dir / "llm"

    @property
    def universe_dir(self) -> Path:
        return self.data_dir / "universe"

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / "dataset"

    @property
    def transcripts_cache_dir(self) -> Path:
        return self.data_dir / "transcripts"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.edgar_cache_dir,
            self.prices_cache_dir,
            self.llm_cache_dir,
            self.universe_dir,
            self.dataset_dir,
            self.transcripts_cache_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def init_env(env_file: Path | None = None) -> None:
    """Load .env into os.environ. Call once at app startup, never from tests."""
    target = env_file if env_file is not None else _project_root() / ".env"
    load_dotenv(target, override=False)
    get_config.cache_clear()


def _require(var: str) -> str:
    value = os.environ.get(var, "").strip()
    if not value:
        raise ConfigError(
            f"Required environment variable {var!r} is missing. "
            f"Copy .env.example to .env, fill it in, and call init_env()."
        )
    return value


def _validate_sec_user_agent(value: str) -> str:
    # SEC requires a User-Agent that identifies a real, contactable requester.
    if "@" not in value or value.startswith("Your Name"):
        raise ConfigError(
            f"SEC_USER_AGENT must be a real 'Name email@domain' (got: {value!r}). "
            f"SEC bans User-Agents that don't identify the requester."
        )
    return value


def _parse_int(var: str, default: int) -> int:
    raw = os.environ.get(var, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{var} must be an integer (got {raw!r}).") from exc


@lru_cache(maxsize=1)
def get_config() -> Config:
    root = _project_root()
    data_dir_raw = os.environ.get("DATA_DIR", "data").strip()
    data_dir = Path(data_dir_raw)
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    fmp_api_key_raw = os.environ.get("FMP_API_KEY", "").strip()

    cfg = Config(
        anthropic_api_key=_require("ANTHROPIC_API_KEY"),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7").strip(),
        sec_user_agent=_validate_sec_user_agent(_require("SEC_USER_AGENT")),
        data_dir=data_dir,
        project_root=root,
        fmp_api_key=fmp_api_key_raw or None,
        # Extended-thinking budget for Opus 4.7. 32k is conservative; the model
        # supports more, but each thinking token is billed as output ($25/MTok).
        anthropic_thinking_budget_tokens=_parse_int(
            "ANTHROPIC_THINKING_BUDGET_TOKENS", 32_000
        ),
        # ~1M-token context for Opus 4.7. We translate input-token budget into
        # an approximate character cap (4 chars/token average) used by the
        # management pipeline when packing many documents.
        anthropic_max_input_chars=_parse_int(
            "ANTHROPIC_MAX_INPUT_CHARS", 3_500_000
        ),
    )
    cfg.ensure_dirs()
    return cfg
