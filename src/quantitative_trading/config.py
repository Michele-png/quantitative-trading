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

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.edgar_cache_dir,
            self.prices_cache_dir,
            self.llm_cache_dir,
            self.universe_dir,
            self.dataset_dir,
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


@lru_cache(maxsize=1)
def get_config() -> Config:
    root = _project_root()
    data_dir_raw = os.environ.get("DATA_DIR", "data").strip()
    data_dir = Path(data_dir_raw)
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    cfg = Config(
        anthropic_api_key=_require("ANTHROPIC_API_KEY"),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929").strip(),
        sec_user_agent=_validate_sec_user_agent(_require("SEC_USER_AGENT")),
        data_dir=data_dir,
        project_root=root,
    )
    cfg.ensure_dirs()
    return cfg
