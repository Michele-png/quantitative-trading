"""Configuration for the weekly paper-trading workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from quantitative_trading.config import ConfigError, get_config

DEFAULT_LEDGER_RELATIVE_PATH = "paper_trading/ledger.jsonl"


def _require_env(var: str) -> str:
    value = os.environ.get(var, "").strip()
    if not value:
        raise ConfigError(f"Required paper-trading environment variable {var!r} is missing.")
    return value


def _optional_env(var: str) -> str | None:
    value = os.environ.get(var, "").strip()
    return value or None


def _require_float(
    var: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = _require_env(var)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{var} must be numeric, got {raw!r}.") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{var} must be >= {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{var} must be <= {maximum}, got {value}.")
    return value


@dataclass(frozen=True)
class PaperTradingConfig:
    """All runtime settings for the weekly paper-trading agent."""

    anthropic_api_key: str
    anthropic_model: str
    alpaca_api_key_id: str
    alpaca_secret_key: str
    alpaca_base_url: str
    polygon_api_key: str
    initial_budget_eur: float
    eur_usd_rate_source: str
    italy_capital_gains_tax_rate: float
    data_dir: Path
    ledger_path: Path
    max_candidates_for_llm: int = 10
    preselect_count: int = 25
    reserve_cash_usd: float = 5.0

    def __post_init__(self) -> None:
        if self.alpaca_base_url != "https://paper-api.alpaca.markets":
            raise ConfigError(
                "APCA_API_BASE_URL must be exactly 'https://paper-api.alpaca.markets' "
                "for this paper-only workflow."
            )
        if self.eur_usd_rate_source != "polygon":
            raise ConfigError("EUR_USD_RATE_SOURCE must be 'polygon' for reliable FX retrieval.")
        if self.max_candidates_for_llm < 1:
            raise ConfigError("max_candidates_for_llm must be at least 1.")
        if self.preselect_count < self.max_candidates_for_llm:
            raise ConfigError("preselect_count must be >= max_candidates_for_llm.")
        if self.reserve_cash_usd < 0:
            raise ConfigError("reserve_cash_usd must be non-negative.")

    @property
    def paper_trading_dir(self) -> Path:
        """Directory for paper-trading artifacts."""
        return self.data_dir / "paper_trading"

    def ensure_dirs(self) -> None:
        """Create directories needed by the paper-trading workflow."""
        self.paper_trading_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_paper_trading_config() -> PaperTradingConfig:
    """Load and validate paper-trading settings from the current environment."""
    base = get_config()
    ledger_raw = os.environ.get("PAPER_TRADING_LEDGER", DEFAULT_LEDGER_RELATIVE_PATH).strip()
    ledger_path = Path(ledger_raw)
    if not ledger_path.is_absolute():
        ledger_path = base.data_dir / ledger_path

    cfg = PaperTradingConfig(
        anthropic_api_key=base.anthropic_api_key,
        anthropic_model=base.anthropic_model,
        alpaca_api_key_id=_require_env("ALPACA_API_KEY_ID"),
        alpaca_secret_key=_require_env("ALPACA_SECRET_KEY"),
        alpaca_base_url=_require_env("APCA_API_BASE_URL").rstrip("/"),
        polygon_api_key=_require_env("POLYGON_API_KEY"),
        initial_budget_eur=_require_float("INITIAL_BUDGET_EUR", minimum=0.01),
        eur_usd_rate_source=_require_env("EUR_USD_RATE_SOURCE"),
        italy_capital_gains_tax_rate=_require_float(
            "ITALY_CAPITAL_GAINS_TAX_RATE", minimum=0.0, maximum=1.0
        ),
        data_dir=base.data_dir,
        ledger_path=ledger_path,
        max_candidates_for_llm=int(_optional_env("MAX_CANDIDATES_FOR_LLM") or "10"),
        preselect_count=int(_optional_env("PRESELECT_COUNT") or "25"),
        reserve_cash_usd=float(_optional_env("RESERVE_CASH_USD") or "5.0"),
    )
    cfg.ensure_dirs()
    return cfg
