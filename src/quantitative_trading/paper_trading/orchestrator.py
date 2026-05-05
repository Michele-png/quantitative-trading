"""End-to-end orchestration for one weekly paper-trading run."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, timedelta

from anthropic import Anthropic

from quantitative_trading.paper_trading.broker import AlpacaPaperBroker
from quantitative_trading.paper_trading.candidate_scoring import CandidateScorer
from quantitative_trading.paper_trading.config import PaperTradingConfig
from quantitative_trading.paper_trading.cost_accounting import TransactionCostEstimator
from quantitative_trading.paper_trading.fx import PolygonFxClient
from quantitative_trading.paper_trading.guardrails import TradeGuardrails
from quantitative_trading.paper_trading.ledger import PaperTradingLedger
from quantitative_trading.paper_trading.market_context import (
    MarketContextBuilder,
    PolygonMarketDataClient,
    SP500UniverseProvider,
)
from quantitative_trading.paper_trading.models import AccountSnapshot, WeeklyRunResult
from quantitative_trading.paper_trading.weekly_decision import ZeroShotDecisionMaker


class WeeklyPaperTrader:
    """Run the weekly decision, guardrail, accounting, and optional execution flow."""

    def __init__(
        self,
        *,
        config: PaperTradingConfig,
        broker: AlpacaPaperBroker | None = None,
        ledger: PaperTradingLedger | None = None,
        context_builder: MarketContextBuilder | None = None,
        decision_maker: ZeroShotDecisionMaker | None = None,
        fx_client: PolygonFxClient | None = None,
        cost_estimator: TransactionCostEstimator | None = None,
    ) -> None:
        self.config = config
        self.broker = broker or AlpacaPaperBroker(
            api_key_id=config.alpaca_api_key_id,
            secret_key=config.alpaca_secret_key,
            base_url=config.alpaca_base_url,
        )
        self.ledger = ledger or PaperTradingLedger(config.ledger_path)
        polygon = PolygonMarketDataClient(config.polygon_api_key)
        self.context_builder = context_builder or MarketContextBuilder(
            polygon=polygon,
            universe_provider=SP500UniverseProvider(),
            scorer=CandidateScorer(),
        )
        anthropic_client = Anthropic(api_key=config.anthropic_api_key)
        self.decision_maker = decision_maker or ZeroShotDecisionMaker(
            anthropic_client=anthropic_client,
            model=config.anthropic_model,
        )
        self.fx_client = fx_client or PolygonFxClient(config.polygon_api_key)
        self.cost_estimator = cost_estimator or TransactionCostEstimator()
        self.guardrails = TradeGuardrails(reserve_cash_usd=config.reserve_cash_usd)

    def run(
        self,
        *,
        dry_run: bool,
        trade_week: date | None = None,
        symbols: list[str] | None = None,
        require_market_open: bool = True,
    ) -> WeeklyRunResult:
        """Run a weekly dry-run or paper execution."""
        target_week = trade_week or current_trade_week()
        if not dry_run and self.ledger.has_executed_week(target_week):
            raise RuntimeError(f"Trade week {target_week.isoformat()} has already been executed.")
        if not dry_run and require_market_open and not self.broker.is_market_open():
            raise RuntimeError("Market is not open according to Alpaca clock.")

        account = self.broker.get_account()
        context = self.context_builder.build(
            trade_week=target_week,
            symbols=symbols,
            preselect_count=self.config.preselect_count,
            max_candidates=self.config.max_candidates_for_llm,
        )
        decision = self.decision_maker.decide(context)
        allowed_symbols = {candidate.symbol for candidate in context.candidates}
        fx_rate = self.fx_client.get_eur_usd(target_week).eur_usd
        budget_account = budget_capped_account(
            account=account,
            initial_budget_eur=self.config.initial_budget_eur,
            eur_usd_rate=fx_rate,
        )
        order_plan = self.guardrails.build_order_plan(
            decision=decision,
            account=budget_account,
            allowed_symbols=allowed_symbols,
            trade_week=target_week,
        )
        selected = next(
            candidate for candidate in context.candidates if candidate.symbol == decision.ticker
        )
        estimated_shares = order_plan.notional_usd / selected.price
        buy_costs = self.cost_estimator.estimate(
            side="buy",
            notional_usd=order_plan.notional_usd,
            shares=estimated_shares,
        )

        order_result = None
        event_type = "dry_run_week"
        if not dry_run:
            self.broker.close_all_positions()
            order_result = self.broker.submit_notional_market_buy(
                symbol=order_plan.symbol,
                notional_usd=order_plan.notional_usd,
            )
            event_type = "executed_week"

        result = WeeklyRunResult(
            trade_week=target_week,
            dry_run=dry_run,
            decision=decision,
            order_plan=order_plan,
            account=account,
            fx_rate_eur_usd=fx_rate,
            estimated_buy_costs=buy_costs,
            order_result=order_result,
            ledger_path=str(self.ledger.path),
        )
        self.ledger.append(
            event_type=event_type,
            trade_week=target_week,
            payload={
                "result": asdict(result),
                "context": context.to_prompt_dict(),
            },
        )
        return result


def current_trade_week(today: date | None = None) -> date:
    """Return the Monday that identifies the current trade week."""
    target = today or date.today()
    return target - timedelta(days=target.weekday())


def budget_capped_account(
    *,
    account: AccountSnapshot,
    initial_budget_eur: float,
    eur_usd_rate: float,
) -> AccountSnapshot:
    """Cap available order size to cash and the experiment's EUR budget."""
    if initial_budget_eur <= 0:
        raise ValueError("initial_budget_eur must be positive.")
    if eur_usd_rate <= 0:
        raise ValueError("eur_usd_rate must be positive.")
    budget_cap_usd = initial_budget_eur * eur_usd_rate
    cash_available = max(account.cash, 0.0)
    cash_and_budget_cap = min(account.buying_power, cash_available, budget_cap_usd)
    return replace(account, buying_power=cash_and_budget_cap)
