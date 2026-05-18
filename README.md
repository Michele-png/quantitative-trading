# value-investing-backend

Production Rule One value-investing library: the Python package that the
[`value-investing-dashboard`](https://github.com/Michele-png/value-investing-dashboard)
ETL imports and calls every week.

Implements [Phil Town's Rule One](https://www.ruleoneinvesting.com/) framework — Big 5 numbers, the 4Ms (Meaning, Moat, Management, Margin of Safety), Sticker Price — using LLM-driven qualitative analysis on SEC filings combined with code-driven quantitative analysis on as-filed financials.

Research, backtests, the historical dataset builder, the 13F investor audit, and the paper-trading orchestrator live in the sibling repo [`value-investing-experiments`](https://github.com/Michele-png/value-investing-experiments).

## What's in here

- **`src/value_investing_backend/agents/rule_one/`** — the value-investing agent. Big 5 ratios, Sticker Price / Margin of Safety, Payback Time, and an LLM-driven 4Ms analyzer that reads the 10-K and returns a structured pass/fail with rationale.
- **`src/value_investing_backend/data/`** — point-in-time data layer. SEC EDGAR client (filings as filed, not as later restated), yfinance wrapper with strict PIT cutoff, historical S&P 500 constituents, earnings transcripts, management documents (proxies, shareholder letters), insider trades.
- **`src/value_investing_backend/screening/`** — the dashboard-facing screening API: `refresh_records`, `screen_tickers`, `ScreenedRecord`. Pure functions that take `(tickers, as_of)` and return rows the dashboard persists to Supabase.

## Public API

```python
from value_investing_backend.screening import refresh_records

records = refresh_records(["AAPL", "MSFT"], date.today())
# Each record is a ScreenedRecord dataclass: Big 5 + Sticker + Payback +
# 4Ms (meaning / moat / management) + MoS percentage + evidence blob.
```

See [`src/value_investing_backend/screening/__init__.py`](src/value_investing_backend/screening/__init__.py) for the full public surface.

## Quick start

```bash
# 1. Install
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Set credentials
cp .env.example .env
# edit .env: ANTHROPIC_API_KEY and SEC_USER_AGENT (must be a real monitored email)

# 3. Run tests to verify the data layer is PIT-correct
pytest tests/

# 4. Smoke-test the EDGAR data layer end-to-end against live SEC
python scripts/smoke_test_edgar.py

# 5. Single-stock evaluation
python -m value_investing_backend.agents.rule_one.agent --ticker AAPL --date 2015-06-30
```

## Methodology

### The agent

For a given `(ticker, T)`, the agent recommends BUY iff **all** of:

| Pillar | Check | Computed by |
|---|---|---|
| Meaning | Business is understandable, durable demand | LLM (10-K Item 1) |
| Moat | Durable competitive advantage | LLM (10-K Item 1, 7) cross-checked vs. ROIC stability |
| Management | Trustworthy, owner-oriented capital allocation | LLM (10-K Item 7, proxy, transcripts, insider trades) |
| Big 5 #1 | ROIC ≥ 10% (10y avg, holding/rising) | code |
| Big 5 #2 | Sales growth ≥ 10% (10y CAGR) | code |
| Big 5 #3 | EPS growth ≥ 10% (10y CAGR) | code |
| Big 5 #4 | Equity growth ≥ 10% (10y CAGR) | code |
| Big 5 #5 | Operating cash flow growth ≥ 10% (10y CAGR) | code |
| Margin of Safety | Price ≤ 50% × Sticker Price | code |
| Payback Time | < 8 years | code |

## Critical risks the design addresses

- **Look-ahead via restated financials** — EDGAR XBRL data is keyed on filing accession + filing date, not period end. Tested against known restatements.
- **Survivorship bias** — historical index membership snapshots, not today's list.
- **LLM training-data leakage** — addressed by experiments-repo ablations (ticker-masked variant).
- **Cost** — aggressive caching at all layers; LLM calls keyed by fiscal year, not quarter.

## Project layout

```
src/value_investing_backend/
  data/               # PIT EDGAR + prices + universe + cache + transcripts + insider trades
  agents/rule_one/    # the agent (big five, sticker, payback, LLM 4Ms, management, decision)
  screening/          # dashboard-facing API: refresh_records, ScreenedRecord
tests/                # pytest suite, focused on PIT correctness
scripts/              # smoke_test_edgar.py
data/                 # local cache (gitignored)
```

## Status

Active library. Not investment advice.
