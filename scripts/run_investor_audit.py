"""Print the audit-plan section 7 statistical analyses on the prebuilt CSVs.

Reads `investor_purchases_audit.csv` and `control_sample.csv` (produced by
`build_investor_purchases.py`) and prints:

    audit plan section 7.A headline (per-criterion CMH + BH-FDR, the only inferential test)
    audit plan section 7.E.1 sensitivity (original-5 subset)
    audit plan section 7.E.2 sensitivity (full_filing_history subset)
    audit plan section 7.B realized-return analyses (KM + holding-stratified + 3y uncensored)
    audit plan section 7.C exploratory (per-investor table, n_pass distribution, criterion correlation)

Usage
-----
    python -m scripts.run_investor_audit [--data-dir data/investors]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quantitative_trading.config import init_env
from quantitative_trading.backtest.investor_audit_report import (
    criterion_correlation_matrix,
    format_headline_table,
    headline_full_filing_history_sensitivity,
    headline_original_five_sensitivity,
    headline_per_criterion_test,
    kaplan_meier_pass_vs_fail,
    n_criteria_distribution,
    per_investor_table,
    realized_returns_summary,
    three_year_uncensored_comparison,
)


def _print_section(title: str) -> None:
    print()
    print("=" * 95)
    print(title)
    print("=" * 95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/investors"))
    args = parser.parse_args()
    init_env()

    elite = pd.read_csv(args.data_dir / "investor_purchases_audit.csv")
    ctrl = pd.read_csv(args.data_dir / "control_sample.csv")

    _print_section("AUDIT PLAN SECTION 7.A HEADLINE: pooled 10 investors, per-criterion CMH + BH-FDR")
    print(format_headline_table(headline_per_criterion_test(elite, ctrl)))
    print("\n  This is the only inferential test (per audit plan section 7.D pre-registration).")
    print("  Read claims of 'criterion X discriminates elite buys from baseline' off the bh_q column.")

    _print_section("AUDIT PLAN SECTION 7.E.1 SENSITIVITY 1: original-5 subset only "
                   "(Munger, Pabrai, Li Lu, Akre, Spier)")
    print(format_headline_table(headline_original_five_sensitivity(elite, ctrl)))
    print("\n  Decision rule (audit plan section 7.E.1): if elite premia agree with the headline within ")
    print("  overlapping CIs, headline is robust to the n=10 sample composition. Material ")
    print("  divergence on any criterion -> composition-dependent finding.")

    _print_section("AUDIT PLAN SECTION 7.E.2 SENSITIVITY 2: full_filing_history subset only "
                   "(>=10y of pre-T_eval data)")
    print(format_headline_table(headline_full_filing_history_sensitivity(elite, ctrl)))
    print("\n  Decision rule (audit plan section 7.E.2): if elite premia agree with the headline, ")
    print("  the lookback-depth concern is empirically negligible.")

    _print_section("AUDIT PLAN SECTION 7.B SECONDARY: realized-return analyses (n_pass>=5 split)")
    km = kaplan_meier_pass_vs_fail(elite, split_threshold=5)
    print(f"  KM log-rank on time-to-exit:  high-pass n={km.group_a_n} (median {km.median_survival_a_quarters} q), "
          f"low-pass n={km.group_b_n} (median {km.median_survival_b_quarters} q), p={km.log_rank_p:.4f}")
    print()
    print("  Realized CAGR by (holding-bucket x n_pass>=5):")
    print(realized_returns_summary(elite, split_threshold=5).to_string(index=False))
    print()
    res = three_year_uncensored_comparison(elite, split_threshold=5)
    print(f"  3y uncensored (held >=12q): high-pass n={res['n_high_pass']} median CAGR {res['median_cagr_high_pass']*100:+.1f}%, "
          f"low-pass n={res['n_low_pass']} median {res['median_cagr_low_pass']*100:+.1f}%, MW p={res['mannwhitney_p']:.3f}")
    print()
    print("  Note: audit plan section 7.B originally specified all_seven_pass vs failed-1+, but ")
    print("  all_seven_pass=0/66 (the section 1 prediction). We split on n_pass>=5 (top-third by Town's bar).")

    _print_section("AUDIT PLAN SECTION 7.C EXPLORATORY: per-investor table (UNDERPOWERED -- DESCRIPTIVE ONLY per section 13.1)")
    print(per_investor_table(elite).to_string(index=False))

    _print_section("AUDIT PLAN SECTION 7.C item 5: n_criteria_passed distribution")
    print(n_criteria_distribution(elite, ctrl).round(3).to_string())

    _print_section("AUDIT PLAN SECTION 7.C item 7: criterion correlation matrix (pooled elite + control)")
    cm = criterion_correlation_matrix(elite, ctrl).round(2)
    print(cm.to_string())
    print("\n  Note the MoS <-> Payback correlation (audit plan section 7 item 7 prediction): ")
    print(f"  pooled corr = {cm.loc['pass_margin_of_safety', 'pass_payback_time']:.2f}")
    print("  -> Payback is mechanically derived from EPS+EPS-growth (per Town's formula),")
    print("     so it co-moves heavily with MoS. They are not 7 independent gates.")

    print()
    print("=" * 95)
    print("CAVEATS (per audit plan section 13, included in every output):")
    print("=" * 95)
    print("13.1: Per-investor pass-rates have wide CIs at n~5-30. Per-investor differences")
    print("      are NOT measured by this audit. Pooled headline (audit plan section 7.A) is the only inference.")
    print("13.2: The 10 investors are not a random sample of 'elite value investors';")
    print("      they were selected for value orientation. Findings apply to 'value-oriented")
    print("      13F filers as exemplified by these 10', not 'elite value investors broadly'.")
    print("13.3: None of the 10 ever claimed to follow Phil Town's strict 7-criteria bar.")
    print("      A LOW PASS-RATE IS EVIDENCE ABOUT TOWN'S BAR BEING OVER-STRICT RELATIVE TO ELITE")
    print("      PRACTICE -- IT IS NOT A CRITIQUE OF THESE INVESTORS' SKILL.")
    print("13.4: 13F structural limits: threshold-crossing and round-trip trades are unfixable")
    print("      from 13F alone. Headline window starts 2017 to mitigate Pabrai/Li Lu lookback gaps.")


if __name__ == "__main__":
    main()
