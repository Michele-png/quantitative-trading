"""Generate the standalone PNG figures used in `docs/REPORT.md`.

Run after `build_investor_purchases.py` has produced the audit CSVs.
Figures land in `docs/figures/`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantitative_trading.backtest.investor_audit_report import (
    CRITERIA_DISPLAY,
    criterion_correlation_matrix,
    headline_per_criterion_test,
    kaplan_meier_pass_vs_fail,
    n_criteria_distribution,
    per_investor_table,
)
from quantitative_trading.config import init_env

log = logging.getLogger(__name__)


# Color palette (consistent across figures).
ELITE_COLOR = "#2563eb"        # blue
CONTROL_COLOR = "#9ca3af"      # gray
HIGHLIGHT_COLOR = "#dc2626"    # red (significance markers)
ACCENT_COLOR = "#10b981"       # green (positive premia)
SECONDARY_COLOR = "#f59e0b"    # orange


def fig_pipeline_architecture(out_path: Path) -> None:
    """Schematic of the audit pipeline: 13F download -> scoring -> stats."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.3, 3.5, 1.6, 0.9, "Investor universe\n(10 CIKs, 12 filings)", ELITE_COLOR),
        (2.2, 3.5, 1.7, 0.9, "13F-HR downloader\n+ XML parser", ELITE_COLOR),
        (4.2, 3.5, 1.7, 0.9, "Purchase detection\n(first-ever appearance)", ELITE_COLOR),
        (6.2, 3.5, 1.7, 0.9, "CUSIP resolver\n(OpenFIGI + SEC)", ELITE_COLOR),
        (8.2, 3.5, 1.4, 0.9, "Rule One agent\n(7 criteria)", ELITE_COLOR),

        (0.3, 1.7, 1.6, 0.9, "S&P 500 universe\n(historical)", CONTROL_COLOR),
        (2.2, 1.7, 1.7, 0.9, "Sector/date matched\ncontrol sampler", CONTROL_COLOR),
        (4.2, 1.7, 1.7, 0.9, "K=10 controls\nper elite buy", CONTROL_COLOR),
        (6.2, 1.7, 1.7, 0.9, "Rule One agent\n(same 7 criteria)", CONTROL_COLOR),

        (3.2, 0.1, 4.0, 0.9, "Audit report\nCMH + BH-FDR + KM survival", HIGHLIGHT_COLOR),
    ]
    for x, y, w, h, txt, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.18,
                                    edgecolor=color, linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=8.5, color="black")

    arrows = [
        ((1.9, 3.95), (2.2, 3.95)),
        ((3.9, 3.95), (4.2, 3.95)),
        ((5.9, 3.95), (6.2, 3.95)),
        ((7.9, 3.95), (8.2, 3.95)),
        ((1.9, 2.15), (2.2, 2.15)),
        ((3.9, 2.15), (4.2, 2.15)),
        ((5.9, 2.15), (6.2, 2.15)),
        ((8.9, 3.5), (5.7, 1.0)),
        ((7.9, 1.7), (5.7, 1.0)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#374151", lw=1.2))

    ax.set_title("Figure 1. Audit pipeline architecture", fontsize=12, loc="left", pad=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_sample_funnel(elite: pd.DataFrame, out_path: Path) -> None:
    """Waterfall: 748 raw -> 547 clean -> ... -> 66 evaluable."""
    n_total = len(elite)
    n_clean = int((elite["lookback_completeness"] == "clean").sum())
    n_clean_evaluable = int(((elite["lookback_completeness"] == "clean")
                             & (elite["non_evaluable_reason"].isna())).sum())
    n_young = int((elite["non_evaluable_reason"] == "young_company").sum())
    n_finc = int((elite["non_evaluable_reason"] == "financial").sum())
    n_for = int((elite["non_evaluable_reason"] == "foreign_no_data").sum())
    n_unres = int((elite["non_evaluable_reason"] == "cusip_unresolved").sum())
    n_other_se = int((elite["non_evaluable_reason"] == "other_security_type").sum())
    n_holdco = int((elite["non_evaluable_reason"] == "holdco").sum())
    n_cik_unk = int((elite["non_evaluable_reason"] == "cik_unknown").sum())
    n_reinit = int((elite["lookback_completeness"] == "re_initiation").sum())
    n_incomplete = int((elite["lookback_completeness"] == "incomplete_lookback").sum())

    stages = [
        ("All NewPosition rows\n(detected from 13F)", n_total, "#1e3a8a"),
        ("Excluded: re-initiations", n_reinit, "#9ca3af"),
        ("Excluded: incomplete lookback", n_incomplete, "#9ca3af"),
        ("Clean first-ever positions", n_clean, ELITE_COLOR),
        ("Excluded: young company (<10y)", n_young, "#fbbf24"),
        ("Excluded: financial (SIC 60xx)", n_finc, "#fbbf24"),
        ("Excluded: foreign / ADR", n_for, "#fbbf24"),
        ("Excluded: CUSIP unresolved", n_unres, "#fbbf24"),
        ("Excluded: other security type", n_other_se, "#fbbf24"),
        ("Excluded: holdco / cik_unknown", n_holdco + n_cik_unk, "#fbbf24"),
        ("EVALUABLE (audit headline)", n_clean_evaluable, HIGHLIGHT_COLOR),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    colors = [s[2] for s in stages]
    bars = ax.barh(range(len(stages)), counts, color=colors, alpha=0.85)
    for i, c in enumerate(counts):
        ax.text(c + 5, i, str(c), va="center", fontsize=10, color="black")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of (investor, ticker, quarter) tuples")
    ax.set_title("Figure 3. Sample funnel: 748 raw new-position events -> 66 evaluable",
                 fontsize=12, loc="left", pad=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_per_criterion_pass_rates(headline: pd.DataFrame, out_path: Path) -> None:
    """Headline §7.A bar chart: elite vs control per criterion, with significance."""
    labels = [CRITERIA_DISPLAY[c] for c in headline["criterion"]]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars_e = ax.bar(x - width / 2, headline["elite_rate"] * 100, width,
                    label=f"Elite (n={int(headline.iloc[0].elite_n)})",
                    color=ELITE_COLOR)
    bars_c = ax.bar(x + width / 2, headline["control_rate"] * 100, width,
                    label=f"Sector/date control (n={int(headline.iloc[0].control_n)})",
                    color=CONTROL_COLOR)

    for i, (q, premium) in enumerate(zip(headline["bh_q"], headline["elite_premium_pp"], strict=False)):
        if q < 0.05:
            marker = "***"
        elif q < 0.10:
            marker = "*"
        else:
            marker = ""
        max_h = max(headline.iloc[i].elite_rate, headline.iloc[i].control_rate) * 100
        ax.text(i, max_h + 1.5, f"{marker}\n+{premium:.1f}pp", ha="center",
                va="bottom", fontsize=9, color=HIGHLIGHT_COLOR if marker else "#374151")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Pass rate (%)", fontsize=11)
    ax.set_ylim(0, 80)
    ax.set_title("Figure 4. Per-criterion pass rate -- elite buys vs sector/date-matched controls\n"
                 "*** = q<0.05 BH-FDR;  * = q<0.10",
                 fontsize=12, loc="left", pad=8)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_n_criteria_distribution(elite: pd.DataFrame, ctrl: pd.DataFrame, out_path: Path) -> None:
    dist = n_criteria_distribution(elite, ctrl)
    x = dist.index.astype(int)
    width = 0.4

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, dist["elite_share"] * 100, width, label="Elite", color=ELITE_COLOR)
    ax.bar(x + width / 2, dist["control_share"] * 100, width, label="Control", color=CONTROL_COLOR)
    ax.set_xticks(range(8))
    ax.set_xlabel("Number of Phil Town criteria passed (out of 7)", fontsize=11)
    ax.set_ylabel("Share of buys (%)", fontsize=11)
    ax.set_title("Figure 5. Distribution of n_criteria_passed -- elite vs control\n"
                 "Elite distribution is right-shifted on every count level >=3",
                 fontsize=12, loc="left", pad=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_criterion_correlation(elite: pd.DataFrame, ctrl: pd.DataFrame, out_path: Path) -> None:
    cm = criterion_correlation_matrix(elite, ctrl)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm)))
    ax.set_yticks(range(len(cm)))
    ax.set_xticklabels([CRITERIA_DISPLAY[c] for c in cm.columns],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels([CRITERIA_DISPLAY[c] for c in cm.index], fontsize=9)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = cm.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=10)
    ax.set_title("Figure 6. Criterion-correlation matrix (pooled elite + control)\n"
                 "Note MoS <-> Payback = 0.82: NOT independent gates",
                 fontsize=12, loc="left", pad=8)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Pearson correlation")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_kaplan_meier(elite: pd.DataFrame, out_path: Path) -> None:
    from lifelines import KaplanMeierFitter
    ev = elite[(elite.lookback_completeness == "clean") & (elite.non_evaluable_reason.isna())].copy()
    ev["event_observed"] = (~ev["is_right_censored"].astype(bool)).astype(int)
    high = ev[ev["n_criteria_passed"] >= 5]
    low = ev[ev["n_criteria_passed"] < 5]

    km_high = KaplanMeierFitter().fit(high["holding_period_quarters"],
                                       event_observed=high["event_observed"],
                                       label=f"n_pass>=5 (n={len(high)})")
    km_low = KaplanMeierFitter().fit(low["holding_period_quarters"],
                                      event_observed=low["event_observed"],
                                      label=f"n_pass<5 (n={len(low)})")

    res = kaplan_meier_pass_vs_fail(elite, split_threshold=5)
    fig, ax = plt.subplots(figsize=(9, 5))
    km_high.plot_survival_function(ax=ax, color=ELITE_COLOR)
    km_low.plot_survival_function(ax=ax, color=CONTROL_COLOR)
    ax.set_xlabel("Holding period (quarters)", fontsize=11)
    ax.set_ylabel("Probability still held", fontsize=11)
    ax.set_title(f"Figure 7. Kaplan-Meier: time-to-exit by criterion-pass cohort\n"
                 f"log-rank p = {res.log_rank_p:.3f} -- criteria do NOT predict holding tenure",
                 fontsize=12, loc="left", pad=8)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_per_investor(elite: pd.DataFrame, out_path: Path) -> None:
    """Per-investor stacked bar: clean evaluable vs non-evaluable subcategories."""
    pi = per_investor_table(elite).set_index("investor")
    cols = ["n_evaluable", "n_young_company", "n_financial",
            "n_foreign", "n_cusip_unresolved", "n_holdco"]
    colors = [HIGHLIGHT_COLOR, "#fbbf24", "#fb7185", "#a78bfa", "#9ca3af", "#f472b6"]
    labels_map = {
        "n_evaluable": "EVALUABLE",
        "n_young_company": "young company",
        "n_financial": "financial",
        "n_foreign": "foreign / ADR",
        "n_cusip_unresolved": "CUSIP unresolved",
        "n_holdco": "holdco",
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(pi))
    for col, c in zip(cols, colors, strict=False):
        ax.bar(pi.index, pi[col], bottom=bottom, color=c, label=labels_map[col])
        bottom = bottom + pi[col].values

    ax.set_xticklabels(pi.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of new positions (clean lookback)", fontsize=11)
    ax.set_title("Figure 8. Per-investor breakdown of clean-lookback new positions\n"
                 "by evaluability bucket (NB: per audit plan section 13.1, per-investor numbers are exploratory)",
                 fontsize=12, loc="left", pad=8)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_research_design(out_path: Path) -> None:
    """Schematic of the experimental design: window, T_eval, lookback, controls."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Timeline
    ax.annotate("", xy=(11.3, 4.5), xytext=(0.7, 4.5),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))
    for x, label in [(1.5, "first 13F\n(varies per investor,\n1999--2017)"),
                     (5.0, "T_eval\n= end of Q-1\n(audit plan section 3 PRIMARY)"),
                     (5.8, "Q\n(buy detected)"),
                     (10.0, "exit Q\n(or window end)")]:
        ax.plot([x, x], [4.4, 4.6], "k-", lw=1.2)
        ax.text(x, 4.2, label, ha="center", va="top", fontsize=9, color="#1f2937")

    ax.text(6.0, 4.85, "ANALYSIS WINDOW: 2017-Q1 -- 2024-Q4", ha="center",
            fontsize=10, fontweight="bold", color=HIGHLIGHT_COLOR)

    # Lookback bracket
    ax.annotate("", xy=(5.0, 3.7), xytext=(1.5, 3.7),
                arrowprops=dict(arrowstyle="<->", color=ELITE_COLOR, lw=1.4))
    ax.text(3.25, 3.5, "effective_lookback_quarters\n(uncapped, full available history)",
            ha="center", va="top", fontsize=9, color=ELITE_COLOR)

    # Conditions box
    ax.text(0.3, 2.6, "Each new position is tagged:", fontsize=10, fontweight="bold")
    conds = [
        "* clean: >=12q lookback AND ticker absent in all prior quarters -> enters headline",
        "* incomplete_lookback: <12q of pre-T_eval data -> separate stratum",
        "* re_initiation: ticker held in some prior quarter -> separate stratum",
        "* lookback_strategy: full_filing_history (>=40q) | truncated_to_first_filing (12-39q)",
    ]
    for i, c in enumerate(conds):
        ax.text(0.3, 2.3 - i * 0.32, c, fontsize=9, color="#374151")

    # Control sampling box
    ax.text(0.3, 0.85, "Control sampling per elite buy (audit plan section 6):", fontsize=10, fontweight="bold")
    ax.text(0.3, 0.55, "K=10 random S&P 500 members at T_eval, same SIC 2-digit sector,",
            fontsize=9, color="#374151")
    ax.text(0.3, 0.25, "same evaluability filters (>=10y XBRL, non-financial, non-holdco)",
            fontsize=9, color="#374151")

    ax.set_title("Figure 2. Research design: time window, T_eval, lookback strata, and control sampling",
                 fontsize=12, loc="left", pad=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/investors"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    init_env()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    elite = pd.read_csv(args.data_dir / "investor_purchases_audit.csv")
    ctrl = pd.read_csv(args.data_dir / "control_sample.csv")
    headline = headline_per_criterion_test(elite, ctrl)

    fig_pipeline_architecture(args.out_dir / "01_pipeline_architecture.png")
    fig_research_design(args.out_dir / "02_research_design.png")
    fig_sample_funnel(elite, args.out_dir / "03_sample_funnel.png")
    fig_per_criterion_pass_rates(headline, args.out_dir / "04_per_criterion_pass_rates.png")
    fig_n_criteria_distribution(elite, ctrl, args.out_dir / "05_n_criteria_distribution.png")
    fig_criterion_correlation(elite, ctrl, args.out_dir / "06_criterion_correlation.png")
    fig_kaplan_meier(elite, args.out_dir / "07_kaplan_meier.png")
    fig_per_investor(elite, args.out_dir / "08_per_investor_breakdown.png")
    print(f"Wrote 8 figures to {args.out_dir}")


if __name__ == "__main__":
    main()
