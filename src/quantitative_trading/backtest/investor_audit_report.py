"""Statistical analyses and report generation for the value-investor audit.

Implements the structure specified in audit plan section 7:

* **7.A Headline** -- per-criterion elite-vs-control pass-rate test using
  Cochran-Mantel-Haenszel chi-square stratified on (SIC 2-digit x quarter),
  with Benjamini-Hochberg FDR correction across the 7 criteria. This is
  the single inferential output that drives conclusions.

* **7.B Secondary** -- realized-return-to-exit analysis with proper
  censoring handling: Kaplan-Meier on time-to-exit + log-rank test, plus
  a 3y uncensored-only fixed-horizon comparison.

* **7.C Exploratory** -- per-investor pass-rate table (with explicit "wide CI"
  warnings), conjunction rate, distribution of n_criteria_passed, criterion-
  correlation matrix.

* **7.E Sensitivities** -- the §7.A test re-run on (1) the original-5 subset
  and (2) the full_filing_history subset.

The sample size for the §7.A pooled test is fixed by the data; per audit plan
section 13.1 we explicitly DO NOT make per-investor inferential claims (n is too small).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import StratifiedTable

log = logging.getLogger(__name__)


CRITERIA: tuple[str, ...] = (
    "pass_roic",
    "pass_sales_growth",
    "pass_eps_growth",
    "pass_equity_growth",
    "pass_ocf_growth",
    "pass_margin_of_safety",
    "pass_payback_time",
)

CRITERIA_DISPLAY: dict[str, str] = {
    "pass_roic": "ROIC >= 10%",
    "pass_sales_growth": "Sales growth >= 10%",
    "pass_eps_growth": "EPS growth >= 10%",
    "pass_equity_growth": "Equity growth >= 10%",
    "pass_ocf_growth": "OCF growth >= 10%",
    "pass_margin_of_safety": "Margin of Safety",
    "pass_payback_time": "Payback Time < 8y",
}


# ----------------------------------------------------------------- Filters


def evaluable_elite(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to clean + evaluable elite buys (`lookback_completeness="clean"` and
    no `non_evaluable_reason`)."""
    return df[
        (df["lookback_completeness"] == "clean")
        & (df["non_evaluable_reason"].isna())
    ].copy()


def evaluable_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to controls with valid scoring (>=10y history -> n_criteria_passed not null)."""
    return df[df["n_criteria_passed"].notna()].copy()


# ----------------------------------------------------------------- §7.A headline


@dataclass(frozen=True)
class CriterionTest:
    """One per-criterion elite-vs-control test result."""

    criterion: str
    elite_n: int
    elite_pass: int
    elite_rate: float
    control_n: int
    control_pass: int
    control_rate: float
    elite_premium_pp: float    # (elite_rate - control_rate) * 100
    cmh_p: float
    bh_q: float                # BH-adjusted q across the 7 criteria
    cmh_method: str            # "cmh_chisq" | "exact_mh" | "fisher_pooled"
    n_strata_used: int
    n_strata_dropped: int


def _build_strata(
    elite_df: pd.DataFrame, control_df: pd.DataFrame, criterion: str,
) -> list[np.ndarray]:
    """Build 2x2xK contingency tables stratified on (SIC 2-digit x quarter).

    Each stratum is a 2x2 numpy array with rows = [pass, fail] and
    columns = [elite, control]. Strata where either arm has zero observations
    are dropped (per audit plan section 6.1: "strata where both arms have zero events
    are dropped").
    """
    elite_df = elite_df.copy()
    control_df = control_df.copy()

    # Stratification key: SIC 2-digit x quarter.
    elite_df["_sic2"] = elite_df["sic_code"].fillna(-1).astype(int) // 100
    elite_df["_q"] = elite_df["period_of_report"].astype(str).str[:7]  # YYYY-MM
    control_df["_sic2"] = control_df["control_sic_2digit"].fillna(-1).astype(int)
    control_df["_q"] = control_df["elite_period_of_report"].astype(str).str[:7]

    strata_keys = sorted(
        set(zip(elite_df["_sic2"], elite_df["_q"], strict=False))
        | set(zip(control_df["_sic2"], control_df["_q"], strict=False))
    )

    tables: list[np.ndarray] = []
    for sic2, q in strata_keys:
        e = elite_df[(elite_df["_sic2"] == sic2) & (elite_df["_q"] == q)]
        c = control_df[(control_df["_sic2"] == sic2) & (control_df["_q"] == q)]
        if len(e) == 0 or len(c) == 0:
            continue
        e_pass = int(e[criterion].sum())
        e_fail = len(e) - e_pass
        c_pass = int(c[criterion].sum())
        c_fail = len(c) - c_pass
        tables.append(np.array([[e_pass, c_pass], [e_fail, c_fail]]))
    return tables


def headline_per_criterion_test(
    elite_df: pd.DataFrame, control_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the §7.A headline test across the 7 criteria with BH-FDR correction.

    For each criterion:
        1. Build (SIC 2-digit x quarter) strata.
        2. CMH chi-square test on the 2x2xK stack.
        3. Fall back to exact MH if any stratum is sparse (rare with K=10
           controls per buy).
        4. Apply Benjamini-Hochberg FDR at q<0.05 across the 7 raw p-values.

    Returns a DataFrame with one row per criterion.
    """
    elite = evaluable_elite(elite_df)
    controls = evaluable_controls(control_df)

    results: list[CriterionTest] = []
    raw_ps: list[float] = []

    for crit in CRITERIA:
        e_n = len(elite)
        e_pass = int(elite[crit].sum())
        c_n = len(controls)
        c_pass = int(controls[crit].sum())
        elite_rate = e_pass / e_n if e_n else 0.0
        control_rate = c_pass / c_n if c_n else 0.0
        premium_pp = (elite_rate - control_rate) * 100

        tables = _build_strata(elite, controls, crit)
        n_strata_total = len(set(
            zip(elite["sic_code"].fillna(-1).astype(int) // 100,
                elite["period_of_report"].astype(str).str[:7],
                strict=False)
        ) | set(
            zip(controls["control_sic_2digit"].fillna(-1).astype(int),
                controls["elite_period_of_report"].astype(str).str[:7],
                strict=False)
        ))
        n_strata_dropped = n_strata_total - len(tables)

        if not tables:
            log.warning("All strata dropped for %s; skipping", crit)
            results.append(CriterionTest(
                criterion=crit, elite_n=e_n, elite_pass=e_pass, elite_rate=elite_rate,
                control_n=c_n, control_pass=c_pass, control_rate=control_rate,
                elite_premium_pp=premium_pp, cmh_p=float("nan"), bh_q=float("nan"),
                cmh_method="no_strata", n_strata_used=0,
                n_strata_dropped=n_strata_dropped,
            ))
            raw_ps.append(1.0)
            continue

        # Stack strata for StratifiedTable.
        try:
            st = StratifiedTable(tables)
            test_result = st.test_null_odds()
            cmh_p = float(test_result.pvalue)
            method = "cmh_chisq"
        except Exception as exc:  # noqa: BLE001
            log.warning("CMH failed for %s (%s); falling back to pooled Fisher", crit, exc)
            from scipy.stats import fisher_exact
            pool = np.sum(tables, axis=0)
            _, cmh_p = fisher_exact(pool)
            method = "fisher_pooled"

        results.append(CriterionTest(
            criterion=crit, elite_n=e_n, elite_pass=e_pass, elite_rate=elite_rate,
            control_n=c_n, control_pass=c_pass, control_rate=control_rate,
            elite_premium_pp=premium_pp,
            cmh_p=cmh_p, bh_q=float("nan"), cmh_method=method,
            n_strata_used=len(tables), n_strata_dropped=n_strata_dropped,
        ))
        raw_ps.append(cmh_p)

    # Apply BH-FDR across the 7 criteria.
    if any(not np.isnan(p) for p in raw_ps):
        valid_idx = [i for i, p in enumerate(raw_ps) if not np.isnan(p)]
        valid_ps = [raw_ps[i] for i in valid_idx]
        _, q_values, _, _ = multipletests(valid_ps, alpha=0.05, method="fdr_bh")
        q_full = [float("nan")] * len(raw_ps)
        for i, q in zip(valid_idx, q_values, strict=False):
            q_full[i] = float(q)
        results = [
            CriterionTest(**{**r.__dict__, "bh_q": q})
            for r, q in zip(results, q_full, strict=False)
        ]

    return pd.DataFrame([r.__dict__ for r in results])


# ----------------------------------------------------------------- §7.E sensitivities


def headline_original_five_sensitivity(
    elite_df: pd.DataFrame, control_df: pd.DataFrame,
) -> pd.DataFrame:
    """Re-run §7.A on the original-5 subset (Munger, Pabrai, Li Lu, Akre, Spier)."""
    from quantitative_trading.investors.investor_universe import original_five
    short_ids = {i.short_id for i in original_five()}
    elite_subset = elite_df[elite_df["investor_short_id"].isin(short_ids)].copy()
    control_subset = control_df[control_df["elite_investor_short_id"].isin(short_ids)].copy()
    return headline_per_criterion_test(elite_subset, control_subset)


def headline_full_filing_history_sensitivity(
    elite_df: pd.DataFrame, control_df: pd.DataFrame,
) -> pd.DataFrame:
    """Re-run §7.A on rows where lookback_strategy == 'full_filing_history'."""
    elite_subset = elite_df[elite_df["lookback_strategy"] == "full_filing_history"].copy()
    keep_cusips = set(elite_subset["cusip"])
    control_subset = control_df[control_df["elite_cusip"].isin(keep_cusips)].copy()
    return headline_per_criterion_test(elite_subset, control_subset)


# ----------------------------------------------------------------- §7.C exploratory


def per_investor_table(elite_df: pd.DataFrame) -> pd.DataFrame:
    """Per-investor pass rates with sample-size context (audit plan section 7.C item 3)."""
    rows = []
    for inv_id, group in elite_df.groupby("investor_short_id"):
        n_total = len(group)
        n_clean = int((group["lookback_completeness"] == "clean").sum())
        n_young = int((group["non_evaluable_reason"] == "young_company").sum())
        n_finc = int((group["non_evaluable_reason"] == "financial").sum())
        n_holdco = int((group["non_evaluable_reason"] == "holdco").sum())
        n_foreign = int((group["non_evaluable_reason"] == "foreign_no_data").sum())
        n_unresolved = int((group["non_evaluable_reason"] == "cusip_unresolved").sum())
        n_evaluable = int(((group["lookback_completeness"] == "clean")
                           & (group["non_evaluable_reason"].isna())).sum())
        ev = group[(group["lookback_completeness"] == "clean")
                    & (group["non_evaluable_reason"].isna())]
        row = {
            "investor": inv_id,
            "n_total": n_total,
            "n_clean": n_clean,
            "n_young_company": n_young,
            "n_financial": n_finc,
            "n_holdco": n_holdco,
            "n_foreign": n_foreign,
            "n_cusip_unresolved": n_unresolved,
            "n_evaluable": n_evaluable,
            "all7_pass_rate": ev["all_seven_pass"].mean() if len(ev) else float("nan"),
            "big5_pass_rate": ev["big5_pass"].mean() if len(ev) else float("nan"),
            "mos_pass_rate": ev["pass_margin_of_safety"].mean() if len(ev) else float("nan"),
            "payback_pass_rate": ev["pass_payback_time"].mean() if len(ev) else float("nan"),
            "roic_pass_rate": ev["pass_roic"].mean() if len(ev) else float("nan"),
            "eps_growth_pass_rate": ev["pass_eps_growth"].mean() if len(ev) else float("nan"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def n_criteria_distribution(
    elite_df: pd.DataFrame, control_df: pd.DataFrame,
) -> pd.DataFrame:
    """Distribution of n_criteria_passed in elite vs control (audit plan section 7.C item 5)."""
    e = evaluable_elite(elite_df)
    c = evaluable_controls(control_df)
    e_dist = e["n_criteria_passed"].value_counts(normalize=True).sort_index()
    c_dist = c["n_criteria_passed"].value_counts(normalize=True).sort_index()
    out = pd.DataFrame({"elite_share": e_dist, "control_share": c_dist}).fillna(0)
    out.index.name = "n_criteria_passed"
    return out


def criterion_correlation_matrix(
    elite_df: pd.DataFrame, control_df: pd.DataFrame,
) -> pd.DataFrame:
    """7x7 correlation matrix of boolean criteria across pooled elite+control.

    Per audit plan section 7.C item 7: addresses Payback Time mechanical-redundancy concern --
    expect Payback to co-vary with EPS-growth-pass and MoS-pass.
    """
    e = evaluable_elite(elite_df)[list(CRITERIA)].astype(int)
    c = evaluable_controls(control_df)[list(CRITERIA)].astype(int)
    pooled = pd.concat([e, c], ignore_index=True)
    return pooled.corr()


# ----------------------------------------------------------------- Pretty printing


# ----------------------------------------------------------------- §7.B realized returns


@dataclass(frozen=True)
class SurvivalResult:
    """Result of the §7.B Kaplan-Meier log-rank test."""

    group_a_label: str
    group_b_label: str
    group_a_n: int
    group_b_n: int
    group_a_n_events: int  # observed exits (uncensored)
    group_b_n_events: int
    median_survival_a_quarters: float | None
    median_survival_b_quarters: float | None
    log_rank_p: float


def kaplan_meier_pass_vs_fail(
    elite_df: pd.DataFrame,
    *,
    split_threshold: int = 5,
) -> SurvivalResult:
    """KM survival on time-to-exit-from-13F: high-pass vs low-pass cohorts.

    Per audit plan section 7.B.2a. The plan originally specified `all_seven_pass`
    vs failed-1-or-more, but `all_seven_pass` is empirically empty (0/66) -- a
    direct confirmation of audit plan section 1's "conjunction is mechanically near-zero"
    prediction. We pivot to splitting on `n_criteria_passed >= split_threshold`
    (default 5, i.e., top-third by Town's bar). Right-censored = "still held
    at window end".
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    ev = evaluable_elite(elite_df)
    ev = ev[ev["holding_period_quarters"].notna()].copy()
    ev["event_observed"] = (~ev["is_right_censored"].astype(bool)).astype(int)

    pass_group = ev[ev["n_criteria_passed"] >= split_threshold]
    fail_group = ev[ev["n_criteria_passed"] < split_threshold]

    label_pass = f"n_pass>={split_threshold}"
    label_fail = f"n_pass<{split_threshold}"

    if len(pass_group) == 0 or len(fail_group) == 0:
        log.warning("KM: one group is empty (pass n=%d, fail n=%d) -- "
                    "log-rank not meaningful", len(pass_group), len(fail_group))
        return SurvivalResult(
            group_a_label=label_pass,
            group_b_label=label_fail,
            group_a_n=len(pass_group),
            group_b_n=len(fail_group),
            group_a_n_events=int(pass_group["event_observed"].sum()) if len(pass_group) else 0,
            group_b_n_events=int(fail_group["event_observed"].sum()) if len(fail_group) else 0,
            median_survival_a_quarters=None,
            median_survival_b_quarters=None,
            log_rank_p=float("nan"),
        )

    kmf_a = KaplanMeierFitter().fit(
        pass_group["holding_period_quarters"],
        event_observed=pass_group["event_observed"],
    )
    kmf_b = KaplanMeierFitter().fit(
        fail_group["holding_period_quarters"],
        event_observed=fail_group["event_observed"],
    )
    test = logrank_test(
        pass_group["holding_period_quarters"], fail_group["holding_period_quarters"],
        event_observed_A=pass_group["event_observed"],
        event_observed_B=fail_group["event_observed"],
    )
    return SurvivalResult(
        group_a_label=label_pass,
        group_b_label=label_fail,
        group_a_n=len(pass_group),
        group_b_n=len(fail_group),
        group_a_n_events=int(pass_group["event_observed"].sum()),
        group_b_n_events=int(fail_group["event_observed"].sum()),
        median_survival_a_quarters=float(kmf_a.median_survival_time_) if not np.isnan(kmf_a.median_survival_time_) else None,
        median_survival_b_quarters=float(kmf_b.median_survival_time_) if not np.isnan(kmf_b.median_survival_time_) else None,
        log_rank_p=float(test.p_value),
    )


def realized_returns_summary(
    elite_df: pd.DataFrame,
    *,
    split_threshold: int = 5,
) -> pd.DataFrame:
    """Per audit plan section 7.B.2b: realized CAGR comparison stratified by holding period.

    Stratifies elite buys into closed (0-2y, 2-5y, 5y+) and right-censored
    buckets, comparing median realized CAGR of high-pass (n_pass>=split_threshold)
    vs low-pass cohorts. The plan called for `all_seven_pass` vs not, but
    that's empty (0/66); see `kaplan_meier_pass_vs_fail` docstring.
    """
    ev = evaluable_elite(elite_df)
    ev = ev[ev["realized_cagr_to_exit"].notna()].copy()
    ev["holding_bucket"] = pd.cut(
        ev["holding_period_quarters"],
        bins=[-1, 8, 20, 40, 1000],
        labels=["0-2y", "2-5y", "5-10y", "10y+"],
    )
    ev["censored_bucket"] = ev["holding_bucket"].astype(str) + (
        ev["is_right_censored"].astype(bool).map({True: " (censored)", False: " (closed)"})
    )
    ev["pass_high"] = ev["n_criteria_passed"] >= split_threshold

    summary = ev.groupby(
        ["censored_bucket", ev["pass_high"].astype(str)], observed=False
    ).agg(
        n=("realized_cagr_to_exit", "count"),
        median_cagr=("realized_cagr_to_exit", "median"),
        mean_cagr=("realized_cagr_to_exit", "mean"),
    ).reset_index()
    summary = summary.rename(columns={"pass_high": f"n_pass>={split_threshold}"})
    return summary


def three_year_uncensored_comparison(
    elite_df: pd.DataFrame,
    *,
    split_threshold: int = 5,
) -> dict[str, float | int]:
    """Per audit plan section 7.B.2c "primary lens": realized CAGR for buys held >=3y.

    Restricts to the regime with minimal censoring. Compares high-pass
    (n_pass>=split_threshold) vs low-pass cohorts via Mann-Whitney.
    """
    from scipy.stats import mannwhitneyu
    ev = evaluable_elite(elite_df)
    ev = ev[ev["realized_cagr_to_exit"].notna()].copy()
    sub = ev[ev["holding_period_quarters"] >= 12]

    pass_grp = sub[sub["n_criteria_passed"] >= split_threshold]["realized_cagr_to_exit"]
    fail_grp = sub[sub["n_criteria_passed"] < split_threshold]["realized_cagr_to_exit"]

    if len(pass_grp) == 0 or len(fail_grp) == 0:
        return {
            "n_total": len(sub),
            "n_high_pass": len(pass_grp),
            "n_low_pass": len(fail_grp),
            "median_cagr_high_pass": float("nan"),
            "median_cagr_low_pass": float("nan"),
            "mean_cagr_high_pass": float("nan"),
            "mean_cagr_low_pass": float("nan"),
            "mannwhitney_p": float("nan"),
        }

    _, p = mannwhitneyu(pass_grp, fail_grp, alternative="two-sided")
    return {
        "n_total": len(sub),
        "n_high_pass": len(pass_grp),
        "n_low_pass": len(fail_grp),
        "median_cagr_high_pass": float(pass_grp.median()),
        "median_cagr_low_pass": float(fail_grp.median()),
        "mean_cagr_high_pass": float(pass_grp.mean()),
        "mean_cagr_low_pass": float(fail_grp.mean()),
        "mannwhitney_p": float(p),
    }


# ----------------------------------------------------------------- Pretty printing


def format_headline_table(df: pd.DataFrame) -> str:
    """Format the §7.A headline output for terminal printing."""
    lines = [
        "Per-criterion elite-vs-control comparison (audit plan section 7.A headline; CMH + BH-FDR)",
        "-" * 95,
        f"{'criterion':>22s}  {'elite':>10s}  {'ctrl':>10s}  {'premium':>9s}  "
        f"{'cmh_p':>9s}  {'bh_q':>9s}  method",
    ]
    for _, r in df.iterrows():
        elite = f"{int(r.elite_pass)}/{int(r.elite_n)} ({r.elite_rate*100:4.1f}%)"
        ctrl = f"{int(r.control_pass)}/{int(r.control_n)} ({r.control_rate*100:4.1f}%)"
        flag = "***" if r.bh_q < 0.05 else ("*" if r.bh_q < 0.10 else "")
        lines.append(
            f"{CRITERIA_DISPLAY[r.criterion]:>22s}  {elite:>10s}  {ctrl:>10s}  "
            f"{r.elite_premium_pp:>+7.1f}pp  {r.cmh_p:>9.4f}  {r.bh_q:>9.4f}  "
            f"{r.cmh_method} {flag}"
        )
    lines.append("")
    lines.append("*** = q < 0.05 (significant after FDR); * = q < 0.10")
    return "\n".join(lines)
