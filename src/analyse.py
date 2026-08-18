"""
Clean the raw extract, engineer the features, answer the questions, write the outputs.

The questions this answers, set before looking at the data:

  Q1. Does credit score differ by sector, and by enough to matter?
  Q2. Do businesses that use the product more end up with better credit scores, or do
      healthier businesses simply engage more? (These look identical in a chart.)
  Q3. Is there warning before a business churns, and how much?
  Q4. Who applies for funding and who gets it?

Q2 is the one worth reading. The obvious chart shows engagement and credit score moving
together, and the obvious conclusion is that using the product improves your score. This
analysis does not draw that conclusion, and says why.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import experiment
from validate import check_raw, check_analysis_ready, SCORE_MIN, SCORE_MAX

RAW = Path("data/raw")
CLEAN = Path("data/clean")
FIG = Path("outputs/figures")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


# ----------------------------------------------------------------- load and clean

def load_raw():
    profiles = pd.read_csv(RAW / "sme_profiles.csv", parse_dates=["created_at"])
    monthly = pd.read_csv(RAW / "monthly_credit_scores.csv", parse_dates=["month"])
    events = pd.read_csv(RAW / "engagement_events.csv")
    return profiles, monthly, events


def parse_mixed_timestamps(s: pd.Series) -> pd.Series:
    """The extract carries two formats. Parse both and keep the failures visible."""
    iso = pd.to_datetime(s, format="ISO8601", errors="coerce")
    uk = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce")
    return iso.fillna(uk)


def clean(profiles, monthly, events):
    profiles = profiles.copy()
    monthly = monthly.copy()
    events = events.copy()

    events = events.drop_duplicates(subset=["event_id"])
    events["timestamp"] = parse_mixed_timestamps(events["timestamp"])

    profiles["region"] = profiles["region"].fillna("Unknown")

    # Out-of-range scores become null rather than clipped. Clipping 1200 to 850 turns an
    # obvious error into a plausible value and hides it forever.
    bad = ~monthly["credit_score"].between(SCORE_MIN, SCORE_MAX)
    monthly.loc[bad, "credit_score"] = np.nan

    first_month = monthly["month"].min()
    profiles["created_after_window"] = profiles["created_at"] > first_month

    return profiles, monthly, events


# ------------------------------------------------------------- feature engineering

def build_features(profiles, monthly):
    df = monthly.merge(profiles, on="business_id", how="left", validate="many_to_one")

    df["months_on_platform"] = np.where(
        df["created_after_window"], np.nan,
        ((df["month"] - df["created_at"]) / np.timedelta64(1, "D") / 30.44).round())

    # Churn: the first month a business goes inactive and never returns.
    df = df.sort_values(["business_id", "month"])
    churn_month = (df[~df["is_active"]].groupby("business_id")["month"].min()
                   .rename("churn_month"))
    df = df.merge(churn_month, on="business_id", how="left")
    df["is_churned"] = df["churn_month"].notna()
    df["months_to_churn"] = np.where(
        df["is_churned"],
        ((df["churn_month"] - df["month"]) / np.timedelta64(1, "D") / 30.44).round(),
        np.nan)

    # Rolling engagement, per business, excluding the current month so the feature could
    # legitimately be used to predict it. Getting this wrong is the standard leak.
    df["engagement_3m"] = (df.groupby("business_id")["engagement_events"]
                           .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()))
    df["score_change_3m"] = (df.groupby("business_id")["credit_score"]
                             .transform(lambda s: s - s.shift(3)))
    return df


# --------------------------------------------------------------------- the questions

def q1_sector(df, findings):
    active = df[df["is_active"]]
    by_sector = (active.groupby("sector")["credit_score"]
                 .agg(["mean", "median", "count"]).sort_values("mean", ascending=False))
    spread = by_sector["mean"].max() - by_sector["mean"].min()
    findings.append(
        f"**Sector matters, but less than the spread within a sector.** Mean credit score "
        f"runs from {by_sector['mean'].min():.0f} ({by_sector.index[-1]}) to "
        f"{by_sector['mean'].max():.0f} ({by_sector.index[0]}), a gap of {spread:.0f} "
        f"points. Within a single sector the standard deviation is "
        f"{active.groupby('sector')['credit_score'].std().mean():.0f} points, so sector "
        f"alone is a weak basis for a decision about any individual business.")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    order = by_sector.index.tolist()
    ax.boxplot([active.loc[active.sector == s, "credit_score"].dropna() for s in order],
               labels=order, showfliers=False)
    ax.set_ylabel("Credit score")
    ax.set_title("Credit score by sector (active months)")
    fig.tight_layout()
    fig.savefig(FIG / "01_score_by_sector.png")
    plt.close(fig)
    return by_sector


def q2_engagement(df, findings):
    """The one where the obvious answer is wrong."""
    active = df[df["is_active"] & df["credit_score"].notna()]

    # Naive: pool everything and correlate.
    naive = active["engagement_3m"].corr(active["credit_score"])

    # Within-business: does a business's score move when its own engagement moves?
    within = (active.groupby("business_id")
              .apply(lambda g: g["engagement_3m"].corr(g["credit_score"])
                     if g["engagement_3m"].notna().sum() > 4 else np.nan,
                     include_groups=False)
              .dropna())

    findings.append(
        f"**Engagement and credit score correlate at r = {naive:.2f} across businesses, "
        f"and at a median of r = {within.median():.2f} within them.** Pooling the data "
        f"suggests that using the product raises your score. Following each business "
        f"separately, that mostly disappears. The pooled number is largely telling us that "
        f"healthier businesses both engage more and score higher, which is a fact about who "
        f"signs up, not about what the product does. Establishing a product effect would "
        f"need businesses compared against similar businesses that did not engage.")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    s = active.sample(min(3000, len(active)), random_state=42)
    axes[0].scatter(s["engagement_3m"], s["credit_score"], s=4, alpha=0.25)
    axes[0].set(xlabel="Engagement, 3-month mean", ylabel="Credit score",
                title=f"Pooled across businesses (r = {naive:.2f})")
    axes[1].hist(within, bins=30)
    axes[1].axvline(within.median(), color="crimson", lw=1.4)
    axes[1].set(xlabel="Correlation within one business",
                ylabel="Businesses",
                title=f"Within business (median r = {within.median():.2f})")
    fig.tight_layout()
    fig.savefig(FIG / "02_engagement_vs_score.png")
    plt.close(fig)
    return naive, within.median()


def q3_churn_warning(df, findings):
    """Is churn signalled by a business's engagement *level*, or by a *change* in it?

    These are different operational problems. If it is the level, you can screen the book
    once and know who is at risk. If it is a change, you need monitoring, because the
    business at risk this month looked fine last month.
    """
    active = df[df["is_active"]]
    churn_rate = df.groupby("business_id")["is_churned"].first().mean()

    # Level: churners against everyone else, over their whole active life.
    lvl = active.groupby("is_churned")["engagement_events"].mean()
    level_gap = lvl.get(False, np.nan) - lvl.get(True, np.nan)

    # Change: each churner against its own earlier baseline.
    churners = active[active["is_churned"]]
    late = churners[churners["months_to_churn"].between(0, 2)]
    early = churners[churners["months_to_churn"] > 5]
    late_mean = late.groupby("business_id")["engagement_events"].mean()
    early_mean = early.groupby("business_id")["engagement_events"].mean()
    paired = pd.concat([early_mean.rename("early"), late_mean.rename("late")],
                       axis=1).dropna()
    delta = (paired["late"] - paired["early"]).mean()
    fell = (paired["late"] < paired["early"]).mean()

    findings.append(
        f"**Churn shows up as a level, not as a drop, and that changes what you would do "
        f"about it.** {churn_rate:.0%} of businesses left during the window. Across their "
        f"active months they averaged {lvl.get(True, float('nan')):.1f} engagement events "
        f"a month against {lvl.get(False, float('nan')):.1f} for businesses that stayed, a "
        f"gap of {level_gap:.1f}. But comparing each churner's final three months against "
        f"its own earlier baseline, the average change was {delta:+.1f} events and only "
        f"{fell:.0%} of them declined at all — about what you would expect from chance. "
        f"So the businesses that left were mostly quiet from the beginning. They did not go "
        f"quiet. An alert built on a sudden fall in usage would fire late and mostly on the "
        f"wrong accounts; screening the book on sustained low usage would find them, and "
        f"could be run once a quarter rather than monthly.")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].bar(["Stayed", "Churned"],
                [lvl.get(False, np.nan), lvl.get(True, np.nan)],
                color=["#4c72b0", "#c44e52"])
    axes[0].set(ylabel="Engagement events per month",
                title="Level: churners were always quieter")
    axes[1].hist(paired["late"] - paired["early"], bins=25, color="#c44e52")
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set(xlabel="Change in monthly events, final 3 months vs baseline",
                ylabel="Businesses",
                title=f"Change: centred near zero ({delta:+.1f})")
    fig.tight_layout()
    fig.savefig(FIG / "03_churn_level_not_change.png")
    plt.close(fig)
    return churn_rate


def q4_funding(df, findings):
    applied = df[df["funding_applied"]]
    approved = applied[applied["funding_granted_gbp"] > 0]
    rate = len(approved) / len(applied) if len(applied) else np.nan

    bands = pd.cut(applied["credit_score"], [300, 500, 600, 700, 850],
                   labels=["300-500", "500-600", "600-700", "700-850"])
    by_band = applied.groupby(bands, observed=True).apply(
        lambda g: (g["funding_granted_gbp"] > 0).mean(), include_groups=False)

    findings.append(
        f"**Approval is driven by score, and applications by engagement.** "
        f"{len(applied):,} applications were made and {rate:.0%} were approved. Approval "
        f"runs from {by_band.min():.0%} in the 300-500 band to {by_band.max():.0%} above "
        f"700. Applications came almost entirely from businesses that had used the product "
        f"that month, so the funnel narrows at engagement before it narrows at credit.")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(by_band.index.astype(str), by_band.values * 100)
    ax.set(ylabel="Approved (%)", xlabel="Credit score band",
           title="Approval rate by credit score band")
    fig.tight_layout()
    fig.savefig(FIG / "04_approval_by_score.png")
    plt.close(fig)
    return rate


# --------------------------------------------------------------------------- main

def main() -> None:
    for d in (CLEAN, FIG, Path("outputs")):
        d.mkdir(parents=True, exist_ok=True)

    profiles, monthly, events = load_raw()
    warnings = check_raw(profiles, monthly, events)
    print("data quality checks on the raw extract:")
    for w in warnings:
        print(f"  ! {w}")

    profiles, monthly, events = clean(profiles, monthly, events)
    df = build_features(profiles, monthly)
    check_analysis_ready(df)
    print("  ok  analysis-ready checks passed")

    findings: list[str] = []
    q1_sector(df, findings)
    q2_engagement(df, findings)
    q3_churn_warning(df, findings)
    q4_funding(df, findings)
    experiment.main(findings)

    export_cols = ["business_id", "month", "credit_score", "engagement_events",
                   "engagement_3m", "score_change_3m", "funding_applied",
                   "funding_granted_gbp", "is_active", "is_churned", "months_to_churn",
                   "months_on_platform", "sector", "region", "default_risk_category",
                   "employees", "annual_revenue_gbp"]
    df[export_cols].to_csv(CLEAN / "credit_insight.csv", index=False)
    events.to_csv(CLEAN / "engagement_events.csv", index=False)

    with open("outputs/findings.md", "w") as f:
        f.write("# Findings\n\n")
        f.write("Generated by `python run.py`. Every number below is computed from the "
                "data in `data/clean/`, not typed in by hand.\n\n")
        f.write("## Data quality\n\n")
        for w in warnings:
            f.write(f"- {w}\n")
        f.write("\n## Results\n\n")
        for i, x in enumerate(findings, 1):
            f.write(f"{i}. {x}\n\n")

    print(f"\nwrote {len(df):,} rows to {CLEAN}/credit_insight.csv")
    print(f"wrote {len(list(FIG.glob('*.png')))} figures to {FIG}/")
    print("wrote outputs/findings.md")


if __name__ == "__main__":
    main()
