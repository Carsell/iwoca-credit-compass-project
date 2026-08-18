"""
Partner funnel, and a pre-qualification experiment analysed twice.

Why this module exists
----------------------
Businesses reach a lender through partner platforms, and the analytics question is where
the application funnel leaks and what to do about it. The interesting part is not the funnel
chart. It is that the obvious way to analyse the experiment gives the wrong answer with
great confidence.

The experiment simulated here randomises **at partner level**, which is what actually
happens when a feature is switched on per integration rather than per user. Two things then
break if you analyse it as though users were randomised:

1. **Clustering.** Customers within a partner are more like each other than like customers
   elsewhere. Treating 20,000 correlated applications as 20,000 independent observations
   shrinks the confidence interval towards nothing and turns noise into a result.

2. **Imbalance.** With a handful of partners, randomisation will not balance traffic
   quality. The arm that happened to get the better partners looks better whether or not
   the feature does anything.

With the seed used here the pooled analysis reports a large, tightly bounded effect with the
wrong sign, because the arms were not comparable to begin with. The clustered analysis is
honest about not being able to tell. The gap between the two is the finding.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 7
FIG = Path("outputs/figures")

# Partner platforms differ in traffic quality. This is the confounder.
PARTNERS = {
    "AlphaPay":     dict(volume=5200, quality=0.72),
    "BridgeLedger": dict(volume=4100, quality=0.55),
    "CoreBooks":    dict(volume=3400, quality=0.61),
    "DeltaTill":    dict(volume=2600, quality=0.38),
    "EmberCart":    dict(volume=2200, quality=0.45),
    "FluxInvoice":  dict(volume=1500, quality=0.34),
}

# The true effect of showing an indicative offer before the customer applies.
TRUE_LIFT_PP = 2.5


def simulate(rng: np.random.Generator) -> pd.DataFrame:
    """One row per application, with the funnel stages it reached."""
    names = list(PARTNERS)
    # Randomise partners, not customers. Three treated, three control.
    treated = set(rng.choice(names, size=3, replace=False))

    rows = []
    for name in names:
        cfg = PARTNERS[name]
        n = cfg["volume"]
        arm = "prequalification" if name in treated else "control"
        q = cfg["quality"]

        # Funnel: start -> complete data collection -> credit assessed -> offer -> accept.
        # Pre-qualification acts at the point where customers decide to keep going.
        base_complete = 0.30 + 0.45 * q
        complete = np.clip(base_complete + (TRUE_LIFT_PP / 100 if arm ==
                                            "prequalification" else 0), 0, 1)

        started = np.ones(n, dtype=bool)
        completed = rng.random(n) < complete
        assessed = completed & (rng.random(n) < 0.92)
        offered = assessed & (rng.random(n) < (0.35 + 0.5 * q))
        accepted = offered & (rng.random(n) < 0.55)

        rows.append(pd.DataFrame({
            "partner": name, "arm": arm, "quality": q,
            "started": started, "completed": completed,
            "assessed": assessed, "offered": offered, "accepted": accepted,
        }))
    return pd.concat(rows, ignore_index=True)


def funnel_table(df: pd.DataFrame) -> pd.DataFrame:
    stages = ["started", "completed", "assessed", "offered", "accepted"]
    counts = [df[s].sum() for s in stages]
    out = pd.DataFrame({"stage": stages, "customers": counts})
    out["pct_of_start"] = out["customers"] / counts[0] * 100
    out["step_conversion"] = [np.nan] + [
        counts[i] / counts[i - 1] * 100 for i in range(1, len(counts))]
    return out


def _proportion_ci(k, n, z=1.96):
    p = k / n
    se = np.sqrt(p * (1 - p) / n)
    return p, p - z * se, p + z * se


def analyse(df: pd.DataFrame, findings: list[str]) -> None:
    # ---------------------------------------------------------------- funnel
    f = funnel_table(df)
    # Rank steps by customers lost, not by conversion rate. A 45% drop late in the funnel
    # costs fewer customers than a 50% drop at the top, and it is customers that matter.
    f["lost"] = f["customers"].shift(1) - f["customers"]
    worst = f["lost"].idxmax()
    biggest_rate_drop = f.iloc[1:]["step_conversion"].idxmin()
    findings.append(
        f"**The partner funnel loses most of its customers before the credit decision.** Of "
        f"{f.iloc[0]['customers']:,.0f} started applications, {f.iloc[-1]['customers']:,.0f} "
        f"({f.iloc[-1]['pct_of_start']:.1f}%) end in an accepted offer. The costliest step is "
        f"**{f.loc[worst, 'stage']}**, where {f.loc[worst, 'lost']:,.0f} customers are lost, "
        f"more than any other. The steepest percentage drop is elsewhere, at "
        f"**{f.loc[biggest_rate_drop, 'stage']}** "
        f"({f.loc[biggest_rate_drop, 'step_conversion']:.0f}% carry through), which is worth "
        f"separating: a bad rate late in the funnel affects fewer people than a mediocre "
        f"rate at the top. Effort goes where the customers are.")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(f["stage"], f["customers"], color="#4c72b0")
    for i, r in f.iterrows():
        ax.text(i, r["customers"], f"{r['pct_of_start']:.0f}%", ha="center",
                va="bottom", fontsize=8)
    ax.set(ylabel="Customers", title="Partner application funnel")
    fig.tight_layout()
    fig.savefig(FIG / "05_partner_funnel.png")
    plt.close(fig)

    # ------------------------------------------------- the experiment, done badly
    g = df.groupby("arm")["completed"].agg(["sum", "count"])
    p_t, lo_t, hi_t = _proportion_ci(g.loc["prequalification", "sum"],
                                     g.loc["prequalification", "count"])
    p_c, lo_c, hi_c = _proportion_ci(g.loc["control", "sum"], g.loc["control", "count"])
    naive_lift = (p_t - p_c) * 100
    naive_se = np.sqrt((p_t * (1 - p_t) / g.loc["prequalification", "count"]) +
                       (p_c * (1 - p_c) / g.loc["control", "count"]))
    naive_ci = 1.96 * naive_se * 100

    # ------------------------------------------------- the experiment, done properly
    per_partner = (df.groupby(["partner", "arm"])["completed"].mean() * 100).reset_index()
    t = per_partner.loc[per_partner.arm == "prequalification", "completed"]
    c = per_partner.loc[per_partner.arm == "control", "completed"]
    cluster_lift = t.mean() - c.mean()
    # Welch standard error on 3 clusters per arm. Wide, and honestly so.
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    cluster_ci = 2.78 * se          # t critical, ~4 df

    quality_gap = (per_partner.merge(
        pd.DataFrame({"partner": list(PARTNERS),
                      "quality": [v["quality"] for v in PARTNERS.values()]}),
        on="partner").groupby("arm")["quality"].mean())

    findings.append(
        f"**The pre-qualification test looks conclusive and is not.** Pooling every "
        f"application, the treated arm completed at {p_t * 100:.1f}% against "
        f"{p_c * 100:.1f}%, a lift of {naive_lift:+.1f} percentage points with a 95% "
        f"interval of ±{naive_ci:.1f}pp. That is the number a standard A/B calculator "
        f"returns, and it is wrong, because the feature was switched on per partner and not "
        f"per customer. There are six independent units in this experiment, not "
        f"{len(df):,}. Comparing partner-level conversion rates instead gives a lift of "
        f"{cluster_lift:+.1f}pp with an interval of ±{cluster_ci:.1f}pp, which does not "
        f"exclude zero. The draw explains most of it: the treated arm averaged "
        f"{quality_gap['prequalification']:.2f} on partner traffic quality against "
        f"{quality_gap['control']:.2f} for control, so the arms were not comparable before "
        f"the feature was switched on. With six partners, randomisation will not fix that. "
        f"The true effect built into this simulation is +{TRUE_LIFT_PP}pp, so the pooled "
        f"analysis does not merely overstate its precision, it gets the sign wrong and "
        f"reports it with a ±{naive_ci:.1f}pp interval. The honest read is that this design "
        f"cannot resolve an effect of the size we care about. The fix is more partners, "
        f"randomisation within partner, or a difference-in-differences against each "
        f"partner's own pre-period. Not more customers.")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.errorbar([0], [naive_lift], yerr=[naive_ci], fmt="o", capsize=6,
                color="#c44e52", label="Pooled by customer (wrong)")
    ax.errorbar([1], [cluster_lift], yerr=[cluster_ci], fmt="o", capsize=6,
                color="#4c72b0", label="By partner (correct)")
    ax.axhline(0, color="black", lw=1)
    ax.axhline(TRUE_LIFT_PP, color="green", ls=":", lw=1.2)
    ax.text(1.35, TRUE_LIFT_PP, " true effect", color="green", fontsize=8, va="center")
    ax.set(xlim=(-0.6, 1.9), xticks=[0, 1],
           xticklabels=["Pooled\nby customer", "Clustered\nby partner"],
           ylabel="Lift in completion (pp)",
           title="Same experiment, two analyses")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "06_experiment_two_analyses.png")
    plt.close(fig)

    return f


def main(findings: list[str] | None = None):
    FIG.mkdir(parents=True, exist_ok=True)
    Path("data/clean").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = simulate(rng)
    local: list[str] = [] if findings is None else findings
    funnel = analyse(df, local)
    df.to_csv("data/clean/partner_applications.csv", index=False)
    funnel.to_csv("data/clean/funnel_summary.csv", index=False)
    return local


if __name__ == "__main__":
    for x in main():
        print("-", x, "\n")
