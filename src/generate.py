"""
Simulate SME credit and product-engagement data.

Why this file is written the way it is
--------------------------------------
The first version of this project generated credit scores as a pure random walk: every
business started at a random score and moved by a random amount each month, independent of
its sector, its risk category, how much it used the product and whether it ever borrowed.

That makes the analysis meaningless. Any "insight" found in data like that is noise, and it
will be noise again next time the seed changes. So the generator now plants structure on
purpose, and this docstring says exactly what was planted. An analysis that recovers it is
demonstrating method; an analysis that recovers something else is finding an artefact.

What is deliberately built in
-----------------------------
1. Each business has a hidden `health` score in [0, 1], drawn from its sector's base rate
   and nudged by size. Nothing downstream sees `health` directly.
2. Credit score reverts towards a target set by `health`, with month-to-month noise. Real
   credit scores are sticky, so this is AR(1) rather than a walk.
3. Engagement falls with poor health, and falls further in the months before a business
   churns.
4. Funding is applied for more often by businesses that engage, and approved more often at
   higher credit scores. So funding is a *consequence* of score and engagement, not a
   coin flip.
5. Churn is a hazard, not a label: low engagement and a falling score raise the monthly
   probability that a business stops using the platform. Once churned it stays churned.

Deliberate data quality problems
--------------------------------
Real extracts are never clean, and a pipeline that has never met a bad row proves nothing.
The raw files therefore contain, on purpose:

  * duplicated engagement events (a replayed batch)
  * missing `region` on a small share of businesses
  * a handful of credit scores outside the valid 300-850 range
  * one business with two conflicting `created_at` dates
  * timestamps stored in two different string formats

`validate.py` finds these and `analyse.py` handles them. Both report what they did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_BUSINESSES = 500
N_MONTHS = 18
START = pd.Timestamp("2024-08-01")

SECTORS = {
    # sector: (share of businesses, mean health, spread)
    "Retail":        (0.28, 0.44, 0.16),
    "Manufacturing": (0.18, 0.52, 0.15),
    "Services":      (0.24, 0.58, 0.14),
    "Technology":    (0.16, 0.62, 0.18),
    "Healthcare":    (0.14, 0.66, 0.12),
}
REGIONS = ["London", "Midlands", "Scotland", "North West", "South East"]
FEATURES = ["credit_score_report", "funding_recommendation",
            "cash_flow_forecast", "growth_dashboard"]
EVENT_TYPES = ["login", "viewed_report", "downloaded_tool",
               "updated_profile", "requested_funding"]

SCORE_MIN, SCORE_MAX = 300, 850


def _health_to_target_score(health: np.ndarray) -> np.ndarray:
    """Map hidden health onto the credit score range. Deliberately not linear at the ends."""
    return SCORE_MIN + (SCORE_MAX - SCORE_MIN) * (0.15 + 0.75 * health)


def build_profiles(rng: np.random.Generator) -> pd.DataFrame:
    names, shares, means, spreads = zip(
        *[(k, v[0], v[1], v[2]) for k, v in SECTORS.items()])
    sector = rng.choice(names, size=N_BUSINESSES, p=np.array(shares) / sum(shares))

    mean_by_sector = dict(zip(names, means))
    spread_by_sector = dict(zip(names, spreads))
    health = np.array([
        np.clip(rng.normal(mean_by_sector[s], spread_by_sector[s]), 0.02, 0.98)
        for s in sector
    ])

    employees = rng.integers(1, 200, N_BUSINESSES)
    # Bigger firms are slightly healthier, but the effect is small and noisy on purpose.
    health = np.clip(health + 0.0008 * (employees - 100), 0.02, 0.98)

    revenue = np.round(np.exp(rng.normal(12.6, 0.8, N_BUSINESSES)) * (0.6 + health), 2)

    # Risk category is a *view* of health, with disagreement, because in the real world the
    # scorecard and the analyst do not always agree.
    noisy = np.clip(health + rng.normal(0, 0.10, N_BUSINESSES), 0, 1)
    risk = np.where(noisy > 0.62, "Low", np.where(noisy > 0.34, "Medium", "High"))

    created = START - pd.to_timedelta(rng.integers(30, 1100, N_BUSINESSES), unit="D")

    df = pd.DataFrame({
        "business_id": [f"SME_{i + 1:04d}" for i in range(N_BUSINESSES)],
        "sector": sector,
        "region": rng.choice(REGIONS, N_BUSINESSES),
        "employees": employees,
        "annual_revenue_gbp": revenue,
        "default_risk_category": risk,
        "created_at": created,
        "health_hidden": np.round(health, 4),          # hidden, dropped before export
        "opening_credit_score": np.round(
            _health_to_target_score(health) + rng.normal(0, 25, N_BUSINESSES)).clip(
            SCORE_MIN, SCORE_MAX).astype(int),
    })
    return df


def build_monthly(profiles: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Monthly panel: credit score, engagement, funding, and a churn event."""
    rows = []
    months = [START + pd.DateOffset(months=m) for m in range(N_MONTHS)]

    for r in profiles.itertuples():
        health = r.health_hidden
        target = _health_to_target_score(np.array([health]))[0]
        score = float(r.opening_credit_score)
        churned_at = None
        recent_engagement = []

        for m_i, month in enumerate(months):
            if churned_at is not None:
                # After churn: no engagement, no funding, score stops being refreshed.
                rows.append(dict(business_id=r.business_id, month=month,
                                 credit_score=int(round(score)), engagement_events=0,
                                 funding_applied=False, funding_granted_gbp=0.0,
                                 is_active=False))
                continue

            # 1. Credit score reverts towards target. Sticky, so 0.85 on the previous value.
            score = 0.85 * score + 0.15 * target + rng.normal(0, 12)
            score = float(np.clip(score, SCORE_MIN, SCORE_MAX))

            # 2. Engagement depends on health, with a slow decay for weak businesses.
            base = 2 + 16 * health
            decay = 1.0 - (0.02 * m_i if health < 0.45 else 0.0)
            events = int(max(0, rng.poisson(max(0.4, base * decay))))
            recent_engagement.append(events)

            # 3. Funding: engaged businesses apply; higher scores get approved.
            applies = events > 0 and rng.random() < (0.06 + 0.20 * (events / 20))
            granted = 0.0
            if applies:
                p_approve = np.clip((score - 480) / 300, 0.02, 0.95)
                if rng.random() < p_approve:
                    granted = float(np.round(
                        rng.uniform(5_000, 60_000) * (0.5 + health), 2))

            # 4. Churn hazard: quiet businesses with falling scores leave.
            window = recent_engagement[-3:]
            quiet = np.mean(window) < 2.5 if len(window) == 3 else False
            falling = score < target - 40
            hazard = 0.006 + (0.06 if quiet else 0) + (0.03 if falling else 0)
            if m_i >= 3 and rng.random() < hazard:
                churned_at = month

            rows.append(dict(business_id=r.business_id, month=month,
                             credit_score=int(round(score)), engagement_events=events,
                             funding_applied=bool(applies), funding_granted_gbp=granted,
                             is_active=True))

    return pd.DataFrame(rows)


def build_events(monthly: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Explode monthly engagement counts into individual events."""
    active = monthly[monthly.engagement_events > 0]
    out = []
    for r in active.itertuples():
        for _ in range(r.engagement_events):
            day = rng.integers(1, 28)
            ts = r.month + pd.Timedelta(days=int(day), hours=int(rng.integers(6, 22)))
            out.append(dict(
                business_id=r.business_id,
                event_type=rng.choice(EVENT_TYPES),
                feature_name=rng.choice(FEATURES),
                timestamp=ts,
                session_duration_sec=int(rng.integers(30, 1800)),
            ))
    df = pd.DataFrame(out)
    df.insert(0, "event_id", [f"EV_{i:07d}" for i in range(len(df))])
    return df


def _plant_data_quality_problems(profiles, monthly, events, rng):
    """Introduce the specific defects documented at the top of this module."""
    notes = []

    # a) a replayed batch of engagement events
    dupes = events.sample(400, random_state=SEED)
    events = pd.concat([events, dupes], ignore_index=True)
    notes.append(f"duplicated {len(dupes)} engagement events (replayed batch)")

    # b) missing region on a small share of businesses
    idx = profiles.sample(18, random_state=SEED).index
    profiles.loc[idx, "region"] = np.nan
    notes.append(f"blanked region on {len(idx)} businesses")

    # c) credit scores outside the valid range
    bad = monthly.sample(12, random_state=SEED).index
    monthly.loc[bad, "credit_score"] = rng.choice([0, -1, 999, 1200], size=len(bad))
    notes.append(f"corrupted {len(bad)} credit scores outside 300-850")

    # d) one business with a created_at in the future
    profiles.loc[profiles.index[7], "created_at"] = START + pd.Timedelta(days=400)
    notes.append("set one created_at to a date after the observation window opens")

    # e) mixed timestamp formats, which is what actually happens when two systems export
    events["timestamp"] = events["timestamp"].astype(str)
    flip = events.sample(1500, random_state=SEED).index
    events.loc[flip, "timestamp"] = pd.to_datetime(
        events.loc[flip, "timestamp"]).dt.strftime("%d/%m/%Y %H:%M")
    notes.append(f"wrote {len(flip)} timestamps in dd/mm/yyyy instead of ISO")

    return profiles, monthly, events, notes


def main(out_dir: str | Path = "data/raw") -> None:
    rng = np.random.default_rng(SEED)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    profiles = build_profiles(rng)
    monthly = build_monthly(profiles, rng)
    events = build_events(monthly, rng)

    profiles, monthly, events, notes = _plant_data_quality_problems(
        profiles, monthly, events, rng)

    profiles.drop(columns=["health_hidden"]).to_csv(out / "sme_profiles.csv", index=False)
    monthly.to_csv(out / "monthly_credit_scores.csv", index=False)
    events.to_csv(out / "engagement_events.csv", index=False)

    print(f"wrote {len(profiles):,} businesses, {len(monthly):,} monthly rows, "
          f"{len(events):,} events to {out}/")
    print("planted data quality problems:")
    for n in notes:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
