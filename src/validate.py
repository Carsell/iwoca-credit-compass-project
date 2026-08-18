"""
Data quality checks.

Two kinds of check, and the difference matters.

  * A **warning** is something wrong with the source data that the pipeline is expected to
    handle. Duplicated events and missing regions are warnings: they get reported, counted,
    and cleaned downstream.

  * A **failure** is something that means the analysis would be wrong and nobody would
    notice. An empty table, a business appearing twice in the profile file, a month missing
    from the panel. These raise, because a report built on them is worse than no report.

The point of separating them is that a check which only ever prints is a check nobody reads.
"""
from __future__ import annotations

import pandas as pd

SCORE_MIN, SCORE_MAX = 300, 850


class DataQualityError(Exception):
    pass


def check_raw(profiles: pd.DataFrame, monthly: pd.DataFrame,
              events: pd.DataFrame) -> list[str]:
    """Run every check against the raw extract. Returns the warning list; raises on failure."""
    warnings, failures = [], []

    # ---- failures: the analysis cannot be trusted if any of these are true --------
    for name, df in [("profiles", profiles), ("monthly", monthly), ("events", events)]:
        if df.empty:
            failures.append(f"{name} is empty")

    dup_ids = profiles["business_id"].duplicated().sum()
    if dup_ids:
        failures.append(f"{dup_ids} duplicate business_id in profiles")

    panel = monthly.groupby("business_id")["month"].nunique()
    if panel.nunique() > 1:
        failures.append(
            f"unbalanced panel: businesses have between {panel.min()} and "
            f"{panel.max()} months of history")

    orphans = set(monthly["business_id"]) - set(profiles["business_id"])
    if orphans:
        failures.append(f"{len(orphans)} business_id in monthly with no profile row")

    if failures:
        raise DataQualityError("; ".join(failures))

    # ---- warnings: real defects, handled downstream ------------------------------
    dup_events = events.duplicated(subset=["event_id"]).sum()
    if dup_events:
        warnings.append(f"{dup_events:,} duplicate event_id — deduplicated on load")

    missing_region = profiles["region"].isna().sum()
    if missing_region:
        warnings.append(
            f"{missing_region} businesses missing region — grouped as 'Unknown'")

    out_of_range = (~monthly["credit_score"].between(SCORE_MIN, SCORE_MAX)).sum()
    if out_of_range:
        warnings.append(
            f"{out_of_range} credit scores outside {SCORE_MIN}-{SCORE_MAX} — set to null, "
            f"not clipped, because a clipped 1200 becomes a plausible 850 and stops "
            f"looking like an error")

    first_month = pd.to_datetime(monthly["month"]).min()
    future = (pd.to_datetime(profiles["created_at"]) > first_month).sum()
    if future:
        warnings.append(
            f"{future} businesses created after the first observed month — flagged, "
            f"tenure not computed for them")

    return warnings


def check_analysis_ready(df: pd.DataFrame) -> None:
    """Final gate before anything is exported for reporting."""
    failures = []

    if df["credit_score"].dropna().between(SCORE_MIN, SCORE_MAX).all() is False:
        failures.append("credit_score still out of range after cleaning")

    if df.duplicated(subset=["business_id", "month"]).any():
        failures.append("duplicate business_id/month rows — a BI tool would double-count")

    # engagement_events is a monthly count, so a business-level total must never be
    # broadcast back onto monthly rows. That bug produced a constant 100 in the original
    # export and would have inflated every chart in Tableau.
    per_business = df.groupby("business_id")["engagement_events"].nunique()
    if (per_business == 1).all() and len(df) > len(per_business):
        failures.append(
            "engagement_events is constant within every business — a total has been "
            "joined onto monthly rows")

    if failures:
        raise DataQualityError("; ".join(failures))
