"""
Tests. Not exhaustive, deliberately.

Each of these exists because something could go wrong quietly. A test that only checks
pandas works is noise; these check the three things that would produce a plausible-looking
but wrong report.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import generate
from validate import check_raw, check_analysis_ready, DataQualityError


def test_generation_is_reproducible():
    """Same seed, same data. Without this, every finding is unverifiable."""
    a = generate.build_profiles(np.random.default_rng(generate.SEED))
    b = generate.build_profiles(np.random.default_rng(generate.SEED))
    pd.testing.assert_frame_equal(a, b)


def test_validator_rejects_an_unbalanced_panel():
    """A missing month would silently skew every monthly average."""
    rng = np.random.default_rng(generate.SEED)
    profiles = generate.build_profiles(rng)
    monthly = generate.build_monthly(profiles, rng)
    events = generate.build_events(monthly, rng)
    broken = monthly.drop(monthly.index[0])
    with pytest.raises(DataQualityError, match="unbalanced panel"):
        check_raw(profiles.drop(columns=["health_hidden"]), broken, events)


def test_validator_catches_a_total_broadcast_onto_monthly_rows():
    """The original export joined a per-business engagement total onto every month, so
    Tableau summed it twelve times. This is that bug, caught."""
    df = pd.DataFrame({
        "business_id": ["A"] * 3 + ["B"] * 3,
        "month": pd.to_datetime(["2024-08-01", "2024-09-01", "2024-10-01"] * 2),
        "credit_score": [600] * 6,
        "engagement_events": [100, 100, 100, 80, 80, 80],   # totals, not monthly counts
    })
    with pytest.raises(DataQualityError, match="constant within every business"):
        check_analysis_ready(df)


def test_rolling_engagement_excludes_the_current_month():
    """The feature is meant to be usable as a predictor, so it must not contain its own
    target month. This is the classic leak and it is invisible in a chart."""
    s = pd.Series([1, 2, 3, 4, 5])
    rolled = s.shift(1).rolling(3, min_periods=1).mean()
    assert pd.isna(rolled.iloc[0])
    assert rolled.iloc[3] == pytest.approx((1 + 2 + 3) / 3)


def test_clustered_interval_is_wider_than_the_pooled_one():
    """The whole point of the experiment module. If this ever flips, the analysis is
    claiming more precision from six partners than from six partners' worth of customers,
    which cannot be right."""
    import experiment
    rng = np.random.default_rng(experiment.SEED)
    df = experiment.simulate(rng)
    findings = []
    experiment.analyse(df, findings)
    text = findings[-1]
    import re
    intervals = [float(x) for x in re.findall(r"±([\d.]+)pp", text)]
    assert len(intervals) >= 2
    assert max(intervals) > min(intervals) * 2
