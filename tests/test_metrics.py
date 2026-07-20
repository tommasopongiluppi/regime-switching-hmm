"""Tests for performance metrics and the Sharpe difference test."""

from __future__ import annotations

import numpy as np

from regime_switching.metrics import (
    annualized_volatility,
    jobson_korkie_test,
    max_drawdown,
    sharpe_ratio,
    summary,
)


def test_sharpe_sign_and_scale(rng):
    r = rng.normal(0.001, 0.01, size=2000)
    sr = sharpe_ratio(r, periods_per_year=252)
    # positive drift -> positive Sharpe, roughly mean/std * sqrt(252)
    expected = 0.001 / 0.01 * np.sqrt(252)
    assert sr > 0
    assert abs(sr - expected) < 1.0


def test_zero_vol_sharpe_is_nan():
    r = np.full(100, 0.001)
    assert np.isnan(sharpe_ratio(r))


def test_annualized_vol_scales_with_sqrt_time(rng):
    r = rng.normal(0.0, 0.02, size=5000)
    v = annualized_volatility(r, periods_per_year=252)
    assert np.isclose(v, 0.02 * np.sqrt(252), atol=1e-3)


def test_max_drawdown_known_path():
    # up 10%, down ~18.2% -> peak at 1.1, trough 0.9 => dd = 0.9/1.1 - 1
    r = np.array([0.10, -0.1818181818])
    dd = max_drawdown(r)
    assert np.isclose(dd, 0.9 / 1.1 - 1.0, atol=1e-6)


def test_summary_keys(rng):
    r = rng.normal(0.0005, 0.01, size=500)
    s = summary(r)
    assert set(s) == {
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "hit_rate",
        "n_periods",
    }
    assert s["n_periods"] == 500


def test_jobson_korkie_identical_series_is_zero(rng):
    r = rng.normal(0.001, 0.01, size=1000)
    z, p = jobson_korkie_test(r, r)
    assert np.isclose(z, 0.0, atol=1e-8)
    assert np.isclose(p, 1.0, atol=1e-8)


def test_jobson_korkie_detects_difference(rng):
    n = 4000
    good = rng.normal(0.0015, 0.01, size=n)
    bad = rng.normal(0.0000, 0.01, size=n)
    z, p = jobson_korkie_test(good, bad)
    assert z > 0  # good has the higher Sharpe
    assert p < 0.05
