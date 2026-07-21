"""Tests for performance metrics and Sharpe-ratio inference."""

from __future__ import annotations

import numpy as np

from regime_switching.metrics import (
    deflated_sharpe_ratio,
    jobson_korkie_memmel,
    performance_summary,
    sharpe_standard_error,
    turnover,
)


def test_performance_summary_keys(rng):
    r = rng.normal(0.0005, 0.01, size=1000)
    s = performance_summary(r)
    assert set(s) == {
        "cagr",
        "ann_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "skew",
        "excess_kurtosis",
        "n_obs",
    }
    assert s["n_obs"] == 1000
    assert s["ann_vol"] > 0


def test_performance_summary_constant_series_zero_vol_no_error():
    r = np.full(300, 0.001)  # constant return -> zero volatility
    s = performance_summary(r)  # must not raise ZeroDivisionError
    assert np.isclose(s["ann_vol"], 0.0, atol=1e-12)
    assert np.isfinite(s["sharpe"])  # guarded to 0.0, not nan/inf
    assert s["sharpe"] == 0.0
    assert np.isfinite(s["sortino"])


def test_sharpe_standard_error_positive_and_finite(rng):
    r = rng.normal(0.0005, 0.01, size=2000)
    se = sharpe_standard_error(r)
    assert np.isfinite(se) and se > 0


def test_jkm_identical_series_z_near_zero(rng):
    r = rng.normal(0.001, 0.01, size=1000)
    z, p = jobson_korkie_memmel(r, r)
    assert np.isclose(z, 0.0, atol=1e-8)
    assert np.isclose(p, 1.0, atol=1e-8)


def test_jkm_rejects_higher_sharpe(rng):
    n = 4000
    good = rng.normal(0.0015, 0.01, size=n)
    bad = rng.normal(0.0000, 0.01, size=n)
    z, p = jobson_korkie_memmel(good, bad)
    assert z > 0  # good has the higher Sharpe
    assert p < 0.05


def test_turnover_constant_is_zero():
    assert turnover(np.full(100, 0.7)) == 0.0


def test_turnover_known_value():
    # positions 0,1,0,1 -> abs changes 1,1,1 -> mean 1.0
    assert np.isclose(turnover(np.array([0.0, 1.0, 0.0, 1.0])), 1.0)


def test_deflated_sharpe_decreases_with_more_trials():
    # more trials tried => harder to clear the selection-bias hurdle => lower DSR
    dsr_few = deflated_sharpe_ratio(0.12, n_trials=1, T=1000, skew=0.0, kurt=3.0)
    dsr_many = deflated_sharpe_ratio(0.12, n_trials=200, T=1000, skew=0.0, kurt=3.0)
    assert 0.0 <= dsr_many <= dsr_few <= 1.0
