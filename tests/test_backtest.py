"""Tests for the walk-forward regime backtest and strategy sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_switching.backtest import (
    benchmarks,
    regime_strategy,
    walk_forward_regimes,
)
from regime_switching.metrics import performance_summary


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _simulate_regimes(T, means, sds, transmat, seed):
    rng = np.random.default_rng(seed)
    means = np.asarray(means)
    sds = np.asarray(sds)
    transmat = np.asarray(transmat)
    states = np.empty(T, dtype=int)
    states[0] = 0
    for t in range(1, T):
        states[t] = rng.choice(len(means), p=transmat[states[t - 1]])
    r = rng.normal(means[states], sds[states])
    return r, states


def _onehot(states, n_states=2):
    probs = np.zeros((len(states), n_states))
    probs[np.arange(len(states)), states] = 1.0
    return probs


# --------------------------------------------------------------------------- #
# oracle test
# --------------------------------------------------------------------------- #
def test_oracle_true_states_beat_buy_and_hold():
    """With the TRUE state sequence the strategy beats buy-and-hold materially;
    with a SHUFFLED state sequence it does not. This shows the machinery can
    detect signal when signal genuinely exists (and not otherwise)."""
    # state 0 is the calm, positive-drift regime; state 1 is the turbulent one
    means = [0.0006, -0.0008]
    sds = [0.006, 0.025]
    transmat = [[0.98, 0.02], [0.04, 0.96]]
    T = 6000
    r, states = _simulate_regimes(T, means, sds, transmat, seed=7)
    dates = pd.bdate_range("2000-01-03", periods=T)
    features = pd.DataFrame({"returns": r}, index=dates)

    bh_sharpe = performance_summary(r)["sharpe"]

    # oracle: handed the true states (one-hot, col 0 = P(calm))
    oracle = regime_strategy(features, _onehot(states), mode="binary", cost_bps=0)
    oracle_sharpe = performance_summary(oracle["net"].to_numpy())["sharpe"]

    # control: handed a shuffled state sequence (destroys the timing)
    shuffled = states.copy()
    np.random.default_rng(0).shuffle(shuffled)
    control = regime_strategy(features, _onehot(shuffled), mode="binary", cost_bps=0)
    control_sharpe = performance_summary(control["net"].to_numpy())["sharpe"]

    # oracle beats buy-and-hold materially...
    assert oracle_sharpe > bh_sharpe + 0.75
    # ...and the shuffled control does not come close to the oracle...
    assert control_sharpe < oracle_sharpe - 0.75
    # ...nor materially beats buy-and-hold.
    assert control_sharpe < bh_sharpe + 0.5


# --------------------------------------------------------------------------- #
# causality guard
# --------------------------------------------------------------------------- #
def _wf_kwargs():
    return dict(n_states=2, train_window=100, refit_every=20, n_init=3, random_state=0)


def test_walk_forward_position_is_lagged_signal():
    r, _ = _simulate_regimes(
        300, [0.0006, -0.0008], [0.006, 0.025], [[0.98, 0.02], [0.04, 0.96]], seed=3
    )
    dates = pd.bdate_range("2004-01-01", periods=300)
    wf = walk_forward_regimes(r, dates, **_wf_kwargs())

    signal = (wf["p_state_0"] >= 0.5).astype(float)
    signal[wf["p_state_0"].isna()] = 0.0
    expected = signal.shift(1).fillna(0.0)
    # position must be EXACTLY the one-day lag of the regime signal
    assert np.array_equal(wf["position"].to_numpy(), expected.to_numpy())


def test_walk_forward_truncation_leaves_earlier_positions_unchanged():
    r, _ = _simulate_regimes(
        300, [0.0006, -0.0008], [0.006, 0.025], [[0.98, 0.02], [0.04, 0.96]], seed=4
    )
    dates = pd.bdate_range("2004-01-01", periods=300)
    full = walk_forward_regimes(r, dates, **_wf_kwargs())

    cut = 240
    trunc = walk_forward_regimes(r[:cut], dates[:cut], **_wf_kwargs())

    common = trunc.index
    for col in ["p_state_0", "p_state_1", "selected_state", "position"]:
        np.testing.assert_array_equal(
            full.loc[common, col].to_numpy(),
            trunc[col].to_numpy(),
            err_msg=f"{col} changed under truncation",
        )


# --------------------------------------------------------------------------- #
# costs
# --------------------------------------------------------------------------- #
def test_costs_zero_equals_gross_and_are_monotonic(rng):
    T = 500
    r = rng.normal(0.0003, 0.01, size=T)
    dates = pd.bdate_range("2010-01-01", periods=T)
    features = pd.DataFrame({"returns": r}, index=dates)
    p_low = rng.uniform(0.0, 1.0, size=T)  # random signal -> nonzero turnover

    zero = regime_strategy(features, p_low, mode="probability", cost_bps=0)
    assert np.allclose(zero["net"].to_numpy(), zero["gross"].to_numpy())

    totals = []
    for c in [0.0, 1.0, 5.0, 20.0, 50.0]:
        s = regime_strategy(features, p_low, mode="probability", cost_bps=c)
        totals.append(s["net"].sum())
    # higher cost strictly reduces cumulative net return
    assert all(totals[i] > totals[i + 1] for i in range(len(totals) - 1))


# --------------------------------------------------------------------------- #
# benchmarks smoke
# --------------------------------------------------------------------------- #
def test_benchmarks_are_causal_and_named(rng):
    T = 400
    r = rng.normal(0.0004, 0.011, size=T)
    dates = pd.bdate_range("2010-01-01", periods=T)
    rv = pd.Series(r, index=dates).rolling(21).std(ddof=1) * np.sqrt(252)
    features = pd.DataFrame({"returns": r, "realised_vol": rv}, index=dates)

    bench = benchmarks(r, features, target_vol=0.10, cost_bps=5)
    assert list(bench.columns) == ["buy_and_hold", "vol_target", "constant_exposure"]
    assert len(bench) == T
    # buy-and-hold net ~ the asset return (minus a one-day entry cost)
    assert np.isclose(bench["buy_and_hold"].iloc[10], r[10], atol=1e-6)
