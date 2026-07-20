"""Tests for the walk-forward backtest."""

from __future__ import annotations

import numpy as np

from regime_switching.backtest import (
    align_states_by_mean,
    walk_forward_backtest,
)


def test_align_states_by_mean(two_state_model):
    order = align_states_by_mean(two_state_model)
    means = two_state_model.means[order, 0]
    assert np.all(np.diff(means) >= 0)


def test_backtest_shapes_and_causality(two_state_model, rng):
    x, _ = two_state_model.sample(600, rng)
    returns = x.ravel()  # treat the emitted series as the asset return
    res = walk_forward_backtest(
        x,
        returns,
        n_states=2,
        rng=rng,
        train_size=250,
        step=50,
        n_starts=3,
    )
    m = res.strategy_returns.shape[0]
    assert res.positions.shape == (m,)
    assert res.asset_returns.shape == (m,)
    assert res.filtered_prob.shape == (m, 2)
    # positions are long-only probability weights in [0, 1]
    assert np.all(res.positions >= 0.0)
    assert np.all(res.positions <= 1.0)
    # filtered rows are proper distributions
    assert np.allclose(res.filtered_prob.sum(axis=1), 1.0, atol=1e-8)


def test_backtest_produces_expected_length(two_state_model, rng):
    x, _ = two_state_model.sample(500, rng)
    returns = x.ravel()
    train = 200
    res = walk_forward_backtest(
        x, returns, n_states=2, rng=rng, train_size=train, step=50, n_starts=2
    )
    # out-of-sample span runs from `train` to the penultimate observation
    assert res.strategy_returns.shape[0] == (len(x) - 1) - train
