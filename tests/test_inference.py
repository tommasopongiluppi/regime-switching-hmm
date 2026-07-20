"""Tests for log-space inference routines."""

from __future__ import annotations

import itertools

import numpy as np

from regime_switching.inference import (
    backward,
    forward,
    log_gamma,
    log_xi,
    viterbi,
)


def _brute_force_loglik(startprob, transmat, log_b):
    """Sum over all state paths directly (exponential, tiny T only)."""
    n_obs, n_states = log_b.shape
    total = -np.inf
    for path in itertools.product(range(n_states), repeat=n_obs):
        lp = np.log(startprob[path[0]]) + log_b[0, path[0]]
        for t in range(1, n_obs):
            lp += np.log(transmat[path[t - 1], path[t]]) + log_b[t, path[t]]
        total = np.logaddexp(total, lp)
    return total


def test_forward_matches_brute_force(rng):
    startprob = np.array([0.4, 0.6])
    transmat = np.array([[0.7, 0.3], [0.2, 0.8]])
    log_b = np.log(rng.uniform(0.1, 1.0, size=(6, 2)))

    _, loglik = forward(np.log(startprob), np.log(transmat), log_b)
    brute = _brute_force_loglik(startprob, transmat, log_b)
    assert np.isclose(loglik, brute, atol=1e-10)


def test_gamma_rows_are_distributions(rng):
    startprob = np.array([0.5, 0.5])
    transmat = np.array([[0.6, 0.4], [0.3, 0.7]])
    log_b = np.log(rng.uniform(0.1, 1.0, size=(20, 2)))

    log_a, _ = forward(np.log(startprob), np.log(transmat), log_b)
    log_be = backward(np.log(transmat), log_b)
    gamma = np.exp(log_gamma(log_a, log_be))
    assert np.allclose(gamma.sum(axis=1), 1.0)
    assert np.all(gamma >= 0)


def test_xi_marginalizes_to_gamma(rng):
    startprob = np.array([0.5, 0.5])
    transmat = np.array([[0.6, 0.4], [0.3, 0.7]])
    log_b = np.log(rng.uniform(0.1, 1.0, size=(15, 2)))

    log_a, loglik = forward(np.log(startprob), np.log(transmat), log_b)
    log_be = backward(np.log(transmat), log_b)
    gamma = np.exp(log_gamma(log_a, log_be))
    xi = np.exp(log_xi(log_a, log_be, np.log(transmat), log_b, loglik))

    # sum_j xi[t, i, j] == gamma[t, i]  for t < T-1
    assert np.allclose(xi.sum(axis=2), gamma[:-1], atol=1e-8)
    # each xi slice is a joint distribution
    assert np.allclose(xi.sum(axis=(1, 2)), 1.0, atol=1e-8)


def test_viterbi_single_state_is_trivial():
    log_b = np.log(np.ones((5, 1)))
    path, logprob = viterbi(np.log([1.0]), np.log([[1.0]]), log_b)
    assert np.all(path == 0)
    assert np.isclose(logprob, 0.0)


def test_viterbi_recovers_obvious_path():
    # Emissions strongly favour state 0 then state 1.
    log_b = np.log(
        np.array(
            [
                [0.99, 0.01],
                [0.99, 0.01],
                [0.01, 0.99],
                [0.01, 0.99],
            ]
        )
    )
    startprob = np.array([0.5, 0.5])
    transmat = np.array([[0.5, 0.5], [0.5, 0.5]])
    path, _ = viterbi(np.log(startprob), np.log(transmat), log_b)
    assert list(path) == [0, 0, 1, 1]


def test_forward_score_via_model(two_state_model, rng):
    x, _ = two_state_model.sample(50, rng)
    log_b = two_state_model.log_emission(x)
    _, loglik = forward(
        np.log(two_state_model.startprob),
        np.log(two_state_model.transmat),
        log_b,
    )
    assert np.isfinite(loglik)
    assert np.isclose(loglik, two_state_model.score(x))
