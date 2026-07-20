"""Tests for Markov chain analytics."""

from __future__ import annotations

import numpy as np

from regime_switching.chain import (
    expected_durations,
    is_ergodic,
    is_irreducible,
    mean_first_passage_times,
    n_step_transition,
    stationary_distribution,
)


def test_stationary_is_fixed_point():
    transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = stationary_distribution(transmat)
    assert np.allclose(pi @ transmat, pi)
    assert np.isclose(pi.sum(), 1.0)
    assert np.all(pi >= 0)


def test_stationary_known_value():
    # For a 2-state chain, pi = [b, a] / (a + b) with a = P(0->1), b = P(1->0).
    a, b = 0.1, 0.2
    transmat = np.array([[1 - a, a], [b, 1 - b]])
    pi = stationary_distribution(transmat)
    expected = np.array([b, a]) / (a + b)
    assert np.allclose(pi, expected)


def test_expected_durations():
    transmat = np.array([[0.95, 0.05], [0.10, 0.90]])
    dur = expected_durations(transmat)
    assert np.isclose(dur[0], 1 / 0.05)
    assert np.isclose(dur[1], 1 / 0.10)


def test_n_step_transition_identity_and_compose():
    transmat = np.array([[0.7, 0.3], [0.4, 0.6]])
    assert np.allclose(n_step_transition(transmat, 0), np.eye(2))
    p2 = n_step_transition(transmat, 2)
    assert np.allclose(p2, transmat @ transmat)
    # rows still stochastic
    assert np.allclose(p2.sum(axis=1), 1.0)


def test_mean_first_passage_diagonal_is_recurrence():
    transmat = np.array([[0.6, 0.4], [0.3, 0.7]])
    pi = stationary_distribution(transmat)
    m = mean_first_passage_times(transmat)
    assert np.allclose(np.diag(m), 1.0 / pi)
    assert np.all(m > 0)


def test_irreducible_and_ergodic():
    ergodic = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert is_irreducible(ergodic)
    assert is_ergodic(ergodic)

    # Reducible: state 0 is absorbing.
    reducible = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert not is_irreducible(reducible)
    assert not is_ergodic(reducible)
