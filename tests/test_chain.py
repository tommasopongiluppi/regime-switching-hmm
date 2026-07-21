"""Tests for Markov chain analytics, each against a closed form."""

from __future__ import annotations

import numpy as np

from regime_switching.chain import (
    absorption_probabilities,
    expected_duration,
    expected_hitting_times,
    expected_return_time,
    expected_steps_to_absorption,
    is_aperiodic,
    is_irreducible,
    n_step,
    relaxation_time,
    second_eigenvalue_modulus,
    simulate,
    stationary_distribution,
)


# --------------------------------------------------------------------------- #
# stationary distribution
# --------------------------------------------------------------------------- #
def test_two_state_stationary_closed_form():
    # P(0->1) = p, P(1->0) = q  =>  pi = [q/(p+q), p/(p+q)]
    p, q = 0.1, 0.25
    P = np.array([[1 - p, p], [q, 1 - q]])
    pi = stationary_distribution(P)
    assert np.allclose(pi, [q / (p + q), p / (p + q)])


def test_doubly_stochastic_is_uniform():
    P = np.array([[0.2, 0.3, 0.5], [0.5, 0.2, 0.3], [0.3, 0.5, 0.2]])
    assert np.allclose(P.sum(axis=0), 1.0)  # confirm doubly stochastic
    pi = stationary_distribution(P)
    assert np.allclose(pi, np.full(3, 1 / 3))


def test_return_time_is_reciprocal_of_pi():
    rng = np.random.default_rng(0)
    # random irreducible P: strictly positive rows guarantee irreducibility
    M = rng.uniform(0.1, 1.0, size=(4, 4))
    P = M / M.sum(axis=1, keepdims=True)
    assert is_irreducible(P)
    pi = stationary_distribution(P)
    rt = expected_return_time(P)
    assert np.allclose(pi * rt, 1.0)


# --------------------------------------------------------------------------- #
# durations / spectral
# --------------------------------------------------------------------------- #
def test_expected_duration_known_diagonal():
    P = np.array([[0.8, 0.2, 0.0], [0.0, 0.9, 0.1], [0.05, 0.0, 0.95]])
    assert np.allclose(expected_duration(P), [5.0, 10.0, 20.0])


def test_second_eigenvalue_and_relaxation_two_state():
    # 2-state eigenvalues are 1 and (1 - p - q); relaxation = 1/(p+q).
    p, q = 0.1, 0.25
    P = np.array([[1 - p, p], [q, 1 - q]])
    assert np.isclose(second_eigenvalue_modulus(P), abs(1 - p - q))
    assert np.isclose(relaxation_time(P), 1.0 / (p + q))


# --------------------------------------------------------------------------- #
# gambler's ruin
# --------------------------------------------------------------------------- #
def _gamblers_ruin(n_total: int, p_up: float) -> np.ndarray:
    """Random walk on 0..N with absorbing barriers at 0 and N."""
    k = n_total + 1
    P = np.zeros((k, k))
    P[0, 0] = 1.0
    P[n_total, n_total] = 1.0
    for i in range(1, n_total):
        P[i, i + 1] = p_up
        P[i, i - 1] = 1.0 - p_up
    return P


def test_fair_gamblers_ruin():
    n_total = 10
    P = _gamblers_ruin(n_total, p_up=0.5)
    absorbing = [0, n_total]  # column 1 corresponds to state N (sorted order)
    B = absorption_probabilities(P, absorbing)
    steps = expected_steps_to_absorption(P, absorbing)
    for i in range(n_total + 1):
        # absorption probability at N starting from i is i/N
        assert np.isclose(B[i, 1], i / n_total)
        # expected steps to absorption is i*(N-i)
        assert np.isclose(steps[i], i * (n_total - i))


def test_biased_gamblers_ruin_closed_form():
    n_total = 10
    p_up = 0.6
    q = 1.0 - p_up
    ratio = q / p_up
    P = _gamblers_ruin(n_total, p_up=p_up)
    B = absorption_probabilities(P, [0, n_total])
    for i in range(n_total + 1):
        # P(reach N before 0 | start i) = (1 - r^i) / (1 - r^N),  r = q/p
        expected = (1 - ratio**i) / (1 - ratio**n_total)
        assert np.isclose(B[i, 1], expected)


# --------------------------------------------------------------------------- #
# pattern waiting times
# --------------------------------------------------------------------------- #
def _pattern_chain(transitions: dict[int, tuple[int, int]], absorbing: int):
    """Build P from {state: (next_on_H, next_on_T)} with fair coin flips."""
    k = absorbing + 1
    P = np.zeros((k, k))
    P[absorbing, absorbing] = 1.0
    for s, (on_h, on_t) in transitions.items():
        P[s, on_h] += 0.5
        P[s, on_t] += 0.5
    return P


def test_pattern_waiting_times_hth_and_hht():
    # Progress states track the longest suffix that is a prefix of the pattern.
    # HTH: "" =0, "H" =1, "HT" =2, "HTH" =3 (absorbing).
    hth = _pattern_chain(
        {
            0: (1, 0),  # ""  : H->"H",   T->""
            1: (1, 2),  # "H" : H->"H",   T->"HT"
            2: (3, 0),  # "HT": H->"HTH", T->""
        },
        absorbing=3,
    )
    # HHT: "" =0, "H" =1, "HH" =2, "HHT" =3 (absorbing).
    hht = _pattern_chain(
        {
            0: (1, 0),  # ""  : H->"H",   T->""
            1: (2, 0),  # "H" : H->"HH",  T->""
            2: (2, 3),  # "HH": H->"HH",  T->"HHT"
        },
        absorbing=3,
    )
    steps_hth = expected_steps_to_absorption(hth, [3])[0]
    steps_hht = expected_steps_to_absorption(hht, [3])[0]
    assert np.isclose(steps_hth, 10.0)
    assert np.isclose(steps_hht, 8.0)
    # Asymmetry: HTH overlaps itself -- its suffix "H" is also its prefix, so a
    # failed final flip (T after "HT") drops progress to "" but a spoiled "H"
    # can still seed the next attempt, whereas HHT has no such prefix/suffix
    # overlap. The self-overlap makes HTH strictly slower to appear (10 vs 8).


# --------------------------------------------------------------------------- #
# hitting times, convergence, irreducibility, simulation
# --------------------------------------------------------------------------- #
def test_hitting_times_match_first_step_recursion():
    P = np.array([[0.5, 0.5, 0.0], [0.2, 0.5, 0.3], [0.0, 0.0, 1.0]])
    h = expected_hitting_times(P, target=[2])
    # closed form via the definition h_i = 1 + sum_j P[i,j] h_j (h_2 = 0)
    assert np.isclose(h[2], 0.0)
    assert np.isclose(h[0], 1 + 0.5 * h[0] + 0.5 * h[1])
    assert np.isclose(h[1], 1 + 0.2 * h[0] + 0.5 * h[1])


def test_n_step_converges_to_stationary_rows():
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert is_irreducible(P) and is_aperiodic(P)
    pi = stationary_distribution(P)
    Pn = n_step(P, 1000)
    for row in Pn:
        assert np.allclose(row, pi, atol=1e-10)


def test_n_step_identity_and_composition():
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    assert np.allclose(n_step(P, 0), np.eye(2))
    assert np.allclose(n_step(P, 2), P @ P)


def test_irreducible_and_aperiodic_flags():
    ergodic = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert is_irreducible(ergodic)
    assert is_aperiodic(ergodic)

    # pure 2-cycle: irreducible but periodic (period 2)
    cycle = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert is_irreducible(cycle)
    assert not is_aperiodic(cycle)

    # absorbing state 0 => reducible
    reducible = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert not is_irreducible(reducible)


def test_simulate_empirical_frequencies_match_pi():
    P = np.array([[0.85, 0.15], [0.30, 0.70]])
    pi = stationary_distribution(P)
    rng = np.random.default_rng(1)
    path = simulate(P, startprob=pi, n=200_000, rng=rng)
    freq = np.bincount(path, minlength=2) / path.size
    assert np.allclose(freq, pi, atol=0.01)
    assert path.dtype == np.int64
