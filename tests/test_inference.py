"""Tests for the log-space inference routines.

The centrepiece is a brute-force check: for a small chain we enumerate every
one of the ``K ** T`` state paths, compute each path's exact joint
log-probability, and confirm by direct summation that the forward
log-likelihood, the smoothed marginals (gamma), the pairwise marginals (xi),
and the Viterbi path all agree with the exact answers.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from regime_switching.inference import (
    filtered_log_probs,
    gaussian_log_emission,
    log_backward,
    log_forward,
    log_gamma,
    log_xi,
    viterbi,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _random_params(rng, n_states, n_obs):
    """Random valid (log_startprob, log_transmat, log_emission)."""
    startprob = rng.dirichlet(np.ones(n_states))
    transmat = np.vstack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)])
    # arbitrary finite log emissions (not necessarily normalised densities)
    log_emission = rng.normal(size=(n_obs, n_states))
    return np.log(startprob), np.log(transmat), log_emission


def _path_log_prob(path, log_startprob, log_transmat, log_emission):
    """Exact joint log-prob of one state path and the observations."""
    lp = log_startprob[path[0]] + log_emission[0, path[0]]
    for t in range(1, len(path)):
        lp += log_transmat[path[t - 1], path[t]] + log_emission[t, path[t]]
    return lp


def _brute_force(log_startprob, log_transmat, log_emission):
    """Enumerate all paths; return exact loglik, gamma, xi, and argmax path."""
    n_obs, n_states = log_emission.shape
    paths = list(itertools.product(range(n_states), repeat=n_obs))
    path_lps = np.array(
        [_path_log_prob(p, log_startprob, log_transmat, log_emission) for p in paths]
    )
    loglik = logsumexp(path_lps)

    # gamma[t, i] = sum over paths with S_t = i
    gamma = np.full((n_obs, n_states), -np.inf)
    for p, lp in zip(paths, path_lps, strict=True):
        for t in range(n_obs):
            gamma[t, p[t]] = np.logaddexp(gamma[t, p[t]], lp)
    gamma = np.exp(gamma - loglik)

    # xi[t, i, j] = sum over paths with S_t = i, S_{t+1} = j
    xi = np.full((n_obs - 1, n_states, n_states), -np.inf)
    for p, lp in zip(paths, path_lps, strict=True):
        for t in range(n_obs - 1):
            xi[t, p[t], p[t + 1]] = np.logaddexp(xi[t, p[t], p[t + 1]], lp)
    xi = np.exp(xi - loglik)

    best_path = np.array(paths[int(np.argmax(path_lps))], dtype=np.int64)
    return loglik, gamma, xi, best_path


# --------------------------------------------------------------------------- #
# the critical brute-force test
# --------------------------------------------------------------------------- #
def test_brute_force_all_quantities():
    rng = np.random.default_rng(0)
    n_states, n_obs = 3, 6
    log_startprob, log_transmat, log_emission = _random_params(rng, n_states, n_obs)

    b_loglik, b_gamma, b_xi, b_path = _brute_force(
        log_startprob, log_transmat, log_emission
    )

    log_alpha, loglik = log_forward(log_startprob, log_transmat, log_emission)
    log_beta = log_backward(log_transmat, log_emission)
    gamma = np.exp(log_gamma(log_alpha, log_beta))
    xi = np.exp(log_xi(log_alpha, log_beta, log_transmat, log_emission))
    path, path_logprob = viterbi(log_startprob, log_transmat, log_emission)

    # (a) forward log-likelihood
    assert np.isclose(loglik, b_loglik, atol=1e-10)
    # (b) smoothed marginals
    assert np.allclose(gamma, b_gamma, atol=1e-10)
    # (c) pairwise marginals
    assert np.allclose(xi, b_xi, atol=1e-10)
    # (d) most likely path
    assert np.array_equal(path, b_path)
    # the reported path log-prob equals the exact joint log-prob of that path
    assert np.isclose(
        path_logprob,
        _path_log_prob(b_path, log_startprob, log_transmat, log_emission),
        atol=1e-10,
    )


# --------------------------------------------------------------------------- #
# forward / backward consistency
# --------------------------------------------------------------------------- #
def test_forward_backward_agreement():
    rng = np.random.default_rng(1)
    log_startprob, log_transmat, log_emission = _random_params(rng, 3, 40)
    log_alpha, loglik = log_forward(log_startprob, log_transmat, log_emission)
    log_beta = log_backward(log_transmat, log_emission)
    # loglik can be recovered from the start of the backward pass too
    recovered = logsumexp(log_startprob + log_emission[0] + log_beta[0])
    assert np.isclose(recovered, loglik, atol=1e-10)
    # ...and from alpha * beta at any t
    for t in [0, 7, 20, 39]:
        assert np.isclose(logsumexp(log_alpha[t] + log_beta[t]), loglik, atol=1e-10)


def test_gamma_and_xi_are_distributions():
    rng = np.random.default_rng(2)
    log_startprob, log_transmat, log_emission = _random_params(rng, 4, 50)
    log_alpha, _ = log_forward(log_startprob, log_transmat, log_emission)
    log_beta = log_backward(log_transmat, log_emission)
    gamma = np.exp(log_gamma(log_alpha, log_beta))
    xi = np.exp(log_xi(log_alpha, log_beta, log_transmat, log_emission))

    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(gamma >= 0)
    assert np.allclose(xi.sum(axis=(1, 2)), 1.0, atol=1e-10)
    # sum_j xi[t, i, j] == gamma[t, i]  for t = 0..T-2
    assert np.allclose(xi.sum(axis=2), gamma[:-1], atol=1e-10)


# --------------------------------------------------------------------------- #
# reductions and edge cases
# --------------------------------------------------------------------------- #
def test_single_state_reduces_to_sum_of_log_densities():
    rng = np.random.default_rng(3)
    n_obs = 25
    log_startprob = np.log([1.0])
    log_transmat = np.log([[1.0]])
    log_emission = rng.normal(size=(n_obs, 1))
    _, loglik = log_forward(log_startprob, log_transmat, log_emission)
    assert np.isclose(loglik, log_emission.sum(), atol=1e-10)

    gamma = np.exp(
        log_gamma(
            log_forward(log_startprob, log_transmat, log_emission)[0],
            log_backward(log_transmat, log_emission),
        )
    )
    assert np.allclose(gamma, 1.0)


def test_no_underflow_on_long_sequence():
    rng = np.random.default_rng(4)
    n_states, n_obs = 3, 10_000
    log_startprob = np.log(rng.dirichlet(np.ones(n_states)))
    log_transmat = np.log(
        np.vstack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)])
    )
    # Genuine Gaussian emission densities: each per-step likelihood is < 1, so
    # the linear-space joint probability is ~exp(loglik) with a hugely negative
    # loglik that would underflow a naive forward pass to exactly 0.0.
    x = rng.normal(size=n_obs)
    means = np.array([-1.0, 0.0, 1.0])
    variances = np.array([0.5, 1.0, 2.0])
    log_emission = gaussian_log_emission(x, means, variances)

    log_alpha, loglik = log_forward(log_startprob, log_transmat, log_emission)
    log_beta = log_backward(log_transmat, log_emission)
    gamma = log_gamma(log_alpha, log_beta)
    xi = log_xi(log_alpha, log_beta, log_transmat, log_emission)

    assert np.isfinite(loglik)
    for arr in (log_alpha, log_beta, gamma, xi):
        assert np.all(np.isfinite(arr))
    # loglik is far below the log of the smallest positive double (~-745), so a
    # naive linear-space forward recursion would have underflowed to -inf here.
    assert loglik < -1000.0
    assert np.exp(loglik) == 0.0  # confirms linear space really would underflow


def test_viterbi_forced_by_permutation_transmat():
    # Deterministic chain: transmat is a permutation (cycle 0 -> 1 -> 2 -> 0).
    n_states, n_obs = 3, 6
    perm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
    # start forced in state 0
    with np.errstate(divide="ignore"):
        log_transmat = np.log(perm)
        log_startprob = np.log(np.array([1.0, 0.0, 0.0]))
    rng = np.random.default_rng(5)
    log_emission = rng.normal(size=(n_obs, n_states))

    path, _ = viterbi(log_startprob, log_transmat, log_emission)
    expected = np.array([(0 + t) % n_states for t in range(n_obs)], dtype=np.int64)
    assert np.array_equal(path, expected)


# --------------------------------------------------------------------------- #
# filtered vs smoothed
# --------------------------------------------------------------------------- #
def test_filtered_matches_smoothed_only_at_final_step():
    rng = np.random.default_rng(6)
    log_startprob, log_transmat, log_emission = _random_params(rng, 3, 30)
    log_alpha, _ = log_forward(log_startprob, log_transmat, log_emission)
    log_beta = log_backward(log_transmat, log_emission)

    filtered = np.exp(filtered_log_probs(log_alpha))
    smoothed = np.exp(log_gamma(log_alpha, log_beta))

    # coincide at the final observation (no future left to condition on)
    assert np.allclose(filtered[-1], smoothed[-1], atol=1e-10)
    # differ earlier: the future genuinely changes the smoothed estimate
    assert not np.allclose(filtered[:-1], smoothed[:-1], atol=1e-6)
    # filtered rows are still proper distributions
    assert np.allclose(filtered.sum(axis=1), 1.0, atol=1e-10)


def test_gaussian_log_emission_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    x = rng.normal(size=15)
    means = np.array([-0.5, 0.0, 1.0])
    variances = np.array([0.25, 1.0, 4.0])
    le = gaussian_log_emission(x, means, variances)
    assert le.shape == (15, 3)
    for i in range(3):
        ref = stats.norm.logpdf(x, loc=means[i], scale=np.sqrt(variances[i]))
        assert np.allclose(le[:, i], ref, atol=1e-12)
