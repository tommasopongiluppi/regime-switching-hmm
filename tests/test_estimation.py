"""Tests for Baum-Welch EM, including a statsmodels cross-check.

``statsmodels`` is used here strictly as an *independent* implementation to
validate our from-scratch results; it is never imported inside the package.
"""

from __future__ import annotations

import numpy as np
import pytest

from regime_switching import GaussianHMM
from regime_switching.estimation import fit_baum_welch, fit_multistart


def test_loglik_monotone_increases(two_state_model, rng):
    x, _ = two_state_model.sample(400, rng)
    res = fit_baum_welch(x, n_states=2, rng=rng, max_iter=100)
    diffs = np.diff(res.history)
    # EM increases the likelihood at every step (allow tiny numerical slack)
    assert np.all(diffs >= -1e-6)


def test_recovers_known_means(two_state_model, rng):
    x, _ = two_state_model.sample(2500, rng)
    res = fit_multistart(x, n_states=2, rng=rng, n_starts=6)
    recovered = np.sort(res.model.means.ravel())
    truth = np.sort(two_state_model.means.ravel())
    assert np.allclose(recovered, truth, atol=5e-3)


def test_transition_persistence_recovered(two_state_model, rng):
    x, _ = two_state_model.sample(3000, rng)
    res = fit_multistart(x, n_states=2, rng=rng, n_starts=6)
    # order states by mean so labels match the ground truth
    order = np.argsort(res.model.means.ravel())
    tm = res.model.transmat[np.ix_(order, order)]
    # both states should be persistent (high self-transition)
    assert tm[0, 0] > 0.8
    assert tm[1, 1] > 0.8


def test_cross_check_loglik_against_statsmodels(two_state_model, rng):
    """Our fitted log-likelihood should be close to statsmodels' Markov switch."""
    sm_api = pytest.importorskip("statsmodels.api")

    x, _ = two_state_model.sample(1200, rng)
    ours = fit_multistart(x, n_states=2, rng=rng, n_starts=6)
    our_loglik = ours.loglik

    # Independent fit with statsmodels' Markov-switching regression (mean +
    # switching variance), used only as an external reference.
    endog = x.ravel()
    sm_mod = sm_api.tsa.MarkovRegression(
        endog, k_regimes=2, trend="c", switching_variance=True
    )
    sm_res = sm_mod.fit(em_iter=50, search_reps=10)

    # Both are Gaussian 2-regime models on the same data; log-likelihoods
    # should be in the same ballpark. Compare per-observation to stay robust.
    n = endog.size
    assert abs(our_loglik - sm_res.llf) / n < 0.05


def test_multistart_beats_or_equals_single(two_state_model, rng):
    x, _ = two_state_model.sample(1000, rng)
    single = fit_baum_welch(x, n_states=2, rng=np.random.default_rng(1))
    multi = fit_multistart(x, n_states=2, rng=np.random.default_rng(2), n_starts=8)
    assert multi.loglik >= single.loglik - 1e-6


def test_fitted_model_is_valid(two_state_model, rng):
    x, _ = two_state_model.sample(800, rng)
    res = fit_multistart(x, n_states=2, rng=rng, n_starts=5)
    m = res.model
    assert isinstance(m, GaussianHMM)
    assert np.allclose(m.transmat.sum(axis=1), 1.0)
    assert np.allclose(m.startprob.sum(), 1.0)
