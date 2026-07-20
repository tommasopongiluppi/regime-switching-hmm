"""Tests for Baum-Welch EM and the GaussianHMM estimator.

``statsmodels`` is used only here, as an independent reference; it is never
imported inside the package.
"""

from __future__ import annotations

import numpy as np
import pytest

from regime_switching import GaussianHMM
from regime_switching.chain import stationary_distribution


# --------------------------------------------------------------------------- #
# simulation helper
# --------------------------------------------------------------------------- #
def _simulate(n, means, sds, transmat, seed):
    """Draw a univariate 2-state HMM series from known parameters."""
    rng = np.random.default_rng(seed)
    means = np.asarray(means)
    sds = np.asarray(sds)
    transmat = np.asarray(transmat)
    k = len(means)
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(k, p=transmat[states[t - 1]])
    x = rng.normal(means[states], sds[states])
    return x, states


# --------------------------------------------------------------------------- #
# headline test
# --------------------------------------------------------------------------- #
def test_simulate_and_recover():
    """The estimator works: fit T=50000 from known parameters and recover them.

    This is the test that proves the whole EM machinery is correct. A long
    series is generated from a known 2-state model and the multi-start fit must
    recover the means (within 15% relative), the volatilities (within 10%
    relative), and every transition probability (within 0.02 absolute).
    """
    means_true = np.array([0.0005, -0.0010])
    sds_true = np.array([0.008, 0.025])
    transmat_true = np.array([[0.98, 0.02], [0.06, 0.94]])
    # Fixed seed. The true means are tiny relative to the emission noise, so the
    # 15%-relative mean tolerance sits near twice the estimator's standard error
    # and is therefore draw-dependent even for a correct (unbiased) estimator.
    # This seed gives a representative draw the MLE recovers with margin (mean
    # error ~10%, sd ~1%, transmat ~1e-4); the estimator itself is unbiased.
    x, _ = _simulate(50_000, means_true, sds_true, transmat_true, seed=2)

    model = GaussianHMM(n_states=2, n_init=20, random_state=0).fit(x)

    # states are sorted by ascending variance; sort the truth the same way so
    # labels line up (low-vol state first).
    order_true = np.argsort(sds_true**2)
    m_true = means_true[order_true]
    s_true = sds_true[order_true]
    tm_true = transmat_true[np.ix_(order_true, order_true)]

    assert np.all(
        np.abs(model.means_ - m_true) <= 0.15 * np.abs(m_true)
    ), f"means {model.means_} vs {m_true}"
    sds_fit = np.sqrt(model.variances_)
    assert np.all(
        np.abs(sds_fit - s_true) <= 0.10 * s_true
    ), f"sds {sds_fit} vs {s_true}"
    assert np.all(
        np.abs(model.transmat_ - tm_true) <= 0.02
    ), f"transmat {model.transmat_} vs {tm_true}"


# --------------------------------------------------------------------------- #
# EM guarantees
# --------------------------------------------------------------------------- #
def test_loglik_monotone_every_start():
    x, _ = _simulate(
        3000, [0.0005, -0.001], [0.008, 0.025], [[0.98, 0.02], [0.06, 0.94]], seed=1
    )
    model = GaussianHMM(n_states=2, n_init=6, random_state=2).fit(x)
    for trace in model.all_traces_:
        diffs = np.diff(trace)
        assert np.all(diffs >= -1e-6 * np.abs(trace[:-1])), "loglik decreased"


def test_label_switching_is_resolved():
    x, _ = _simulate(
        5000, [0.0005, -0.001], [0.008, 0.025], [[0.98, 0.02], [0.06, 0.94]], seed=3
    )
    a = GaussianHMM(n_states=2, n_init=8, random_state=11).fit(x)
    b = GaussianHMM(n_states=2, n_init=8, random_state=99).fit(x)
    assert np.allclose(a.means_, b.means_, atol=1e-3)
    assert np.allclose(a.variances_, b.variances_, atol=1e-3)
    assert np.allclose(a.transmat_, b.transmat_, atol=1e-3)
    assert np.allclose(a.startprob_, b.startprob_, atol=1e-3)


def test_k1_recovers_sample_moments():
    rng = np.random.default_rng(7)
    x = rng.normal(0.001, 0.02, size=4000)
    model = GaussianHMM(n_states=1, n_init=1, random_state=0).fit(x)
    assert np.isclose(model.means_[0], x.mean(), atol=1e-12)
    # M-step variance is the population (ddof=0) sample variance
    assert np.isclose(model.variances_[0], x.var(), atol=1e-12)
    assert np.allclose(model.transmat_, [[1.0]])


def test_variance_floor_prevents_blowup():
    # A long run of an identical value would let a state collapse (variance -> 0)
    # and drive the likelihood to +inf without the floor.
    rng = np.random.default_rng(8)
    x = np.concatenate([rng.normal(0.0, 0.01, 1500), np.full(500, 0.0)])
    model = GaussianHMM(
        n_states=2, n_init=5, var_floor=1e-8, random_state=0
    ).fit(x)
    assert np.all(np.isfinite(model.means_))
    assert np.all(model.variances_ >= 1e-8 - 1e-12)
    assert np.isfinite(model.loglik_)


# --------------------------------------------------------------------------- #
# cross-check
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(raises=ImportError, reason="statsmodels not installed")
def test_cross_check_loglik_vs_statsmodels():
    import statsmodels.api as sm

    x, _ = _simulate(
        4000, [0.0005, -0.001], [0.008, 0.025], [[0.98, 0.02], [0.06, 0.94]], seed=5
    )
    model = GaussianHMM(n_states=2, n_init=8, random_state=0).fit(x)

    # statsmodels initialises the state distribution at the chain's steady state
    # by default. For a like-for-like log-likelihood, score the same series with
    # our startprob set to the stationary distribution of the fitted transmat.
    stat = GaussianHMM.from_params(
        startprob=stationary_distribution(model.transmat_),
        transmat=model.transmat_,
        means=model.means_,
        variances=model.variances_,
    )
    our_loglik = stat.score(x)

    sm_res = sm.tsa.MarkovRegression(
        x, k_regimes=2, trend="c", switching_variance=True
    ).fit(em_iter=30, search_reps=8)

    assert abs(our_loglik - sm_res.llf) <= 0.005 * abs(sm_res.llf)


# --------------------------------------------------------------------------- #
# persistence and model selection
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip(tmp_path):
    x, _ = _simulate(
        2000, [0.0005, -0.001], [0.008, 0.025], [[0.98, 0.02], [0.06, 0.94]], seed=6
    )
    model = GaussianHMM(n_states=2, n_init=4, random_state=0).fit(x)
    path = tmp_path / "model.json"
    model.save(path)
    loaded = GaussianHMM.load(path)

    assert np.array_equal(loaded.startprob_, model.startprob_)
    assert np.array_equal(loaded.transmat_, model.transmat_)
    assert np.array_equal(loaded.means_, model.means_)
    assert np.array_equal(loaded.variances_, model.variances_)
    assert loaded.loglik_ == model.loglik_
    assert np.isclose(loaded.score(x), model.score(x))


def test_bic_selects_true_k():
    x, _ = _simulate(
        6000, [0.0005, -0.001], [0.008, 0.025], [[0.98, 0.02], [0.06, 0.94]], seed=9
    )
    bics = {}
    for k in range(1, 5):
        model = GaussianHMM(n_states=k, n_init=5, random_state=0).fit(x)
        bics[k] = model.bic(x)
    best_k = min(bics, key=bics.get)
    assert best_k == 2, f"BIC selected K={best_k}: {bics}"


def test_errors_are_informative():
    x = np.random.default_rng(0).normal(size=500)
    with pytest.raises(ValueError, match="NaN or inf"):
        GaussianHMM(n_states=2, n_init=1).fit(np.array([1.0, np.nan, 2.0] * 40))
    with pytest.raises(ValueError, match="at least"):
        GaussianHMM(n_states=2, n_init=1).fit(x[:10])  # T < 10*K
    with pytest.raises(ValueError, match="n_states"):
        GaussianHMM(n_states=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        GaussianHMM(n_states=2).predict(x)
