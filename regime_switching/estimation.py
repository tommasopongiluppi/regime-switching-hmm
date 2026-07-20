"""Baum-Welch expectation-maximisation for the univariate Gaussian HMM.

The M-step re-estimates every parameter in closed form, and each estimate is a
**gamma-weighted version of the ordinary estimator**: the initial distribution
is the state posterior at ``t = 0``; a transition probability is the expected
transition count normalised by the expected time in the source state; a mean is
the responsibility-weighted sample mean; a variance is the
responsibility-weighted sample variance. Replace the responsibilities with hard
0/1 assignments and these collapse to the familiar empirical estimators.

The E-step uses the log-space :func:`~regime_switching.inference.log_gamma` and
:func:`~regime_switching.inference.log_xi`; only the resulting posteriors are
exponentiated, and the closed-form updates operate on those probabilities.

Four safeguards protect the fit; each is flagged at its site in the code:

1. Variance floor — stops a state collapsing onto one observation.
2. Monotonicity assertion — EM must never decrease the log-likelihood.
3. Multi-start — EM only finds a local optimum, so we keep the best of many.
4. Label ordering — the likelihood is label-invariant, so we sort states.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .inference import (
    gaussian_log_emission,
    log_backward,
    log_forward,
    log_gamma,
    log_xi,
)

__all__ = ["EMResult", "fit_em_single", "fit_em"]

logger = logging.getLogger(__name__)


@dataclass
class EMResult:
    """Parameters and diagnostics from an EM fit.

    Attributes
    ----------
    startprob : numpy.ndarray
        Initial state distribution, shape ``(K,)``.
    transmat : numpy.ndarray
        Transition matrix, shape ``(K, K)``, rows summing to 1.
    means : numpy.ndarray
        Per-state emission means, shape ``(K,)``.
    variances : numpy.ndarray
        Per-state emission variances, shape ``(K,)``.
    loglik : float
        Final log-likelihood.
    converged : bool
        Whether the relative log-likelihood change fell below tolerance.
    n_iter : int
        Number of EM iterations performed.
    loglik_trace : list of float
        Log-likelihood at each iteration.
    """

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    loglik: float
    converged: bool
    n_iter: int
    loglik_trace: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# initialisation
# --------------------------------------------------------------------------- #
def _causal_rolling_std(x: np.ndarray, window: int = 21) -> np.ndarray:
    """Trailing rolling standard deviation, shape ``(T,)``.

    Uses only observations up to and including ``t`` (causal). This drives the
    quantile split for the first initialisation; it is not a model feature.
    """
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for t in range(n):
        lo = max(0, t - window + 1)
        seg = x[lo : t + 1]
        out[t] = seg.std() if seg.size > 1 else 0.0
    return out


def _strong_diagonal_transmat(
    n_states: int, rng: np.random.Generator | None
) -> np.ndarray:
    """Transition matrix with a strong diagonal in ``[0.90, 0.97]``.

    Financial regimes are persistent, so seeding EM with sticky states rules out
    a large family of poor local optima (rapidly switching solutions).
    """
    transmat = np.empty((n_states, n_states), dtype=np.float64)
    if n_states == 1:
        return np.ones((1, 1), dtype=np.float64)
    diag = (
        np.full(n_states, 0.95)
        if rng is None
        else rng.uniform(0.90, 0.97, size=n_states)
    )
    for i in range(n_states):
        off = (1.0 - diag[i]) / (n_states - 1)
        transmat[i, :] = off
        transmat[i, i] = diag[i]
    return transmat


def _init_from_quantiles(
    x: np.ndarray, n_states: int, var_floor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """First initialisation: quantile split on a 21-day rolling std.

    Groups observations by quantiles of their trailing volatility and takes each
    group's mean and variance, giving low- and high-volatility seed regimes.
    """
    rstd = _causal_rolling_std(x, 21)
    edges = np.quantile(rstd, np.linspace(0.0, 1.0, n_states + 1))
    means = np.empty(n_states, dtype=np.float64)
    variances = np.empty(n_states, dtype=np.float64)
    for k in range(n_states):
        if k == n_states - 1:
            mask = rstd >= edges[k]
        else:
            mask = (rstd >= edges[k]) & (rstd < edges[k + 1])
        if mask.sum() < 2:
            mask = np.ones(x.shape[0], dtype=bool)
        means[k] = x[mask].mean()
        variances[k] = max(x[mask].var(), var_floor)

    startprob = np.full(n_states, 1.0 / n_states)
    transmat = _strong_diagonal_transmat(n_states, rng=None)
    return startprob, transmat, means, variances


def _init_random(
    x: np.ndarray, n_states: int, var_floor: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random initialisation: means from the sample, variances log-uniform.

    Means are drawn from a Gaussian matched to the sample; variances are drawn
    log-uniformly within a factor of ``e**1.5`` of the sample variance.
    """
    sample_mean = float(x.mean())
    sample_std = float(x.std())
    sample_var = max(float(x.var()), var_floor)

    means = rng.normal(sample_mean, sample_std, size=n_states)
    log_v = rng.uniform(
        np.log(sample_var) - 1.5, np.log(sample_var) + 1.5, size=n_states
    )
    variances = np.maximum(np.exp(log_v), var_floor)

    startprob = rng.dirichlet(np.ones(n_states))
    transmat = _strong_diagonal_transmat(n_states, rng=rng)
    return startprob, transmat, means, variances


# --------------------------------------------------------------------------- #
# E-step / M-step
# --------------------------------------------------------------------------- #
def _e_step(
    x: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Posterior responsibilities and log-likelihood.

    Returns ``(gamma, xi, loglik)`` with ``gamma`` shape ``(T, K)`` and ``xi``
    shape ``(T - 1, K, K)`` (probabilities, already exponentiated).
    """
    log_em = gaussian_log_emission(x, means, variances)
    with np.errstate(divide="ignore"):
        log_sp = np.log(startprob)
        log_tm = np.log(transmat)

    log_alpha, loglik = log_forward(log_sp, log_tm, log_em)
    log_beta = log_backward(log_tm, log_em)
    gamma = np.exp(log_gamma(log_alpha, log_beta))
    xi = np.exp(log_xi(log_alpha, log_beta, log_tm, log_em))
    return gamma, xi, loglik


def _m_step(
    x: np.ndarray,
    gamma: np.ndarray,
    xi: np.ndarray,
    var_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form parameter re-estimation from responsibilities.

    Each update is a gamma-weighted version of the ordinary estimator.
    """
    eps = 1e-300  # guards divisions when a state carries ~zero responsibility

    # startprob_j = gamma[0, j]
    startprob = gamma[0].copy()
    startprob = startprob / startprob.sum()

    # transmat_ij = sum_t xi[t, i, j] / sum_t gamma[t, i]   (t = 0..T-2)
    xi_sum = xi.sum(axis=0)                       # (K, K)
    gamma_from = gamma[:-1].sum(axis=0)           # (K,)
    transmat = xi_sum / (gamma_from[:, None] + eps)
    transmat = transmat / transmat.sum(axis=1, keepdims=True)

    # mean_k = sum_t gamma[t, k] r_t / sum_t gamma[t, k]
    nk = gamma.sum(axis=0)                         # (K,)
    means = (gamma * x[:, None]).sum(axis=0) / (nk + eps)

    # var_k = sum_t gamma[t, k] (r_t - mean_k)^2 / sum_t gamma[t, k]
    diff2 = (x[:, None] - means[None, :]) ** 2
    variances = (gamma * diff2).sum(axis=0) / (nk + eps)

    # SAFEGUARD 1 -- VARIANCE FLOOR. Without it a state can collapse onto a
    # single observation: its variance -> 0 and the Gaussian likelihood at that
    # point -> +inf, so the total log-likelihood diverges to +infinity and the
    # fit is meaningless.
    variances = np.maximum(variances, var_floor)
    return startprob, transmat, means, variances


# --------------------------------------------------------------------------- #
# single run and multi-start
# --------------------------------------------------------------------------- #
def fit_em_single(
    x: np.ndarray,
    n_states: int,
    init: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    n_iter: int = 500,
    tol: float = 1e-8,
    var_floor: float = 1e-10,
) -> EMResult:
    """Run Baum-Welch EM from a single initialisation.

    Parameters
    ----------
    x : numpy.ndarray
        Observations, shape ``(T,)``.
    n_states : int
        Number of states ``K``.
    init : tuple of numpy.ndarray
        ``(startprob, transmat, means, variances)`` initial parameters.
    n_iter : int, optional
        Maximum number of EM iterations.
    tol : float, optional
        Convergence threshold on the relative log-likelihood change.
    var_floor : float, optional
        Lower bound on every state variance.

    Returns
    -------
    EMResult
        Fitted parameters and the log-likelihood trace.
    """
    startprob, transmat, means, variances = (
        np.array(init[0], dtype=np.float64),
        np.array(init[1], dtype=np.float64),
        np.array(init[2], dtype=np.float64),
        np.array(init[3], dtype=np.float64),
    )

    trace: list[float] = []
    prev = -np.inf
    converged = False
    it = 0

    for it in range(1, n_iter + 1):
        gamma, xi, loglik = _e_step(x, startprob, transmat, means, variances)
        trace.append(loglik)

        if it > 1:
            # SAFEGUARD 2 -- MONOTONICITY ASSERTION. EM cannot decrease the
            # log-likelihood; a drop beyond numerical tolerance signals a bug in
            # the E-step -- most often a mis-normalised xi (per-slice normaliser
            # not equal to the sequence likelihood).
            if loglik < prev - 1e-6 * abs(prev):
                raise RuntimeError(
                    "EM log-likelihood decreased from "
                    f"{prev:.10g} to {loglik:.10g} at iteration {it}; "
                    "this violates the EM guarantee and usually means the xi "
                    "normalisation in the E-step is wrong."
                )
            # convergence on relative change
            if abs(loglik - prev) <= tol * abs(prev):
                converged = True
                prev = loglik
                break

        prev = loglik
        startprob, transmat, means, variances = _m_step(x, gamma, xi, var_floor)

    return EMResult(
        startprob=startprob,
        transmat=transmat,
        means=means,
        variances=variances,
        loglik=prev,
        converged=converged,
        n_iter=it,
        loglik_trace=trace,
    )


def _sort_by_variance(res: EMResult) -> EMResult:
    """Permute states into ascending variance order.

    SAFEGUARD 4 -- LABEL ORDERING. The likelihood is invariant to relabelling
    states, so two runs can return the same model under different index
    permutations. Sorting by variance (rows and columns of ``transmat``
    consistently) makes the labelling canonical and comparable across runs.
    """
    order = np.argsort(res.variances)
    return EMResult(
        startprob=res.startprob[order],
        transmat=res.transmat[np.ix_(order, order)],
        means=res.means[order],
        variances=res.variances[order],
        loglik=res.loglik,
        converged=res.converged,
        n_iter=res.n_iter,
        loglik_trace=res.loglik_trace,
    )


def fit_em(
    x: np.ndarray,
    n_states: int,
    *,
    n_iter: int = 500,
    tol: float = 1e-8,
    n_init: int = 20,
    var_floor: float = 1e-10,
    random_state: int | np.random.Generator | None = None,
) -> tuple[EMResult, list[list[float]]]:
    """Multi-start Baum-Welch EM.

    SAFEGUARD 3 -- MULTI-START. EM converges only to a local optimum, so the fit
    is repeated from ``n_init`` initialisations and the run with the highest
    final log-likelihood is kept. The first start uses the volatility-quantile
    seed; the rest are random. Every log-likelihood trace is retained.

    Parameters
    ----------
    x : numpy.ndarray
        Observations, shape ``(T,)``.
    n_states : int
        Number of states ``K``.
    n_iter : int, optional
        Maximum EM iterations per start.
    tol : float, optional
        Relative log-likelihood convergence threshold.
    n_init : int, optional
        Number of initialisations.
    var_floor : float, optional
        Lower bound on every state variance.
    random_state : int, numpy.random.Generator, or None, optional
        Seed or generator controlling the random initialisations.

    Returns
    -------
    best : EMResult
        The highest-likelihood fit, with states sorted by ascending variance.
    all_traces : list of list of float
        The log-likelihood trace of every start, in run order.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    rng = np.random.default_rng(random_state)

    inits: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = [
        _init_from_quantiles(x, n_states, var_floor)
    ]
    for _ in range(max(0, n_init - 1)):
        inits.append(_init_random(x, n_states, var_floor, rng))

    best: EMResult | None = None
    all_traces: list[list[float]] = []
    for s, init in enumerate(inits):
        res = fit_em_single(
            x, n_states, init, n_iter=n_iter, tol=tol, var_floor=var_floor
        )
        all_traces.append(res.loglik_trace)
        logger.debug(
            "start %d: loglik=%.6f converged=%s iters=%d",
            s,
            res.loglik,
            res.converged,
            res.n_iter,
        )
        if best is None or res.loglik > best.loglik:
            best = res

    assert best is not None  # n_init >= 1 always yields at least one run
    best = _sort_by_variance(best)
    logger.info(
        "multi-start best loglik=%.6f over %d starts", best.loglik, len(inits)
    )
    return best, all_traces
