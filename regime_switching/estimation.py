"""Baum-Welch expectation-maximisation for the Gaussian HMM.

Everything here is implemented from scratch on top of the log-space routines in
:mod:`regime_switching.inference`. The E-step accumulates smoothed state
posteriors (``gamma``) and transition posteriors (``xi``); the M-step
re-estimates the initial distribution, transition matrix, and per-state
Gaussian emissions in closed form.

Multi-start fitting runs EM from several random initialisations and keeps the
model with the highest converged log-likelihood, which guards against the many
local optima of the HMM likelihood surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .hmm import GaussianHMM, gaussian_logpdf
from .inference import backward, forward, log_gamma, log_xi

__all__ = ["EMResult", "fit_baum_welch", "fit_multistart"]

logger = logging.getLogger(__name__)


@dataclass
class EMResult:
    """Outcome of a single Baum-Welch run.

    Attributes
    ----------
    model : GaussianHMM
        The fitted model.
    loglik : float
        Final log-likelihood.
    n_iter : int
        Number of EM iterations performed.
    converged : bool
        Whether the log-likelihood change fell below tolerance.
    history : list of float
        Log-likelihood at each iteration.
    """

    model: GaussianHMM
    loglik: float
    n_iter: int
    converged: bool
    history: list[float] = field(default_factory=list)


def _init_params(
    x: np.ndarray, n_states: int, rng: np.random.Generator, reg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random-but-reasonable initial parameters for one EM start.

    Means are drawn from randomly selected observations; covariances start at
    the global covariance; the initial distribution and transition matrix are
    drawn from a Dirichlet with a diagonal bias toward persistence.
    """
    n_obs, n_dim = x.shape

    # startprob and transmat: persistent-ish, from Dirichlet draws. Each
    # transition row gets a diagonal bias so states start out sticky.
    startprob = rng.dirichlet(np.ones(n_states))
    transmat = np.empty((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        alpha = np.ones(n_states)
        alpha[i] += 5.0
        transmat[i] = rng.dirichlet(alpha)

    # means: pick distinct random observations as seeds
    idx = rng.choice(n_obs, size=n_states, replace=n_obs < n_states)
    means = x[idx].copy()

    global_cov = np.cov(x, rowvar=False)
    if global_cov.ndim == 0:  # single feature
        global_cov = global_cov.reshape(1, 1)
    global_cov = global_cov + reg * np.eye(n_dim)
    covars = np.repeat(global_cov[None, :, :], n_states, axis=0)
    return startprob, transmat, means, covars


def fit_baum_welch(
    x: np.ndarray,
    n_states: int,
    rng: np.random.Generator,
    *,
    max_iter: int = 200,
    tol: float = 1e-4,
    reg: float = 1e-6,
    init: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> EMResult:
    """Fit a Gaussian HMM by Baum-Welch EM from a single initialisation.

    Parameters
    ----------
    x : numpy.ndarray
        Observations, shape ``(T,)`` or ``(T, D)``.
    n_states : int
        Number of hidden states ``K``.
    rng : numpy.random.Generator
        Explicit generator for initialisation. Global ``numpy.random`` state is
        never used.
    max_iter : int, optional
        Maximum EM iterations.
    tol : float, optional
        Convergence tolerance on the absolute change in log-likelihood.
    reg : float, optional
        Diagonal regularisation added to each covariance for stability.
    init : tuple, optional
        Explicit ``(startprob, transmat, means, covars)`` initialisation. When
        ``None``, a random initialisation is drawn from ``rng``.

    Returns
    -------
    EMResult
        Fitted model plus convergence diagnostics.
    """
    x = GaussianHMM._as_2d(x)
    n_obs, n_dim = x.shape

    if init is None:
        startprob, transmat, means, covars = _init_params(x, n_states, rng, reg)
    else:
        startprob, transmat, means, covars = (np.array(p, dtype=np.float64) for p in init)

    prev_loglik = -np.inf
    history: list[float] = []
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        # ---- E-step ---------------------------------------------------------
        log_b = gaussian_logpdf(x, means, covars)
        log_a, loglik = forward(np.log(startprob), np.log(transmat), log_b)
        log_be = backward(np.log(transmat), log_b)
        lgamma = log_gamma(log_a, log_be)           # (T, K)
        lxi = log_xi(log_a, log_be, np.log(transmat), log_b, loglik)  # (T-1, K, K)

        gamma = np.exp(lgamma)
        history.append(loglik)

        # ---- convergence check ---------------------------------------------
        if loglik - prev_loglik < tol and it > 1:
            converged = True
            prev_loglik = loglik
            break
        prev_loglik = loglik

        # ---- M-step ---------------------------------------------------------
        startprob = gamma[0] + 1e-12
        startprob /= startprob.sum()

        # transmat[i, j] = sum_t xi[t,i,j] / sum_t gamma[t,i]  (t = 0..T-2)
        xi_sum = np.exp(logsumexp(lxi, axis=0))          # (K, K)
        gamma_sum_from = gamma[:-1].sum(axis=0)          # (K,)
        transmat = xi_sum / (gamma_sum_from[:, None] + 1e-12)
        transmat /= transmat.sum(axis=1, keepdims=True)

        # emissions
        nk = gamma.sum(axis=0)                            # (K,)
        means = (gamma.T @ x) / (nk[:, None] + 1e-12)     # (K, D)
        covars = np.empty((n_states, n_dim, n_dim), dtype=np.float64)
        for k in range(n_states):
            diff = x - means[k]                           # (T, D)
            weighted = diff * gamma[:, k][:, None]        # (T, D)
            cov_k = (weighted.T @ diff) / (nk[k] + 1e-12)
            covars[k] = cov_k + reg * np.eye(n_dim)

    model = GaussianHMM(startprob, transmat, means, covars)
    return EMResult(
        model=model,
        loglik=prev_loglik,
        n_iter=it,
        converged=converged,
        history=history,
    )


def fit_multistart(
    x: np.ndarray,
    n_states: int,
    rng: np.random.Generator,
    *,
    n_starts: int = 10,
    max_iter: int = 200,
    tol: float = 1e-4,
    reg: float = 1e-6,
) -> EMResult:
    """Fit a Gaussian HMM by multi-start Baum-Welch EM.

    Runs :func:`fit_baum_welch` from ``n_starts`` independent random
    initialisations and returns the result with the highest converged
    log-likelihood.

    Parameters
    ----------
    x : numpy.ndarray
        Observations, shape ``(T,)`` or ``(T, D)``.
    n_states : int
        Number of hidden states ``K``.
    rng : numpy.random.Generator
        Explicit generator. Each start consumes fresh draws from it.
    n_starts : int, optional
        Number of random initialisations.
    max_iter : int, optional
        Maximum EM iterations per start.
    tol : float, optional
        Convergence tolerance on the log-likelihood change.
    reg : float, optional
        Diagonal covariance regularisation.

    Returns
    -------
    EMResult
        Best fit across all starts.
    """
    best: EMResult | None = None
    for s in range(n_starts):
        try:
            res = fit_baum_welch(
                x, n_states, rng, max_iter=max_iter, tol=tol, reg=reg
            )
        except np.linalg.LinAlgError:
            logger.warning("EM start %d failed with a linear-algebra error", s)
            continue
        logger.debug(
            "start %d: loglik=%.4f converged=%s iters=%d",
            s,
            res.loglik,
            res.converged,
            res.n_iter,
        )
        if best is None or res.loglik > best.loglik:
            best = res

    if best is None:
        raise RuntimeError("all EM starts failed")
    logger.info("multi-start best loglik=%.4f over %d starts", best.loglik, n_starts)
    return best
