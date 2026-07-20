"""Log-space inference routines for the Gaussian hidden Markov model.

Every recursion over time is carried out in log space with
:func:`scipy.special.logsumexp`. Raw densities are never multiplied: the naive
forward recursion underflows to exactly zero within a few hundred observations
and — because zero is a perfectly valid float — returns no error when it does.
Working in logs keeps the whole sequence finite.

Notation (``K`` states, ``T`` observations, univariate returns ``r_t``):

- ``log_emission[t, i] = log N(r_t ; mu_i, sigma_i**2)``  shape ``(T, K)``.
- ``log_transmat[i, j] = log P(S_t = j | S_{t-1} = i)``   shape ``(K, K)``.
- ``log_startprob[i]   = log P(S_0 = i)``                 shape ``(K,)``.

Each function vectorises across the ``K`` states within a time step and loops
over time.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "gaussian_log_emission",
    "log_forward",
    "log_backward",
    "log_gamma",
    "log_xi",
    "filtered_log_probs",
    "viterbi",
]


def gaussian_log_emission(
    x: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Log emission likelihoods for univariate Gaussian states.

    Computes ``log_emission[t, i] = log N(x_t ; means[i], variances[i])`` for a
    univariate return series and ``K`` Gaussian states.

    Parameters
    ----------
    x : numpy.ndarray
        Observations, shape ``(T,)``.
    means : numpy.ndarray
        Per-state means ``mu_i``, shape ``(K,)``.
    variances : numpy.ndarray
        Per-state variances ``sigma_i**2`` (not standard deviations), shape
        ``(K,)``. Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        Log-densities, shape ``(T, K)``, entry ``[t, i]`` being
        ``log N(x_t ; means[i], variances[i])``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    means = np.asarray(means, dtype=np.float64).ravel()
    variances = np.asarray(variances, dtype=np.float64).ravel()
    if np.any(variances <= 0):
        raise ValueError("variances must be strictly positive")

    # Broadcast to (T, K); this is a plain log-density, not a recursion.
    diff = x[:, None] - means[None, :]                      # (T, K)
    return -0.5 * (
        np.log(2.0 * np.pi * variances[None, :]) + diff**2 / variances[None, :]
    )


def log_forward(
    log_startprob: np.ndarray,
    log_transmat: np.ndarray,
    log_emission: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Forward recursion in log space.

    Computes ``log_alpha[t, i] = log P(r_0, ..., r_t, S_t = i)`` together with
    the sequence log-likelihood ``loglik = log P(r_0, ..., r_{T-1})``.

    Parameters
    ----------
    log_startprob : numpy.ndarray
        Log initial distribution, shape ``(K,)``.
    log_transmat : numpy.ndarray
        Log transition matrix, shape ``(K, K)``.
    log_emission : numpy.ndarray
        Log emission likelihoods, shape ``(T, K)``.

    Returns
    -------
    log_alpha : numpy.ndarray
        Forward log-probabilities, shape ``(T, K)``.
    loglik : float
        Total log-likelihood ``logsumexp(log_alpha[T-1])``.
    """
    log_startprob = np.asarray(log_startprob, dtype=np.float64)
    log_transmat = np.asarray(log_transmat, dtype=np.float64)
    log_emission = np.asarray(log_emission, dtype=np.float64)

    n_obs, n_states = log_emission.shape
    log_alpha = np.empty((n_obs, n_states), dtype=np.float64)

    log_alpha[0] = log_startprob + log_emission[0]
    for t in range(1, n_obs):
        # For each destination state j, logsumexp over the source state i.
        log_alpha[t] = log_emission[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_transmat, axis=0
        )

    loglik = float(logsumexp(log_alpha[-1]))
    return log_alpha, loglik


def log_backward(
    log_transmat: np.ndarray,
    log_emission: np.ndarray,
) -> np.ndarray:
    """Backward recursion in log space.

    Computes ``log_beta[t, i] = log P(r_{t+1}, ..., r_{T-1} | S_t = i)`` with the
    convention ``log_beta[T-1] = 0``.

    Parameters
    ----------
    log_transmat : numpy.ndarray
        Log transition matrix, shape ``(K, K)``.
    log_emission : numpy.ndarray
        Log emission likelihoods, shape ``(T, K)``.

    Returns
    -------
    numpy.ndarray
        Backward log-probabilities, shape ``(T, K)``.
    """
    log_transmat = np.asarray(log_transmat, dtype=np.float64)
    log_emission = np.asarray(log_emission, dtype=np.float64)

    n_obs, n_states = log_emission.shape
    log_beta = np.empty((n_obs, n_states), dtype=np.float64)

    log_beta[-1] = 0.0
    for t in range(n_obs - 2, -1, -1):
        # For each source state i, logsumexp over the destination state j.
        log_beta[t] = logsumexp(
            log_transmat + (log_emission[t + 1] + log_beta[t + 1])[None, :],
            axis=1,
        )
    return log_beta


def log_gamma(log_alpha: np.ndarray, log_beta: np.ndarray) -> np.ndarray:
    """Smoothed state log-probabilities ``log P(S_t = i | r_0, ..., r_{T-1})``.

    Each row is normalised with :func:`~scipy.special.logsumexp`. This is a
    *smoothed* estimate: it conditions on the **entire** sample, including
    observations after ``t``. It is the right quantity for describing regimes in
    hindsight but is unsafe as a trading signal (see
    :func:`filtered_log_probs`).

    Parameters
    ----------
    log_alpha : numpy.ndarray
        Forward log-probabilities, shape ``(T, K)``.
    log_beta : numpy.ndarray
        Backward log-probabilities, shape ``(T, K)``.

    Returns
    -------
    numpy.ndarray
        Smoothed posterior log-probabilities, shape ``(T, K)``. Each row
        exponentiates to a distribution summing to 1.
    """
    log_ab = np.asarray(log_alpha, dtype=np.float64) + np.asarray(
        log_beta, dtype=np.float64
    )
    return log_ab - logsumexp(log_ab, axis=1, keepdims=True)


def log_xi(
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
    log_transmat: np.ndarray,
    log_emission: np.ndarray,
) -> np.ndarray:
    """Smoothed transition log-probabilities.

    Computes ``log P(S_t = i, S_{t+1} = j | r_0, ..., r_{T-1})`` for
    ``t = 0, ..., T - 2``. Each ``(K, K)`` slice is normalised with
    :func:`~scipy.special.logsumexp` so it exponentiates to a joint distribution
    summing to 1.

    Parameters
    ----------
    log_alpha : numpy.ndarray
        Forward log-probabilities, shape ``(T, K)``.
    log_beta : numpy.ndarray
        Backward log-probabilities, shape ``(T, K)``.
    log_transmat : numpy.ndarray
        Log transition matrix, shape ``(K, K)``.
    log_emission : numpy.ndarray
        Log emission likelihoods, shape ``(T, K)``.

    Returns
    -------
    numpy.ndarray
        Transition posteriors, shape ``(T - 1, K, K)``.
    """
    log_alpha = np.asarray(log_alpha, dtype=np.float64)
    log_beta = np.asarray(log_beta, dtype=np.float64)
    log_transmat = np.asarray(log_transmat, dtype=np.float64)
    log_emission = np.asarray(log_emission, dtype=np.float64)

    # Unnormalised: a[t,i] + A[i,j] + b[t+1,j] + beta[t+1,j], shape (T-1, K, K).
    lxi = (
        log_alpha[:-1, :, None]
        + log_transmat[None, :, :]
        + log_emission[1:, None, :]
        + log_beta[1:, None, :]
    )
    # Normalise each (K, K) slice. The per-slice normaliser equals the sequence
    # log-likelihood at every t, so this is equivalent to subtracting loglik.
    lxi -= logsumexp(lxi, axis=(1, 2), keepdims=True)
    return lxi


def filtered_log_probs(log_alpha: np.ndarray) -> np.ndarray:
    """Filtered state log-probabilities ``log P(S_t = i | r_0, ..., r_t)``.

    Obtained by normalising each row of ``log_alpha`` with
    :func:`~scipy.special.logsumexp`.

    This quantity conditions on information available through time ``t`` **only**
    — it never looks past the current observation — and is therefore the
    **only** state estimate safe to use when generating positions in a backtest.

    By contrast, :func:`log_gamma` (the smoothed posterior) conditions on the
    **entire** sample, so using ``gamma`` to form positions leaks future
    information into past trades and is look-ahead bias. The two coincide only at
    the final observation ``t = T - 1``, where there is no future left to
    condition on.

    Parameters
    ----------
    log_alpha : numpy.ndarray
        Forward log-probabilities, shape ``(T, K)``.

    Returns
    -------
    numpy.ndarray
        Filtered posterior log-probabilities, shape ``(T, K)``. Each row
        exponentiates to a distribution summing to 1.
    """
    log_alpha = np.asarray(log_alpha, dtype=np.float64)
    return log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)


def viterbi(
    log_startprob: np.ndarray,
    log_transmat: np.ndarray,
    log_emission: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Most likely state path via the Viterbi max-plus recursion.

    Runs the max-plus analogue of the forward recursion with backpointers and
    returns the single most probable state sequence.

    This is **not** the same as taking ``argmax`` of :func:`log_gamma` at each
    ``t``. The pointwise-most-likely states maximise each marginal separately and
    can string together transitions of probability zero, yielding a path the
    chain can never actually take; Viterbi maximises the joint probability of the
    whole path and so is always internally consistent.

    Parameters
    ----------
    log_startprob : numpy.ndarray
        Log initial distribution, shape ``(K,)``.
    log_transmat : numpy.ndarray
        Log transition matrix, shape ``(K, K)``.
    log_emission : numpy.ndarray
        Log emission likelihoods, shape ``(T, K)``.

    Returns
    -------
    path : numpy.ndarray
        Most likely state sequence, shape ``(T,)``, dtype ``int64``.
    log_prob : float
        Joint log-probability of that path.
    """
    log_startprob = np.asarray(log_startprob, dtype=np.float64)
    log_transmat = np.asarray(log_transmat, dtype=np.float64)
    log_emission = np.asarray(log_emission, dtype=np.float64)

    n_obs, n_states = log_emission.shape
    log_delta = np.empty((n_obs, n_states), dtype=np.float64)
    backptr = np.zeros((n_obs, n_states), dtype=np.int64)

    log_delta[0] = log_startprob + log_emission[0]
    for t in range(1, n_obs):
        # scores[i, j] = best log-prob of reaching j at t via i at t-1.
        scores = log_delta[t - 1][:, None] + log_transmat
        backptr[t] = np.argmax(scores, axis=0)
        log_delta[t] = log_emission[t] + np.max(scores, axis=0)

    path = np.empty(n_obs, dtype=np.int64)
    path[-1] = int(np.argmax(log_delta[-1]))
    log_prob = float(log_delta[-1, path[-1]])
    for t in range(n_obs - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path, log_prob
