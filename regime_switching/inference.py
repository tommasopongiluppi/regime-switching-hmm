"""Log-space inference routines for hidden Markov models.

All recursions over time are carried out in log space to avoid numerical
underflow. The public functions here operate on *log* quantities:

- ``log_startprob`` : shape ``(K,)``   log of the initial state distribution.
- ``log_transmat``  : shape ``(K, K)`` log transition matrix, where
  ``transmat[i, j] = P(S_t = j | S_{t-1} = i)`` so rows sum to 1.
- ``log_b``         : shape ``(T, K)`` log emission likelihoods,
  ``log_b[t, k] = log p(x_t | S_t = k)``.

``T`` is the number of time steps and ``K`` the number of hidden states.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "forward",
    "backward",
    "log_gamma",
    "log_xi",
    "viterbi",
]


def forward(
    log_startprob: np.ndarray,
    log_transmat: np.ndarray,
    log_b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run the forward recursion in log space.

    Computes ``log_alpha[t, j] = log P(x_1, ..., x_t, S_t = j)`` and the total
    log-likelihood ``log P(x_1, ..., x_T)``.

    Parameters
    ----------
    log_startprob : numpy.ndarray
        Log initial state distribution, shape ``(K,)``.
    log_transmat : numpy.ndarray
        Log transition matrix, shape ``(K, K)``.
    log_b : numpy.ndarray
        Log emission likelihoods, shape ``(T, K)``.

    Returns
    -------
    log_alpha : numpy.ndarray
        Forward log-probabilities, shape ``(T, K)``.
    loglik : float
        Total log-likelihood ``log P(x_1, ..., x_T)``.
    """
    log_startprob = np.asarray(log_startprob, dtype=np.float64)
    log_transmat = np.asarray(log_transmat, dtype=np.float64)
    log_b = np.asarray(log_b, dtype=np.float64)

    n_obs, n_states = log_b.shape
    log_alpha = np.empty((n_obs, n_states), dtype=np.float64)

    log_alpha[0] = log_startprob + log_b[0]
    # work_buffer[i, j] = log_alpha[t-1, i] + log_transmat[i, j]
    for t in range(1, n_obs):
        # logsumexp over the previous state i (axis 0)
        log_alpha[t] = log_b[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_transmat, axis=0
        )

    loglik = float(logsumexp(log_alpha[-1]))
    return log_alpha, loglik


def backward(
    log_transmat: np.ndarray,
    log_b: np.ndarray,
) -> np.ndarray:
    """Run the backward recursion in log space.

    Computes ``log_beta[t, i] = log P(x_{t+1}, ..., x_T | S_t = i)``.

    Parameters
    ----------
    log_transmat : numpy.ndarray
        Log transition matrix, shape ``(K, K)``.
    log_b : numpy.ndarray
        Log emission likelihoods, shape ``(T, K)``.

    Returns
    -------
    log_beta : numpy.ndarray
        Backward log-probabilities, shape ``(T, K)``.
    """
    log_transmat = np.asarray(log_transmat, dtype=np.float64)
    log_b = np.asarray(log_b, dtype=np.float64)

    n_obs, n_states = log_b.shape
    log_beta = np.empty((n_obs, n_states), dtype=np.float64)

    log_beta[-1] = 0.0
    for t in range(n_obs - 2, -1, -1):
        # logsumexp over the next state j (axis 1)
        log_beta[t] = logsumexp(
            log_transmat + log_b[t + 1] + log_beta[t + 1], axis=1
        )
    return log_beta


def log_gamma(log_alpha: np.ndarray, log_beta: np.ndarray) -> np.ndarray:
    """Posterior state log-probabilities (smoothed).

    Computes ``log P(S_t = j | x_1, ..., x_T)``. This is a *smoothed* estimate:
    it conditions on the entire observation sequence, including the future. Do
    **not** use it as a causal signal in a backtest.

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
    log_ab = log_alpha + log_beta
    return log_ab - logsumexp(log_ab, axis=1, keepdims=True)


def log_xi(
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
    log_transmat: np.ndarray,
    log_b: np.ndarray,
    loglik: float | None = None,
) -> np.ndarray:
    """Posterior transition log-probabilities.

    Computes ``log P(S_t = i, S_{t+1} = j | x_1, ..., x_T)`` for
    ``t = 0, ..., T - 2``.

    Parameters
    ----------
    log_alpha : numpy.ndarray
        Forward log-probabilities, shape ``(T, K)``.
    log_beta : numpy.ndarray
        Backward log-probabilities, shape ``(T, K)``.
    log_transmat : numpy.ndarray
        Log transition matrix, shape ``(K, K)``.
    log_b : numpy.ndarray
        Log emission likelihoods, shape ``(T, K)``.
    loglik : float, optional
        Total log-likelihood. If ``None`` it is recomputed from ``log_alpha``.

    Returns
    -------
    numpy.ndarray
        Transition posteriors, shape ``(T - 1, K, K)``. Each ``(K, K)`` slice
        exponentiates to a joint distribution summing to 1.
    """
    log_alpha = np.asarray(log_alpha, dtype=np.float64)
    log_beta = np.asarray(log_beta, dtype=np.float64)
    log_transmat = np.asarray(log_transmat, dtype=np.float64)
    log_b = np.asarray(log_b, dtype=np.float64)

    if loglik is None:
        loglik = float(logsumexp(log_alpha[-1]))

    n_obs, n_states = log_b.shape
    # log_xi[t, i, j] = a[t,i] + A[i,j] + b[t+1,j] + beta[t+1,j] - loglik
    lxi = (
        log_alpha[:-1, :, None]
        + log_transmat[None, :, :]
        + log_b[1:, None, :]
        + log_beta[1:, None, :]
        - loglik
    )
    return lxi


def viterbi(
    log_startprob: np.ndarray,
    log_transmat: np.ndarray,
    log_b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Most-likely state path via the Viterbi algorithm in log space.

    Parameters
    ----------
    log_startprob : numpy.ndarray
        Log initial state distribution, shape ``(K,)``.
    log_transmat : numpy.ndarray
        Log transition matrix, shape ``(K, K)``.
    log_b : numpy.ndarray
        Log emission likelihoods, shape ``(T, K)``.

    Returns
    -------
    path : numpy.ndarray
        Integer state sequence, shape ``(T,)``, dtype ``int64``.
    logprob : float
        Log-probability of the decoded path.
    """
    log_startprob = np.asarray(log_startprob, dtype=np.float64)
    log_transmat = np.asarray(log_transmat, dtype=np.float64)
    log_b = np.asarray(log_b, dtype=np.float64)

    n_obs, n_states = log_b.shape
    log_delta = np.empty((n_obs, n_states), dtype=np.float64)
    psi = np.zeros((n_obs, n_states), dtype=np.int64)

    log_delta[0] = log_startprob + log_b[0]
    for t in range(1, n_obs):
        # scores[i, j] = log_delta[t-1, i] + log_transmat[i, j]
        scores = log_delta[t - 1][:, None] + log_transmat
        psi[t] = np.argmax(scores, axis=0)
        log_delta[t] = log_b[t] + np.max(scores, axis=0)

    path = np.empty(n_obs, dtype=np.int64)
    path[-1] = int(np.argmax(log_delta[-1]))
    logprob = float(log_delta[-1, path[-1]])
    for t in range(n_obs - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path, logprob
