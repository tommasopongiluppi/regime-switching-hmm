"""Markov chain analytics on a fitted transition matrix.

All functions take a row-stochastic transition matrix ``transmat`` of shape
``(K, K)`` with ``transmat[i, j] = P(S_t = j | S_{t-1} = i)``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "stationary_distribution",
    "expected_durations",
    "n_step_transition",
    "mean_first_passage_times",
    "is_irreducible",
    "is_ergodic",
]


def _check_transmat(transmat: np.ndarray) -> np.ndarray:
    """Validate and return a copy of a row-stochastic transition matrix."""
    transmat = np.asarray(transmat, dtype=np.float64)
    if transmat.ndim != 2 or transmat.shape[0] != transmat.shape[1]:
        raise ValueError("transmat must be square, shape (K, K)")
    if not np.allclose(transmat.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("transmat rows must sum to 1")
    return transmat


def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Stationary distribution of an ergodic Markov chain.

    Solves ``pi P = pi`` with ``sum(pi) = 1`` via the left eigenvector of the
    transition matrix associated with eigenvalue 1.

    Parameters
    ----------
    transmat : numpy.ndarray
        Transition matrix, shape ``(K, K)``.

    Returns
    -------
    numpy.ndarray
        Stationary distribution, shape ``(K,)``, non-negative and summing to 1.
    """
    transmat = _check_transmat(transmat)
    eigvals, eigvecs = np.linalg.eig(transmat.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx])
    pi = np.abs(pi)  # eigenvector sign is arbitrary
    total = pi.sum()
    if total == 0:
        raise ValueError("degenerate stationary distribution")
    return pi / total


def expected_durations(transmat: np.ndarray) -> np.ndarray:
    """Expected sojourn time in each state.

    For a state ``i`` with self-transition probability ``p = transmat[i, i]``,
    the number of consecutive steps spent in ``i`` is geometric with mean
    ``1 / (1 - p)``.

    Parameters
    ----------
    transmat : numpy.ndarray
        Transition matrix, shape ``(K, K)``.

    Returns
    -------
    numpy.ndarray
        Expected durations, shape ``(K,)``, in units of time steps. States with
        ``transmat[i, i] == 1`` (absorbing) return ``inf``.
    """
    transmat = _check_transmat(transmat)
    diag = np.diag(transmat)
    with np.errstate(divide="ignore"):
        durations = 1.0 / (1.0 - diag)
    return durations


def n_step_transition(transmat: np.ndarray, n: int) -> np.ndarray:
    """``n``-step transition matrix ``P^n``.

    Parameters
    ----------
    transmat : numpy.ndarray
        Transition matrix, shape ``(K, K)``.
    n : int
        Number of steps, ``n >= 0``. ``n = 0`` returns the identity.

    Returns
    -------
    numpy.ndarray
        The matrix power ``P^n``, shape ``(K, K)``.
    """
    transmat = _check_transmat(transmat)
    if n < 0:
        raise ValueError("n must be non-negative")
    return np.linalg.matrix_power(transmat, n)


def mean_first_passage_times(transmat: np.ndarray) -> np.ndarray:
    """Mean first passage times between states.

    ``M[i, j]`` is the expected number of steps to reach state ``j`` for the
    first time starting from state ``i``. Diagonal entries ``M[i, i]`` give the
    mean recurrence time ``1 / pi[i]``.

    Parameters
    ----------
    transmat : numpy.ndarray
        Transition matrix, shape ``(K, K)``, assumed ergodic.

    Returns
    -------
    numpy.ndarray
        Mean first passage / recurrence times, shape ``(K, K)``.

    Notes
    -----
    Uses the fundamental-matrix formula (Kemeny & Snell). Requires an ergodic
    chain so that the stationary distribution is strictly positive.
    """
    transmat = _check_transmat(transmat)
    k = transmat.shape[0]
    pi = stationary_distribution(transmat)
    if np.any(pi <= 0):
        raise ValueError("chain must be ergodic (strictly positive stationary pi)")

    w = np.tile(pi, (k, 1))                       # rows are pi
    z = np.linalg.inv(np.eye(k) - transmat + w)   # fundamental matrix
    m = np.empty((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            m[i, j] = (z[j, j] - z[i, j]) / pi[j]
    # diagonal: mean recurrence time = 1 / pi
    np.fill_diagonal(m, 1.0 / pi)
    return m


def is_irreducible(transmat: np.ndarray, tol: float = 1e-12) -> bool:
    """Whether the chain is irreducible (all states communicate).

    Parameters
    ----------
    transmat : numpy.ndarray
        Transition matrix, shape ``(K, K)``.
    tol : float, optional
        Threshold below which a reachability probability is treated as zero.

    Returns
    -------
    bool
        ``True`` if every state can reach every other state.
    """
    transmat = _check_transmat(transmat)
    k = transmat.shape[0]
    # reachability = sum_{n=0}^{K-1} P^n > 0
    reach = np.eye(k)
    acc = np.eye(k)
    power = np.eye(k)
    for _ in range(1, k):
        power = power @ transmat
        acc = acc + power
    reach = acc
    return bool(np.all(reach > tol))


def is_ergodic(transmat: np.ndarray, tol: float = 1e-12) -> bool:
    """Whether the chain is ergodic (irreducible and aperiodic).

    A finite irreducible chain is aperiodic when some power ``P^n`` is strictly
    positive. This checks ``P^{(K-1)^2 + 1}`` as a sufficient condition.

    Parameters
    ----------
    transmat : numpy.ndarray
        Transition matrix, shape ``(K, K)``.
    tol : float, optional
        Threshold below which an entry is treated as zero.

    Returns
    -------
    bool
        ``True`` if the chain is ergodic.
    """
    transmat = _check_transmat(transmat)
    if not is_irreducible(transmat, tol):
        return False
    k = transmat.shape[0]
    exponent = (k - 1) ** 2 + 1
    powered = np.linalg.matrix_power(transmat, exponent)
    return bool(np.all(powered > tol))
