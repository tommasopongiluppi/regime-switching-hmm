"""Analytics on a discrete-time Markov chain transition matrix.

Dependency-light (NumPy only) and self-contained, written to double as a study
reference. Every function takes a row-stochastic transition matrix ``P`` of
shape ``(K, K)`` with ``P[i, j] = P(S_t = j | S_{t-1} = i)`` (rows sum to 1) and
validates that on entry.

Each function's docstring states the formula in plain text and the conditions
under which it applies (irreducible? aperiodic? absorbing states present?).
"""

from __future__ import annotations

from math import gcd

import numpy as np

__all__ = [
    "stationary_distribution",
    "expected_duration",
    "expected_return_time",
    "n_step",
    "second_eigenvalue_modulus",
    "relaxation_time",
    "expected_hitting_times",
    "absorption_probabilities",
    "expected_steps_to_absorption",
    "is_irreducible",
    "is_aperiodic",
    "simulate",
]

_TOL = 1e-9


def _validate(P: np.ndarray) -> np.ndarray:
    """Validate and return a row-stochastic transition matrix as float64.

    Checks that ``P`` is square, non-negative, and has rows summing to 1.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be a square (K, K) matrix, got shape {P.shape}")
    if np.any(P < -_TOL):
        raise ValueError("P has negative entries; not a valid transition matrix")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("P rows must sum to 1 (row-stochastic)")
    return P


def _check_index_set(name: str, idx, k: int) -> np.ndarray:
    """Validate a set of state indices and return it sorted and unique."""
    arr = np.unique(np.asarray(idx, dtype=np.int64))
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.min() < 0 or arr.max() >= k:
        raise ValueError(f"{name} contains out-of-range state indices")
    return arr


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Stationary distribution ``pi`` solving ``pi P = pi`` with ``sum(pi) = 1``.

    Formula: find ``pi`` with ``pi (P - I) = 0`` and ``sum(pi) = 1``. We stack
    ``P.T - I`` on top of an all-ones row and solve the (over-determined) system
    ``A pi = b`` with least squares, where ``b = [0, ..., 0, 1]``. Least squares
    is better conditioned than an eigendecomposition and always returns a real
    vector.

    Applies to: any chain. It is *the* stationary vector and is unique and
    strictly positive when the chain is irreducible; for a reducible chain the
    least-squares solution still satisfies the equations but need not be unique.

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.

    Returns
    -------
    numpy.ndarray
        Stationary distribution, shape ``(K,)``, non-negative and summing to 1.
    """
    P = _validate(P)
    k = P.shape[0]
    A = np.vstack([P.T - np.eye(k), np.ones(k)])
    b = np.concatenate([np.zeros(k), [1.0]])
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.clip(pi, 0.0, None)  # kill tiny negative round-off
    total = pi.sum()
    if total <= 0:
        raise ValueError("degenerate stationary distribution")
    return pi / total


def expected_duration(P: np.ndarray) -> np.ndarray:
    """Expected number of consecutive steps spent in each state.

    Formula: the sojourn time in state ``i`` is geometric with success
    probability ``1 - P[i, i]``, so its mean is ``1 / (1 - P[i, i])``.

    Applies to: any chain. An absorbing state (``P[i, i] = 1``) returns ``inf``.

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.

    Returns
    -------
    numpy.ndarray
        Expected sojourn times, shape ``(K,)``, in time steps.
    """
    P = _validate(P)
    with np.errstate(divide="ignore"):
        return 1.0 / (1.0 - np.diag(P))


def expected_return_time(P: np.ndarray) -> np.ndarray:
    """Mean recurrence time of each state.

    Formula: for a positive-recurrent chain the mean time to return to state
    ``i`` is ``1 / pi[i]`` where ``pi`` is the stationary distribution.

    Applies to: irreducible chains (so that ``pi`` is strictly positive).

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.

    Returns
    -------
    numpy.ndarray
        Mean return times, shape ``(K,)``, in time steps.
    """
    P = _validate(P)
    pi = stationary_distribution(P)
    if np.any(pi <= 0):
        raise ValueError("expected_return_time requires an irreducible chain")
    return 1.0 / pi


def n_step(P: np.ndarray, n: int) -> np.ndarray:
    """``n``-step transition matrix ``P**n``.

    Formula: ``(P**n)[i, j] = P(S_{t+n} = j | S_t = i)``, the ``n``-th matrix
    power of ``P``. ``n = 0`` gives the identity.

    Applies to: any chain.

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.
    n : int
        Number of steps, ``n >= 0``.

    Returns
    -------
    numpy.ndarray
        The matrix power ``P**n``, shape ``(K, K)``.
    """
    P = _validate(P)
    if n < 0:
        raise ValueError("n must be non-negative")
    return np.linalg.matrix_power(P, n)


def second_eigenvalue_modulus(P: np.ndarray) -> float:
    """Modulus of the second-largest eigenvalue, ``|lambda_2|``.

    Formula: order the eigenvalues of ``P`` by decreasing modulus. The largest
    is ``lambda_1 = 1`` (Perron root); the spectral gap ``1 - |lambda_2|``
    controls how fast the chain forgets its initial state.

    Applies to: any chain. Interpreted as a mixing rate for irreducible
    aperiodic chains, where ``|lambda_2| < 1``.

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.

    Returns
    -------
    float
        ``|lambda_2|``. For a 1-state chain (no second eigenvalue) returns 0.
    """
    P = _validate(P)
    if P.shape[0] == 1:
        return 0.0
    moduli = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    return float(moduli[1])


def relaxation_time(P: np.ndarray) -> float:
    """Relaxation time ``1 / (1 - |lambda_2|)``.

    Formula: the reciprocal of the spectral gap; the characteristic number of
    steps over which transient correlations decay.

    Applies to: irreducible aperiodic chains (so ``|lambda_2| < 1``). Returns
    ``inf`` when ``|lambda_2| = 1`` (reducible or periodic chain).

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.

    Returns
    -------
    float
        Relaxation time in time steps.
    """
    P = _validate(P)
    gap = 1.0 - second_eigenvalue_modulus(P)
    if gap <= _TOL:
        return float("inf")
    return 1.0 / gap


def expected_hitting_times(P: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Expected number of steps to first reach a target set.

    Formula (first-step analysis): with ``h_i`` the expected steps to hit the
    target set starting from ``i``, ``h_i = 0`` for ``i`` in ``target`` and
    ``h_i = 1 + sum_j P[i, j] h_j`` otherwise. Restricting to the non-target
    states ``S`` this is the linear system ``(I - Q) h_S = 1``, where ``Q`` is
    ``P`` restricted to ``S x S``.

    Applies to: any chain in which the target set is reachable from every state
    (otherwise some hitting times are infinite and the system is singular).

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.
    target : array_like of int
        Indices of the target states.

    Returns
    -------
    numpy.ndarray
        Expected hitting times, shape ``(K,)``; zero on the target set.
    """
    P = _validate(P)
    k = P.shape[0]
    tgt = _check_index_set("target", target, k)

    mask = np.ones(k, dtype=bool)
    mask[tgt] = False
    S = np.flatnonzero(mask)  # non-target states

    h = np.zeros(k, dtype=np.float64)
    if S.size == 0:
        return h  # every state is a target

    Q = P[np.ix_(S, S)]
    try:
        h_S = np.linalg.solve(np.eye(S.size) - Q, np.ones(S.size))
    except np.linalg.LinAlgError as exc:  # pragma: no cover - singular system
        raise ValueError(
            "target set is not reachable from every state; hitting times diverge"
        ) from exc
    h[S] = h_S
    return h


def _canonical_partition(
    P: np.ndarray, absorbing: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split states into transient/absorbing and return ``(trans, Q, R, N)``."""
    k = P.shape[0]
    # every listed absorbing state must actually be absorbing (P[a, a] == 1)
    if not np.allclose(np.diag(P)[absorbing], 1.0, atol=1e-8):
        raise ValueError("every listed absorbing state must have P[a, a] = 1")

    is_abs = np.zeros(k, dtype=bool)
    is_abs[absorbing] = True
    trans = np.flatnonzero(~is_abs)
    if trans.size == 0:
        raise ValueError("no transient states; nothing to absorb")

    Q = P[np.ix_(trans, trans)]
    R = P[np.ix_(trans, absorbing)]
    try:
        N = np.linalg.inv(np.eye(trans.size) - Q)  # fundamental matrix
    except np.linalg.LinAlgError as exc:  # pragma: no cover - singular system
        raise ValueError(
            "I - Q is singular; an absorbing state is unreachable from some "
            "transient state"
        ) from exc
    return trans, Q, R, N


def absorption_probabilities(P: np.ndarray, absorbing: np.ndarray) -> np.ndarray:
    """Probability of being absorbed in each absorbing state.

    Formula (canonical form): reorder states so ``P = [[Q, R], [0, I]]`` with
    ``Q`` the transient-to-transient block. The fundamental matrix is
    ``N = (I - Q)^-1`` and the absorption probabilities are ``B = N R``, where
    ``B[i, a]`` is the probability of ending in absorbing state ``a`` starting
    from transient state ``i``.

    Applies to: absorbing chains where every listed state is absorbing
    (``P[a, a] = 1``) and every absorbing state is reachable from each transient
    state.

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.
    absorbing : array_like of int
        Indices of the absorbing states.

    Returns
    -------
    numpy.ndarray
        Absorption probabilities, shape ``(K, len(absorbing))``. Rows are
        indexed by all states; an absorbing state's row is one-hot on itself.
        Columns follow the sorted order of ``absorbing``.
    """
    P = _validate(P)
    k = P.shape[0]
    absorbing = _check_index_set("absorbing", absorbing, k)
    trans, _, R, N = _canonical_partition(P, absorbing)

    B = np.zeros((k, absorbing.size), dtype=np.float64)
    B[trans] = N @ R
    # absorbing rows: one-hot on their own column
    for col, a in enumerate(absorbing):
        B[a, col] = 1.0
    return B


def expected_steps_to_absorption(P: np.ndarray, absorbing: np.ndarray) -> np.ndarray:
    """Expected number of steps until absorption.

    Formula: with fundamental matrix ``N = (I - Q)^-1``, the expected steps to
    absorption from each transient state are the row sums ``N @ 1``. Absorbing
    states take 0 steps.

    Applies to: absorbing chains where every listed state is absorbing and is
    reachable from each transient state.

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.
    absorbing : array_like of int
        Indices of the absorbing states.

    Returns
    -------
    numpy.ndarray
        Expected steps to absorption, shape ``(K,)``; zero on absorbing states.
    """
    P = _validate(P)
    k = P.shape[0]
    absorbing = _check_index_set("absorbing", absorbing, k)
    trans, _, _, N = _canonical_partition(P, absorbing)

    t = np.zeros(k, dtype=np.float64)
    t[trans] = N @ np.ones(trans.size)
    return t


def _reachability(P: np.ndarray) -> np.ndarray:
    """Boolean matrix: ``R[i, j]`` is True if ``j`` is reachable from ``i``.

    Uses ``sum_{n=0}^{K-1} P**n > 0``; ``K - 1`` steps suffice to connect any
    pair that communicates in a ``K``-state chain.
    """
    k = P.shape[0]
    acc = np.eye(k)
    power = np.eye(k)
    for _ in range(1, k):
        power = power @ P
        acc = acc + power
    return acc > _TOL


def is_irreducible(P: np.ndarray) -> bool:
    """Whether the chain is irreducible (all states communicate).

    Formula: ``i`` and ``j`` communicate when each is reachable from the other;
    the chain is irreducible when this holds for every pair, i.e. the
    reachability matrix ``sum_{n=0}^{K-1} P**n`` is strictly positive.

    Applies to: any chain.

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.

    Returns
    -------
    bool
        ``True`` if every state can reach every other state.
    """
    P = _validate(P)
    return bool(np.all(_reachability(P)))


def is_aperiodic(P: np.ndarray) -> bool:
    """Whether the chain is aperiodic (period 1).

    Formula: the period of a state ``i`` is ``gcd{ n >= 1 : (P**n)[i, i] > 0 }``;
    the chain is aperiodic when that gcd is 1. Self-return times are scanned up
    to ``2K`` steps, which is sufficient to fix the gcd for a ``K``-state chain.

    Applies to: intended for irreducible chains, where all states share one
    period; the whole-chain period is taken as the gcd over all observed
    self-return times.

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.

    Returns
    -------
    bool
        ``True`` if the chain is aperiodic.
    """
    P = _validate(P)
    k = P.shape[0]
    n_max = 2 * k + 1
    period = 0
    power = np.eye(k)
    for n in range(1, n_max + 1):
        power = power @ P
        # For an irreducible chain all states share one period, so any step n
        # at which some state can return to itself contributes n to the gcd.
        if np.any(np.diag(power) > _TOL):
            period = gcd(period, n)
        if period == 1:
            return True
    return period == 1


def simulate(
    P: np.ndarray,
    startprob: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a state path from the chain.

    Draws ``S_0`` from ``startprob`` and then steps forward, sampling ``S_t``
    from row ``P[S_{t-1}]`` at each step.

    Applies to: any chain.

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.
    startprob : array_like
        Initial distribution, shape ``(K,)``, summing to 1.
    n : int
        Number of states to generate.
    rng : numpy.random.Generator
        Explicit generator; global ``numpy.random`` state is never used.

    Returns
    -------
    numpy.ndarray
        State path, shape ``(n,)``, dtype ``int64``.
    """
    P = _validate(P)
    k = P.shape[0]
    startprob = np.asarray(startprob, dtype=np.float64)
    if startprob.shape != (k,):
        raise ValueError(f"startprob must have shape ({k},)")
    if not np.isclose(startprob.sum(), 1.0, atol=1e-8):
        raise ValueError("startprob must sum to 1")
    if n < 0:
        raise ValueError("n must be non-negative")

    path = np.empty(n, dtype=np.int64)
    if n == 0:
        return path
    path[0] = rng.choice(k, p=startprob)
    for t in range(1, n):
        path[t] = rng.choice(k, p=P[path[t - 1]])
    return path
