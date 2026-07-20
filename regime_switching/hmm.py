"""Gaussian hidden Markov model.

The :class:`GaussianHMM` bundles the model parameters (initial distribution,
transition matrix, per-state Gaussian emissions) and exposes scoring,
filtering, smoothing, and decoding built on the log-space routines in
:mod:`regime_switching.inference`.

Transition matrix convention: ``transmat[i, j] = P(S_t = j | S_{t-1} = i)`` so
rows sum to 1.
"""

from __future__ import annotations

import logging

import numpy as np

from .inference import backward, forward, log_gamma, viterbi

__all__ = ["GaussianHMM", "gaussian_logpdf"]

logger = logging.getLogger(__name__)

_LOG_2PI = float(np.log(2.0 * np.pi))


def gaussian_logpdf(
    x: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
) -> np.ndarray:
    """Log-density of multivariate Gaussians evaluated on a batch.

    Parameters
    ----------
    x : numpy.ndarray
        Observations, shape ``(T, D)``.
    means : numpy.ndarray
        Per-state means, shape ``(K, D)``.
    covars : numpy.ndarray
        Per-state covariance matrices, shape ``(K, D, D)``.

    Returns
    -------
    numpy.ndarray
        Log-densities, shape ``(T, K)``, where entry ``[t, k]`` is
        ``log N(x_t; means[k], covars[k])``.
    """
    x = np.asarray(x, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    covars = np.asarray(covars, dtype=np.float64)

    n_obs, n_dim = x.shape
    n_states = means.shape[0]
    out = np.empty((n_obs, n_states), dtype=np.float64)

    for k in range(n_states):
        cov = covars[k]
        # Cholesky-based log-density for numerical stability.
        chol = np.linalg.cholesky(cov)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        diff = x - means[k]  # (T, D)
        # solve L z = diff^T  ->  Mahalanobis distance = sum z**2
        solved = _solve_lower(chol, diff.T)
        maha = np.sum(solved**2, axis=0)  # (T,)
        out[:, k] = -0.5 * (n_dim * _LOG_2PI + log_det + maha)
    return out


def _solve_lower(chol: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``L z = rhs`` for lower-triangular ``L`` via SciPy if available.

    Falls back to :func:`numpy.linalg.solve` when SciPy is not importable.
    """
    from scipy.linalg import solve_triangular

    return solve_triangular(chol, rhs, lower=True)


class GaussianHMM:
    """Gaussian hidden Markov model with full covariance emissions.

    Parameters
    ----------
    startprob : numpy.ndarray
        Initial state distribution, shape ``(K,)``. Sums to 1.
    transmat : numpy.ndarray
        Transition matrix, shape ``(K, K)``, ``transmat[i, j] =
        P(S_t = j | S_{t-1} = i)``. Rows sum to 1.
    means : numpy.ndarray
        Per-state means, shape ``(K, D)``.
    covars : numpy.ndarray
        Per-state covariance matrices, shape ``(K, D, D)``.

    Attributes
    ----------
    n_states : int
        Number of hidden states ``K``.
    n_dim : int
        Observation dimension ``D``.
    """

    def __init__(
        self,
        startprob: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
    ) -> None:
        self.startprob = np.asarray(startprob, dtype=np.float64)
        self.transmat = np.asarray(transmat, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.covars = np.asarray(covars, dtype=np.float64)
        self._validate()

    # ------------------------------------------------------------------ utils
    def _validate(self) -> None:
        k = self.startprob.shape[0]
        if self.transmat.shape != (k, k):
            raise ValueError(
                f"transmat must be ({k}, {k}), got {self.transmat.shape}"
            )
        if self.means.shape[0] != k:
            raise ValueError("means must have K rows")
        if self.covars.shape[0] != k:
            raise ValueError("covars must have K entries")
        if not np.allclose(self.startprob.sum(), 1.0, atol=1e-6):
            raise ValueError("startprob must sum to 1")
        if not np.allclose(self.transmat.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("transmat rows must sum to 1")

    @property
    def n_states(self) -> int:
        """Number of hidden states ``K``."""
        return self.startprob.shape[0]

    @property
    def n_dim(self) -> int:
        """Observation dimension ``D``."""
        return self.means.shape[1]

    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        """Coerce ``(T,)`` observations to ``(T, 1)``; leave ``(T, D)`` as is."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        return x

    # --------------------------------------------------------------- emissions
    def log_emission(self, x: np.ndarray) -> np.ndarray:
        """Per-state log emission likelihoods.

        Parameters
        ----------
        x : numpy.ndarray
            Observations, shape ``(T,)`` or ``(T, D)``.

        Returns
        -------
        numpy.ndarray
            Log-likelihoods, shape ``(T, K)``.
        """
        return gaussian_logpdf(self._as_2d(x), self.means, self.covars)

    # ----------------------------------------------------------------- scoring
    def score(self, x: np.ndarray) -> float:
        """Total log-likelihood ``log P(x_1, ..., x_T)``.

        Parameters
        ----------
        x : numpy.ndarray
            Observations, shape ``(T,)`` or ``(T, D)``.

        Returns
        -------
        float
            Sequence log-likelihood.
        """
        log_b = self.log_emission(x)
        _, loglik = forward(np.log(self.startprob), np.log(self.transmat), log_b)
        return loglik

    def filter(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Causal filtered state probabilities ``P(S_t | x_1, ..., x_t)``.

        These are **causal**: each row conditions only on observations up to and
        including time ``t``. This is the signal to use in a backtest.

        Parameters
        ----------
        x : numpy.ndarray
            Observations, shape ``(T,)`` or ``(T, D)``.

        Returns
        -------
        filtered : numpy.ndarray
            Filtered probabilities, shape ``(T, K)``. Rows sum to 1.
        loglik : float
            Sequence log-likelihood.
        """
        log_b = self.log_emission(x)
        log_alpha, loglik = forward(
            np.log(self.startprob), np.log(self.transmat), log_b
        )
        # normalise each row: P(S_t | x_1..t) = softmax over states of alpha_t
        log_norm = log_alpha - _logsumexp_rows(log_alpha)
        return np.exp(log_norm), loglik

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Smoothed posterior probabilities ``P(S_t | x_1, ..., x_T)``.

        These are **non-causal** (they use the full sequence, including the
        future). Do not use as a trading signal.

        Parameters
        ----------
        x : numpy.ndarray
            Observations, shape ``(T,)`` or ``(T, D)``.

        Returns
        -------
        numpy.ndarray
            Smoothed posteriors, shape ``(T, K)``. Rows sum to 1.
        """
        log_b = self.log_emission(x)
        log_a, _ = forward(np.log(self.startprob), np.log(self.transmat), log_b)
        log_be = backward(np.log(self.transmat), log_b)
        return np.exp(log_gamma(log_a, log_be))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Most-likely state path via Viterbi decoding.

        Parameters
        ----------
        x : numpy.ndarray
            Observations, shape ``(T,)`` or ``(T, D)``.

        Returns
        -------
        numpy.ndarray
            Integer state sequence, shape ``(T,)``.
        """
        log_b = self.log_emission(x)
        path, _ = viterbi(
            np.log(self.startprob), np.log(self.transmat), log_b
        )
        return path

    # ---------------------------------------------------------------- sampling
    def sample(
        self, n_samples: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw a sample path and observations from the model.

        Parameters
        ----------
        n_samples : int
            Number of time steps ``T`` to generate.
        rng : numpy.random.Generator
            Explicit random generator. Global ``numpy.random`` state is never
            used.

        Returns
        -------
        x : numpy.ndarray
            Observations, shape ``(T, D)``.
        states : numpy.ndarray
            Hidden state sequence, shape ``(T,)``.
        """
        states = np.empty(n_samples, dtype=np.int64)
        x = np.empty((n_samples, self.n_dim), dtype=np.float64)

        states[0] = rng.choice(self.n_states, p=self.startprob)
        for t in range(1, n_samples):
            states[t] = rng.choice(self.n_states, p=self.transmat[states[t - 1]])
        for t in range(n_samples):
            k = states[t]
            x[t] = rng.multivariate_normal(self.means[k], self.covars[k])
        return x, states

    def __repr__(self) -> str:
        return (
            f"GaussianHMM(n_states={self.n_states}, n_dim={self.n_dim})"
        )


def _logsumexp_rows(log_x: np.ndarray) -> np.ndarray:
    """Row-wise ``logsumexp`` keeping dims, shape ``(T, 1)``."""
    from scipy.special import logsumexp

    return logsumexp(log_x, axis=1, keepdims=True)
