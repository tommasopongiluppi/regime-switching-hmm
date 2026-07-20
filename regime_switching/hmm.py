"""Univariate Gaussian hidden Markov model with an sklearn-style API.

:class:`GaussianHMM` fits a ``K``-state Gaussian HMM to a one-dimensional return
series by multi-start Baum-Welch EM (see :mod:`regime_switching.estimation`) and
exposes scoring, decoding, filtering, smoothing, sampling, model selection
(AIC/BIC), JSON persistence, and a readable ``summary``.

Transition matrix convention: ``transmat_[i, j] = P(S_t = j | S_{t-1} = i)`` so
rows sum to 1.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .chain import expected_durations
from .estimation import fit_em
from .inference import (
    filtered_log_probs,
    gaussian_log_emission,
    log_backward,
    log_forward,
    log_gamma,
    viterbi,
)

__all__ = ["GaussianHMM"]

logger = logging.getLogger(__name__)

_FITTED_ATTRS = (
    "startprob_",
    "transmat_",
    "means_",
    "variances_",
    "loglik_",
    "converged_",
    "n_iter_",
    "loglik_trace_",
    "all_traces_",
)


class GaussianHMM:
    """Multi-start Baum-Welch Gaussian HMM for univariate returns.

    Parameters
    ----------
    n_states : int, optional
        Number of hidden states ``K``. Must be ``>= 1``.
    n_iter : int, optional
        Maximum EM iterations per initialisation.
    tol : float, optional
        Convergence threshold on the relative log-likelihood change.
    n_init : int, optional
        Number of random initialisations (multi-start).
    var_floor : float, optional
        Lower bound on every state variance.
    random_state : int, numpy.random.Generator, or None, optional
        Seed or generator controlling initialisation.

    Attributes
    ----------
    startprob_ : numpy.ndarray
        Fitted initial distribution, shape ``(K,)``.
    transmat_ : numpy.ndarray
        Fitted transition matrix, shape ``(K, K)``.
    means_ : numpy.ndarray
        Fitted state means, shape ``(K,)``, sorted by ascending variance.
    variances_ : numpy.ndarray
        Fitted state variances, shape ``(K,)``, ascending.
    loglik_ : float
        Log-likelihood of the best fit.
    converged_ : bool
        Whether the best fit converged.
    n_iter_ : int
        Iterations taken by the best fit.
    loglik_trace_ : list of float
        Log-likelihood trace of the best fit.
    all_traces_ : list of list of float
        Log-likelihood traces of every initialisation.
    """

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 500,
        tol: float = 1e-8,
        n_init: int = 20,
        var_floor: float = 1e-10,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if n_states < 1:
            raise ValueError(f"n_states must be >= 1, got {n_states}")
        self.n_states = int(n_states)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.var_floor = float(var_floor)
        self.random_state = random_state

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _check_1d(x: np.ndarray) -> np.ndarray:
        """Validate and return a 1-D float64 observation array."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            x = x.ravel()
        if not np.all(np.isfinite(x)):
            raise ValueError("X contains NaN or inf; clean the series first")
        return x

    def _check_fitted(self) -> None:
        if not hasattr(self, "startprob_"):
            raise RuntimeError(
                "model is not fitted; call fit(X) before this method"
            )

    def _log_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Log start and transition parameters, with -inf where a prob is 0."""
        with np.errstate(divide="ignore"):
            return np.log(self.startprob_), np.log(self.transmat_)

    @classmethod
    def from_params(
        cls,
        startprob: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
    ) -> GaussianHMM:
        """Build a fitted model directly from known parameters.

        Useful for simulation and testing. The returned model is treated as
        fitted (its ``*_`` attributes are populated) but carries no EM history.

        Parameters
        ----------
        startprob : numpy.ndarray
            Initial distribution, shape ``(K,)``.
        transmat : numpy.ndarray
            Transition matrix, shape ``(K, K)``.
        means : numpy.ndarray
            State means, shape ``(K,)``.
        variances : numpy.ndarray
            State variances, shape ``(K,)``.

        Returns
        -------
        GaussianHMM
            A model with the given parameters set as fitted attributes.
        """
        startprob = np.asarray(startprob, dtype=np.float64)
        transmat = np.asarray(transmat, dtype=np.float64)
        means = np.asarray(means, dtype=np.float64)
        variances = np.asarray(variances, dtype=np.float64)
        k = startprob.shape[0]
        obj = cls(n_states=k)
        obj.startprob_ = startprob
        obj.transmat_ = transmat
        obj.means_ = means
        obj.variances_ = variances
        obj.loglik_ = float("nan")
        obj.converged_ = False
        obj.n_iter_ = 0
        obj.loglik_trace_ = []
        obj.all_traces_ = []
        return obj

    # -------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray) -> GaussianHMM:
        """Fit the model to a univariate return series by multi-start EM.

        Parameters
        ----------
        X : numpy.ndarray
            Observations, shape ``(T,)``.

        Returns
        -------
        GaussianHMM
            ``self``, fitted.
        """
        x = self._check_1d(X)
        if x.shape[0] < 10 * self.n_states:
            raise ValueError(
                f"need at least 10*K = {10 * self.n_states} observations for "
                f"K={self.n_states}, got {x.shape[0]}"
            )

        best, all_traces = fit_em(
            x,
            self.n_states,
            n_iter=self.n_iter,
            tol=self.tol,
            n_init=self.n_init,
            var_floor=self.var_floor,
            random_state=self.random_state,
        )
        self.startprob_ = best.startprob
        self.transmat_ = best.transmat
        self.means_ = best.means
        self.variances_ = best.variances
        self.loglik_ = best.loglik
        self.converged_ = best.converged
        self.n_iter_ = best.n_iter
        self.loglik_trace_ = best.loglik_trace
        self.all_traces_ = all_traces
        return self

    # ---------------------------------------------------------------- scoring
    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        return gaussian_log_emission(x, self.means_, self.variances_)

    def score(self, X: np.ndarray) -> float:
        """Log-likelihood ``log P(X)`` under the fitted model.

        Parameters
        ----------
        X : numpy.ndarray
            Observations, shape ``(T,)``.

        Returns
        -------
        float
            Sequence log-likelihood.
        """
        self._check_fitted()
        x = self._check_1d(X)
        log_sp, log_tm = self._log_params()
        _, loglik = log_forward(log_sp, log_tm, self._log_emission(x))
        return loglik

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most likely state path via Viterbi decoding.

        Parameters
        ----------
        X : numpy.ndarray
            Observations, shape ``(T,)``.

        Returns
        -------
        numpy.ndarray
            Integer state sequence, shape ``(T,)``.
        """
        self._check_fitted()
        x = self._check_1d(X)
        log_sp, log_tm = self._log_params()
        path, _ = viterbi(log_sp, log_tm, self._log_emission(x))
        return path

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Smoothed state posteriors ``P(S_t | X)``.

        Non-causal (conditions on the whole sequence). Use :meth:`filter` for a
        trading signal.

        Parameters
        ----------
        X : numpy.ndarray
            Observations, shape ``(T,)``.

        Returns
        -------
        numpy.ndarray
            Smoothed posteriors, shape ``(T, K)``.
        """
        self._check_fitted()
        x = self._check_1d(X)
        log_sp, log_tm = self._log_params()
        log_em = self._log_emission(x)
        log_a, _ = log_forward(log_sp, log_tm, log_em)
        log_b = log_backward(log_tm, log_em)
        return np.exp(log_gamma(log_a, log_b))

    def filter(self, X: np.ndarray) -> np.ndarray:
        """Filtered state probabilities ``P(S_t | X_0, ..., X_t)``.

        Causal: each row conditions only on data up to and including time ``t``.
        This is the trading-safe state estimate (see
        :func:`~regime_switching.inference.filtered_log_probs`).

        Parameters
        ----------
        X : numpy.ndarray
            Observations, shape ``(T,)``.

        Returns
        -------
        numpy.ndarray
            Filtered posteriors, shape ``(T, K)``.
        """
        self._check_fitted()
        x = self._check_1d(X)
        log_sp, log_tm = self._log_params()
        log_a, _ = log_forward(log_sp, log_tm, self._log_emission(x))
        return np.exp(filtered_log_probs(log_a))

    # --------------------------------------------------------------- sampling
    def sample(
        self, n_samples: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw observations and hidden states from the fitted model.

        Parameters
        ----------
        n_samples : int
            Number of steps ``T`` to generate.
        rng : numpy.random.Generator
            Explicit generator (global ``numpy.random`` state is never used).

        Returns
        -------
        X : numpy.ndarray
            Observations, shape ``(T,)``.
        states : numpy.ndarray
            Hidden state sequence, shape ``(T,)``.
        """
        self._check_fitted()
        states = np.empty(n_samples, dtype=np.int64)
        x = np.empty(n_samples, dtype=np.float64)
        sds = np.sqrt(self.variances_)

        states[0] = rng.choice(self.n_states, p=self.startprob_)
        for t in range(1, n_samples):
            states[t] = rng.choice(self.n_states, p=self.transmat_[states[t - 1]])
        for t in range(n_samples):
            k = states[t]
            x[t] = rng.normal(self.means_[k], sds[k])
        return x, states

    # -------------------------------------------------------- model selection
    def _n_parameters(self) -> int:
        """Number of free parameters: (K-1) + K(K-1) + K + K."""
        k = self.n_states
        return (k - 1) + k * (k - 1) + 2 * k

    def aic(self, X: np.ndarray) -> float:
        """Akaike information criterion ``2p - 2 logL``.

        Parameters
        ----------
        X : numpy.ndarray
            Observations, shape ``(T,)``.

        Returns
        -------
        float
            AIC (lower is better).
        """
        self._check_fitted()
        return 2.0 * self._n_parameters() - 2.0 * self.score(X)

    def bic(self, X: np.ndarray) -> float:
        """Bayesian information criterion ``p log T - 2 logL``.

        Parameters
        ----------
        X : numpy.ndarray
            Observations, shape ``(T,)``.

        Returns
        -------
        float
            BIC (lower is better).
        """
        self._check_fitted()
        x = self._check_1d(X)
        return self._n_parameters() * np.log(x.shape[0]) - 2.0 * self.score(x)

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Serialise the fitted model to JSON.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file.
        """
        self._check_fitted()
        payload = {
            "hyperparams": {
                "n_states": self.n_states,
                "n_iter": self.n_iter,
                "tol": self.tol,
                "n_init": self.n_init,
                "var_floor": self.var_floor,
            },
            "startprob_": self.startprob_.tolist(),
            "transmat_": self.transmat_.tolist(),
            "means_": self.means_.tolist(),
            "variances_": self.variances_.tolist(),
            "loglik_": self.loglik_,
            "converged_": bool(self.converged_),
            "n_iter_": int(self.n_iter_),
            "loglik_trace_": list(self.loglik_trace_),
            "all_traces_": [list(t) for t in self.all_traces_],
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> GaussianHMM:
        """Load a model previously written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            Source JSON file.

        Returns
        -------
        GaussianHMM
            The reconstructed, fitted model.
        """
        payload = json.loads(Path(path).read_text())
        hp = payload["hyperparams"]
        obj = cls(
            n_states=hp["n_states"],
            n_iter=hp["n_iter"],
            tol=hp["tol"],
            n_init=hp["n_init"],
            var_floor=hp["var_floor"],
        )
        obj.startprob_ = np.asarray(payload["startprob_"], dtype=np.float64)
        obj.transmat_ = np.asarray(payload["transmat_"], dtype=np.float64)
        obj.means_ = np.asarray(payload["means_"], dtype=np.float64)
        obj.variances_ = np.asarray(payload["variances_"], dtype=np.float64)
        obj.loglik_ = float(payload["loglik_"])
        obj.converged_ = bool(payload["converged_"])
        obj.n_iter_ = int(payload["n_iter_"])
        obj.loglik_trace_ = list(payload["loglik_trace_"])
        obj.all_traces_ = [list(t) for t in payload["all_traces_"]]
        return obj

    # ------------------------------------------------------------------ report
    def summary(self) -> str:
        """Readable parameter table including implied expected durations.

        Returns
        -------
        str
            A multi-line summary of the fitted parameters. The expected duration
            of state ``i`` is ``1 / (1 - transmat_[i, i])`` time steps.
        """
        self._check_fitted()
        durations = expected_durations(self.transmat_)
        lines = [
            f"GaussianHMM(n_states={self.n_states})  "
            f"loglik={self.loglik_:.4f}  converged={self.converged_}",
            "",
            f"{'state':>5} {'startprob':>10} {'mean':>12} {'std':>12} "
            f"{'self-trans':>11} {'exp_dur':>9}",
        ]
        for i in range(self.n_states):
            lines.append(
                f"{i:>5} {self.startprob_[i]:>10.4f} {self.means_[i]:>12.6f} "
                f"{np.sqrt(self.variances_[i]):>12.6f} "
                f"{self.transmat_[i, i]:>11.4f} {durations[i]:>9.1f}"
            )
        lines.append("")
        lines.append("transition matrix P(S_t=j | S_{t-1}=i):")
        for i in range(self.n_states):
            row = " ".join(f"{self.transmat_[i, j]:.4f}" for j in range(self.n_states))
            lines.append(f"  {i}: {row}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        if hasattr(self, "startprob_"):
            return (
                f"GaussianHMM(n_states={self.n_states}, fitted, "
                f"loglik={self.loglik_:.2f})"
            )
        return f"GaussianHMM(n_states={self.n_states}, unfitted)"
