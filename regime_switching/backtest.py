"""Walk-forward regime-switching strategy on filtered probabilities.

The backtest is strictly causal. The HMM is refit on a trailing window and, for
each out-of-sample block, the model produces **filtered** state probabilities
``P(S_t | x_1, ..., x_t)`` (never smoothed). Positions are formed from the
filtered signal and applied to the *next* period's return, so no future
information leaks into any trade.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .estimation import fit_multistart
from .hmm import GaussianHMM

__all__ = ["BacktestResult", "walk_forward_backtest", "align_states_by_mean"]

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Output of a walk-forward backtest.

    Attributes
    ----------
    strategy_returns : numpy.ndarray
        Realised per-period strategy returns, shape ``(M,)``, aligned with
        ``dates`` and ``asset_returns``.
    positions : numpy.ndarray
        Position held into each period, shape ``(M,)``.
    filtered_prob : numpy.ndarray
        Filtered state probability used to form each position, shape
        ``(M, K)``.
    asset_returns : numpy.ndarray
        Buy-and-hold returns over the same periods, shape ``(M,)``.
    dates : numpy.ndarray
        Period labels aligned with the returns, shape ``(M,)``, or ``None``.
    """

    strategy_returns: np.ndarray
    positions: np.ndarray
    filtered_prob: np.ndarray
    asset_returns: np.ndarray
    dates: np.ndarray | None = None


def align_states_by_mean(model: GaussianHMM) -> np.ndarray:
    """Order state indices by ascending emission mean (first feature).

    Baum-Welch labels states arbitrarily, so a fitted model's state ``0`` is not
    comparable across refits. This returns a permutation that sorts states from
    lowest to highest mean of the first observation dimension, giving a stable
    "bear -> bull" ordering for return data.

    Parameters
    ----------
    model : GaussianHMM
        A fitted model.

    Returns
    -------
    numpy.ndarray
        Index array ``order`` of shape ``(K,)`` such that ``order[0]`` is the
        lowest-mean state.
    """
    return np.argsort(model.means[:, 0])


def _default_position_rule(filtered_row: np.ndarray, order: np.ndarray) -> float:
    """Long the highest-mean state, flat otherwise.

    Uses the filtered probability of the top-ranked (highest-mean) state as the
    long weight, giving a probability-weighted long-only position in ``[0, 1]``.
    """
    bull_state = int(order[-1])
    return float(filtered_row[bull_state])


def walk_forward_backtest(
    observations: np.ndarray,
    asset_returns: np.ndarray,
    n_states: int,
    rng: np.random.Generator,
    *,
    train_size: int,
    step: int = 21,
    n_starts: int = 5,
    position_rule: Callable[[np.ndarray, np.ndarray], float] | None = None,
    dates: np.ndarray | None = None,
) -> BacktestResult:
    """Run a walk-forward regime-switching backtest.

    The model is fit on the first ``train_size`` observations, then used to
    filter the next ``step`` observations out of sample; the window then rolls
    forward by ``step`` and the model is refit. Positions derived from the
    filtered probability at time ``t`` are applied to the return from ``t`` to
    ``t + 1``, so trades never use contemporaneous or future information.

    Parameters
    ----------
    observations : numpy.ndarray
        Model input features, shape ``(T,)`` or ``(T, D)``.
    asset_returns : numpy.ndarray
        Per-period asset returns aligned with ``observations``, shape ``(T,)``.
    n_states : int
        Number of hidden states ``K``.
    rng : numpy.random.Generator
        Explicit generator passed through to estimation.
    train_size : int
        Number of observations in the initial (and rolling) training window.
    step : int, optional
        Number of out-of-sample periods between refits.
    n_starts : int, optional
        Multi-start count for each refit.
    position_rule : callable, optional
        Maps ``(filtered_row (K,), order (K,))`` to a position weight. Defaults
        to a probability-weighted long of the highest-mean state.
    dates : numpy.ndarray, optional
        Period labels aligned with ``observations``.

    Returns
    -------
    BacktestResult
        Realised strategy returns and diagnostics over the out-of-sample span.
    """
    obs = GaussianHMM._as_2d(observations)
    asset_returns = np.asarray(asset_returns, dtype=np.float64).ravel()
    n_obs = obs.shape[0]
    if asset_returns.shape[0] != n_obs:
        raise ValueError("observations and asset_returns must align in length")
    if train_size >= n_obs:
        raise ValueError("train_size must be smaller than the sample length")
    if position_rule is None:
        position_rule = _default_position_rule

    positions: list[float] = []
    filt_rows: list[np.ndarray] = []
    strat_ret: list[float] = []
    asset_ret: list[float] = []
    out_dates: list[object] = []

    start = train_size
    model: GaussianHMM | None = None
    order: np.ndarray | None = None

    while start < n_obs - 1:
        stop = min(start + step, n_obs - 1)

        # Refit on the trailing window ending just before this OOS block.
        train = obs[start - train_size : start]
        model = fit_multistart(
            train, n_states, rng, n_starts=n_starts
        ).model
        order = align_states_by_mean(model)

        # Filter the block *including* history up to each point. We filter the
        # full sequence up to `stop` and read the causal rows in [start, stop).
        block = obs[: stop]
        filtered, _ = model.filter(block)

        for t in range(start, stop):
            weight = position_rule(filtered[t], order)
            # position formed at t is applied to the t -> t+1 return
            positions.append(weight)
            filt_rows.append(filtered[t])
            strat_ret.append(weight * asset_returns[t + 1])
            asset_ret.append(asset_returns[t + 1])
            if dates is not None:
                out_dates.append(dates[t + 1])

        start = stop

    logger.info(
        "walk-forward produced %d out-of-sample periods", len(strat_ret)
    )
    return BacktestResult(
        strategy_returns=np.asarray(strat_ret, dtype=np.float64),
        positions=np.asarray(positions, dtype=np.float64),
        filtered_prob=np.asarray(filt_rows, dtype=np.float64),
        asset_returns=np.asarray(asset_ret, dtype=np.float64),
        dates=np.asarray(out_dates) if dates is not None else None,
    )
