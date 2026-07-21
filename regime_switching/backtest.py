"""Walk-forward regime-switching backtest and benchmarks.

The backtest is strictly causal. The timing convention, enforced throughout, is:

    the position applied to the return of day t is a function of information
    available at the close of day t-1.

Concretely, the filtered state probability estimated from data up to a rebalance
is turned into a position, and that position is **shifted forward one day** before
it earns any return. The shift is asserted, not merely commented.

GaussianHMM sorts its states by ascending variance, so state 0 is always the
low-variance ("calm") regime and its probability column is comparable across
refits.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .hmm import GaussianHMM

__all__ = [
    "walk_forward_regimes",
    "regime_strategy",
    "benchmarks",
    "compare",
]

logger = logging.getLogger(__name__)

LOW_VAR_STATE = 0  # GaussianHMM sorts states by ascending variance


def _returns_from_features(features: pd.DataFrame) -> pd.Series:
    """Extract the asset return column from a features frame."""
    for col in ("returns", "log_return", "return"):
        if col in features.columns:
            return features[col].astype(float)
    raise ValueError(
        "features must contain a 'returns' or 'log_return' column"
    )


def walk_forward_regimes(
    returns: np.ndarray,
    dates: np.ndarray,
    n_states: int = 2,
    train_window: int = 1000,
    refit_every: int = 21,
    expanding: bool = True,
    n_init: int = 10,
    random_state: int = 0,
) -> pd.DataFrame:
    """Walk-forward regime detection with periodic HMM refits.

    At each rebalance date ``t`` (every ``refit_every`` days, starting once
    ``train_window`` observations are available):

    - fit a :class:`GaussianHMM` on ``returns[:t]`` (expanding) or on the
      trailing ``train_window`` (``expanding=False``);
    - compute filtered probabilities on that history;
    - take the filtered probability at the last available observation (the close
      of day ``t-1``);
    - hold the resulting position for the next ``refit_every`` days.

    Timing convention (enforced by an assertion below): the position applied to
    the return of day ``t`` is a function of information available at the close
    of day ``t-1``. The per-state filtered probability is placed at the rebalance
    row (the estimate available going into that block) and the position is that
    signal shifted forward one day, so it never uses the current day's return.

    Parameters
    ----------
    returns : numpy.ndarray
        Return series, shape ``(T,)``.
    dates : numpy.ndarray
        Date labels aligned with ``returns``, shape ``(T,)``.
    n_states : int, optional
        Number of regimes ``K``.
    train_window : int, optional
        Minimum history (and rolling window when ``expanding=False``).
    refit_every : int, optional
        Days between refits; the position is held across the block.
    expanding : bool, optional
        Expanding window if ``True``, else a trailing ``train_window``.
    n_init : int, optional
        Multi-start count for each refit.
    random_state : int, optional
        Base seed. Each rebalance is seeded deterministically from it and the
        rebalance index, so truncating the input leaves earlier fits unchanged.

    Returns
    -------
    pandas.DataFrame
        Indexed by date, with columns ``p_state_0 ... p_state_{K-1}`` (filtered
        probabilities, held across each block), ``selected_state`` (argmax), and
        ``position`` (the causal, one-day-lagged low-variance-regime signal).
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    n_obs = returns.shape[0]
    dates = np.asarray(dates)
    if dates.shape[0] != n_obs:
        raise ValueError("returns and dates must align in length")
    if train_window >= n_obs:
        raise ValueError("train_window must be smaller than the sample length")

    prob = np.full((n_obs, n_states), np.nan, dtype=np.float64)

    rebalance_points = range(train_window, n_obs, refit_every)
    for j, t in enumerate(rebalance_points):
        history = returns[:t] if expanding else returns[t - train_window : t]
        model = GaussianHMM(
            n_states=n_states, n_init=n_init, random_state=random_state + j
        ).fit(history)
        filtered = model.filter(history)  # (len(history), K), causal
        # filtered[-1] = P(S_{t-1} | information through the close of day t-1)
        prob[t] = filtered[-1]
        logger.info(
            "rebalance %d at index %d: P(low-var)=%.3f", j, t, filtered[-1, LOW_VAR_STATE]
        )

    frame = pd.DataFrame(
        prob, columns=[f"p_state_{k}" for k in range(n_states)], index=pd.Index(dates, name="date")
    )
    # hold each rebalance estimate across its block (piecewise-constant)
    frame = frame.ffill()

    # contemporaneous signal implied by the held probabilities...
    selected_state = frame.to_numpy().argmax(axis=1).astype(float)
    all_nan = frame.isna().all(axis=1).to_numpy()
    selected_state[all_nan] = np.nan
    frame["selected_state"] = selected_state

    signal = (frame[f"p_state_{LOW_VAR_STATE}"] >= 0.5).astype(float)
    signal[frame[f"p_state_{LOW_VAR_STATE}"].isna()] = 0.0
    # ...applied one day later: position for day t uses the signal through t-1
    position = signal.shift(1).fillna(0.0)
    frame["position"] = position

    # ENFORCE the timing convention: position is exactly the one-day lag of the
    # signal (never same-day). If this ever fails, the strategy is peeking.
    assert np.array_equal(
        frame["position"].to_numpy(), signal.shift(1).fillna(0.0).to_numpy()
    ), "position must be the one-day lag of the regime signal (look-ahead guard)"

    return frame


def _sized_signal(
    p_low: np.ndarray, mode: str, max_leverage: float
) -> np.ndarray:
    """Map low-variance-regime probability to a raw exposure (pre-lag)."""
    if mode == "binary":
        raw = (p_low >= 0.5).astype(float)
    elif mode == "probability":
        raw = p_low.astype(float)
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'binary' or 'probability'")
    return np.clip(raw, 0.0, max_leverage)


def regime_strategy(
    features: pd.DataFrame,
    probs: np.ndarray | pd.DataFrame | pd.Series,
    mode: str = "probability",
    target_vol: float = 0.10,
    max_leverage: float = 1.0,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Size a regime signal into positions and compute gross/net returns.

    Two sizing modes:

    - ``"binary"``: fully invested (1.0) in the low-variance state, flat
      otherwise.
    - ``"probability"``: exposure scaled by ``P(low-variance state)``.

    Exposure is capped at ``max_leverage``. The position is applied to the
    *next* day's return (one-day lag), so no trade uses contemporaneous
    information. Transaction costs are proportional to the absolute change in
    position, charged in basis points.

    Parameters
    ----------
    features : pandas.DataFrame
        Must contain a ``returns`` or ``log_return`` column (the asset return),
        indexed by date.
    probs : array_like, pandas.DataFrame, or pandas.Series
        Low-variance-regime probability per date. A 2-D input is interpreted as
        per-state probabilities and column 0 (``p_state_0``) is used.
    mode : str, optional
        ``"binary"`` or ``"probability"``.
    target_vol : float, optional
        Reserved for volatility-scaled sizing (used by :func:`benchmarks`); the
        two modes here are unlevered exposures and do not consume it. Must be
        positive.
    max_leverage : float, optional
        Cap on absolute exposure.
    cost_bps : float, optional
        Per-unit-turnover transaction cost in basis points.

    Returns
    -------
    pandas.DataFrame
        Indexed by date with columns ``position``, ``gross``, ``net``, ``cost``.
    """
    if target_vol <= 0:
        raise ValueError("target_vol must be positive")
    returns = _returns_from_features(features)
    index = features.index

    p = probs.to_numpy() if isinstance(probs, (pd.DataFrame, pd.Series)) else np.asarray(probs)
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 2:
        p_low = p[:, LOW_VAR_STATE]
    else:
        p_low = p.ravel()
    if p_low.shape[0] != len(index):
        raise ValueError("probs and features must align in length")

    signal = _sized_signal(np.nan_to_num(p_low, nan=0.0), mode, max_leverage)
    signal = pd.Series(signal, index=index)
    # causal: position on day t uses the signal formed at the close of day t-1
    position = signal.shift(1).fillna(0.0)

    gross = position * returns
    # cost proportional to |change in position|; day 0 trades in from flat
    dpos = position.diff()
    dpos.iloc[0] = position.iloc[0]
    cost = (cost_bps / 1e4) * dpos.abs()
    net = gross - cost

    return pd.DataFrame(
        {"position": position, "gross": gross, "net": net, "cost": cost},
        index=index,
    )


def benchmarks(
    returns: np.ndarray,
    features: pd.DataFrame,
    target_vol: float = 0.10,
    max_leverage: float = 1.0,
    cost_bps: float = 5.0,
    avg_exposure: float | None = None,
) -> pd.DataFrame:
    """Benchmark net-return series to compare a regime strategy against.

    Columns returned (all **net** of costs):

    - ``buy_and_hold``: fully invested at all times.
    - ``vol_target``: ``position = target_vol / trailing_realised_vol``, capped
      at ``max_leverage``, applied with a one-day lag.
    - ``constant_exposure``: a fixed position equal to ``avg_exposure`` (or, if
      not given, the average exposure of the ``vol_target`` benchmark). This
      isolates the *timing* of vol targeting from its average leverage.

    Parameters
    ----------
    returns : numpy.ndarray
        Asset return series, shape ``(T,)``, aligned with ``features``.
    features : pandas.DataFrame
        Must contain ``realised_vol`` (annualised trailing volatility), indexed
        by date.
    target_vol : float, optional
        Annualised volatility target for the ``vol_target`` benchmark.
    max_leverage : float, optional
        Cap on the vol-target position.
    cost_bps : float, optional
        Per-unit-turnover transaction cost in basis points.
    avg_exposure : float, optional
        Exposure for ``constant_exposure``. Pass the strategy's average exposure
        to compare like for like; defaults to the vol-target average.

    Returns
    -------
    pandas.DataFrame
        Indexed by date, one net-return column per benchmark.

    Notes
    -----
    ``vol_target`` is the benchmark that matters. A large share of what regime
    switching appears to deliver is also available from simply scaling exposure
    by inverse trailing volatility; a regime model that does not beat vol
    targeting is not adding anything.

    ``buy_and_hold`` understates the true return of holding: both supported
    equity series (S&P 500, EURO STOXX 50) are **price indices and exclude
    dividends**, so holding actually earns roughly 1.5% a year more for the S&P
    500 and roughly 3% a year more for the EURO STOXX 50. The distortion — which
    flatters any strategy measured against buy-and-hold — is therefore
    materially larger in the European configuration, and should be kept in mind
    when reading the results table.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    index = features.index
    if "realised_vol" not in features.columns:
        raise ValueError("features must contain a 'realised_vol' column")
    r_ser = pd.Series(r, index=index)
    cost_rate = cost_bps / 1e4

    def _net(position: pd.Series) -> pd.Series:
        pos = position.shift(1).fillna(0.0)  # causal one-day lag
        dpos = pos.diff()
        dpos.iloc[0] = pos.iloc[0]
        return pos * r_ser - cost_rate * dpos.abs()

    # buy and hold: fully invested, so the lag is a no-op after day 0
    bh_pos = pd.Series(1.0, index=index)

    # vol target: scale by inverse trailing realised vol, capped
    rv = features["realised_vol"].astype(float)
    vt_raw = (target_vol / rv).clip(upper=max_leverage)
    vt_pos = vt_raw.fillna(0.0)

    if avg_exposure is None:
        # average of the *applied* (lagged) vol-target exposure
        avg_exposure = float(vt_pos.shift(1).fillna(0.0).mean())
    ce_pos = pd.Series(avg_exposure, index=index)

    return pd.DataFrame(
        {
            "buy_and_hold": _net(bh_pos),
            "vol_target": _net(vt_pos),
            "constant_exposure": _net(ce_pos),
        },
        index=index,
    )


def compare(
    strategy: pd.DataFrame,
    bench: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Side-by-side performance table with Sharpe-difference tests.

    Computes :func:`~regime_switching.metrics.performance_summary` for the
    strategy and each benchmark, then appends the Jobson-Korkie-Memmel
    z-statistic and p-value of the strategy against each benchmark.

    Parameters
    ----------
    strategy : pandas.DataFrame
        Output of :func:`regime_strategy`; its ``net`` column is used.
    bench : pandas.DataFrame
        Output of :func:`benchmarks`; each column is a benchmark net-return
        series.
    periods_per_year : int, optional
        Annualisation factor.

    Returns
    -------
    pandas.DataFrame
        Metrics as rows; ``strategy`` and each benchmark as columns, plus
        ``jk_z`` and ``jk_p`` rows (the strategy's own column leaves these NaN).

    Notes
    -----
    The ``buy_and_hold`` column understates true holding returns because the
    underlying equity indices exclude dividends (see :func:`benchmarks`); read
    any outperformance over it with that upward bias in mind.
    """
    from .metrics import jobson_korkie_memmel, performance_summary

    strat_net = strategy["net"].to_numpy()
    columns: dict[str, dict[str, float]] = {}

    strat_summary = performance_summary(strat_net, periods_per_year)
    strat_summary["jk_z"] = float("nan")
    strat_summary["jk_p"] = float("nan")
    columns["strategy"] = strat_summary

    for name in bench.columns:
        bnet = bench[name].to_numpy()
        summary = performance_summary(bnet, periods_per_year)
        z, p = jobson_korkie_memmel(strat_net, bnet)
        summary["jk_z"] = z
        summary["jk_p"] = p
        columns[name] = summary

    return pd.DataFrame(columns)
