"""Performance statistics and Sharpe-ratio inference.

Return series are periodic simple (or log) returns. Annualisation uses a
``periods_per_year`` factor (252 for daily data). Ratios whose denominator is
zero (e.g. a constant series has zero volatility) are reported as ``0.0`` rather
than raising.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "performance_summary",
    "sharpe_standard_error",
    "jobson_korkie_memmel",
    "deflated_sharpe_ratio",
    "turnover",
]

_EULER_MASCHERONI = 0.5772156649015329
_VOL_FLOOR = 1e-12  # a std below this is treated as zero (constant series)


def _as_returns(returns: np.ndarray) -> np.ndarray:
    """Coerce to a 1-D float64 return series."""
    return np.asarray(returns, dtype=np.float64).ravel()


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 when the denominator is ~0 (avoids div-by-zero)."""
    if abs(denominator) <= _VOL_FLOOR:
        return 0.0
    return numerator / denominator


def performance_summary(
    returns: np.ndarray, periods_per_year: int = 252
) -> dict[str, float]:
    """Headline performance statistics for a return series.

    Parameters
    ----------
    returns : numpy.ndarray
        Periodic returns, shape ``(T,)``.
    periods_per_year : int, optional
        Number of periods in a year (252 for daily).

    Returns
    -------
    dict of str to float
        Keys: ``cagr``, ``ann_vol``, ``sharpe``, ``sortino``, ``max_drawdown``,
        ``calmar``, ``hit_rate``, ``skew``, ``excess_kurtosis``, ``n_obs``. A
        constant series yields ``ann_vol == 0`` with the risk-adjusted ratios
        reported as ``0.0`` rather than raising.
    """
    r = _as_returns(returns)
    n = r.size
    if n == 0:
        return {
            "cagr": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
            "hit_rate": float("nan"),
            "skew": float("nan"),
            "excess_kurtosis": float("nan"),
            "n_obs": 0.0,
        }

    mean = float(np.mean(r))
    sd = float(np.std(r, ddof=1)) if n > 1 else 0.0
    if sd <= _VOL_FLOOR:
        sd = 0.0

    # compound growth path
    equity = np.cumprod(1.0 + r)
    growth = float(equity[-1])
    cagr = growth ** (periods_per_year / n) - 1.0 if growth > 0 else -1.0

    ann_vol = sd * np.sqrt(periods_per_year)
    sharpe = _safe_div(mean, sd) * np.sqrt(periods_per_year)

    downside = np.minimum(r, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    sortino = _safe_div(mean, downside_dev) * np.sqrt(periods_per_year)

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())
    calmar = _safe_div(cagr, abs(max_dd))

    return {
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "hit_rate": float(np.mean(r > 0.0)),
        # higher moments are undefined for a (near-)constant series
        "skew": float(stats.skew(r)) if (n > 2 and sd > 0.0) else 0.0,
        "excess_kurtosis": (
            float(stats.kurtosis(r, fisher=True)) if (n > 3 and sd > 0.0) else 0.0
        ),
        "n_obs": float(n),
    }


def sharpe_standard_error(
    returns: np.ndarray, periods_per_year: int = 252
) -> float:
    """Standard error of the (annualised) Sharpe ratio.

    Computes ``sqrt((1 + 0.5 * SR**2) / T)`` in per-period units (Lo, 2002) and
    annualises by ``sqrt(periods_per_year)``, where ``SR`` is the per-period
    Sharpe ratio and ``T`` the number of observations.

    Parameters
    ----------
    returns : numpy.ndarray
        Periodic returns, shape ``(T,)``.
    periods_per_year : int, optional
        Periods in a year.

    Returns
    -------
    float
        Annualised standard error of the Sharpe ratio.

    Notes
    -----
    This assumes i.i.d. normal returns. It **understates** the true uncertainty
    when returns are autocorrelated or fat-tailed; for those cases prefer a
    HAC / bootstrap estimate.
    """
    r = _as_returns(returns)
    t = r.size
    if t < 2:
        return float("nan")
    sd = np.std(r, ddof=1)
    sr_period = _safe_div(float(np.mean(r)), float(sd))
    se_period = np.sqrt((1.0 + 0.5 * sr_period**2) / t)
    return float(se_period * np.sqrt(periods_per_year))


def jobson_korkie_memmel(
    returns_a: np.ndarray, returns_b: np.ndarray
) -> tuple[float, float]:
    """Test equality of two Sharpe ratios (Jobson-Korkie with Memmel's fix).

    Tests ``H0: SR_a = SR_b`` for two return series measured over the *same*
    sample, using the Memmel (2003) corrected asymptotic variance, which
    accounts for the correlation between the two series.

    Parameters
    ----------
    returns_a : numpy.ndarray
        First return series, shape ``(T,)``.
    returns_b : numpy.ndarray
        Second return series, shape ``(T,)``, paired with ``returns_a``.

    Returns
    -------
    z_stat : float
        Test statistic, asymptotically standard normal under ``H0``.
    p_value : float
        Two-sided p-value.

    Notes
    -----
    This is the correct test when comparing a strategy to a benchmark over the
    same sample: the two Sharpe ratios are estimated on correlated data, so
    comparing their standard errors separately (as if independent) is wrong and
    overstates significance.
    """
    a = _as_returns(returns_a)
    b = _as_returns(returns_b)
    if a.size != b.size:
        raise ValueError("return series must have equal length")
    n = a.size
    if n < 3:
        raise ValueError("need at least 3 observations")

    mu_a, mu_b = np.mean(a), np.mean(b)
    var_a = np.var(a, ddof=0)
    var_b = np.var(b, ddof=0)
    sd_a, sd_b = np.sqrt(var_a), np.sqrt(var_b)
    if sd_a <= _VOL_FLOOR or sd_b <= _VOL_FLOOR:
        raise ValueError("a series has (near) zero variance; Sharpe undefined")

    sr_a = mu_a / sd_a
    sr_b = mu_b / sd_b
    corr = np.mean((a - mu_a) * (b - mu_b)) / (sd_a * sd_b)

    # Memmel (2003) asymptotic variance of the Sharpe-ratio difference
    theta = (
        2.0
        - 2.0 * corr
        + 0.5 * (sr_a**2 + sr_b**2 - 2.0 * sr_a * sr_b * corr**2)
    ) / n

    diff = sr_a - sr_b
    if theta <= 0:
        # perfectly correlated, identical-Sharpe series: no evidence against H0
        if abs(diff) < 1e-12:
            return 0.0, 1.0
        raise ValueError("non-positive variance estimate")

    z_stat = diff / np.sqrt(theta)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    return float(z_stat), float(p_value)


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    T: int,
    skew: float,
    kurt: float,
) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014).

    Adjusts an observed Sharpe ratio for **selection bias**: when many strategy
    variants are tried, the best one's Sharpe is inflated by multiple testing.
    The DSR is the probability that the true Sharpe exceeds zero after that
    adjustment.

    Parameters
    ----------
    observed_sr : float
        Observed **per-period** (non-annualised) Sharpe ratio of the selected
        strategy.
    n_trials : int
        Number of independent strategy configurations tried.
    T : int
        Number of return observations.
    skew : float
        Skewness of the selected strategy's returns.
    kurt : float
        Non-excess kurtosis of the returns (3 for a normal distribution).

    Returns
    -------
    float
        Deflated Sharpe ratio in ``[0, 1]`` (a probability).

    Notes
    -----
    The variance of the Sharpe estimates across trials is taken to equal the
    single-strategy Sharpe estimation variance (the Mertens standard error);
    the expected maximum of ``n_trials`` such estimates gives the benchmark
    ``SR_0`` that ``observed_sr`` must clear.
    """
    if n_trials < 1 or T < 2:
        return float("nan")
    # Mertens standard error of the (per-period) Sharpe under non-normality
    se = np.sqrt(
        (1.0 - skew * observed_sr + (kurt - 1.0) / 4.0 * observed_sr**2) / (T - 1.0)
    )
    if not np.isfinite(se) or se <= 0:
        return float("nan")

    # expected maximum of n_trials standard-normal draws (Bailey-Lopez de Prado)
    n = float(n_trials)
    z_max = (1.0 - _EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n) + (
        _EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    )
    sr0 = se * z_max
    return float(stats.norm.cdf((observed_sr - sr0) / se))


def turnover(positions: np.ndarray) -> float:
    """Average per-period turnover of a position series.

    Turnover is the mean absolute change in position from one period to the
    next: ``mean(|position_t - position_{t-1}|)``. A constant position has zero
    turnover.

    Parameters
    ----------
    positions : numpy.ndarray
        Position (exposure) series, shape ``(T,)``.

    Returns
    -------
    float
        Mean absolute period-over-period position change.
    """
    p = _as_returns(positions)
    if p.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(p))))
