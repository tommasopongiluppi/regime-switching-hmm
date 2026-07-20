"""Performance statistics and Sharpe-ratio difference tests.

Return series are treated as simple periodic returns unless stated otherwise.
Annualisation uses a ``periods_per_year`` factor (252 for daily data).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "hit_rate",
    "summary",
    "jobson_korkie_test",
]


def _as_returns(returns: np.ndarray) -> np.ndarray:
    """Coerce to a 1-D float64 return series."""
    r = np.asarray(returns, dtype=np.float64).ravel()
    return r


def annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Geometric annualised return.

    Parameters
    ----------
    returns : numpy.ndarray
        Periodic simple returns, shape ``(T,)``.
    periods_per_year : int, optional
        Number of periods in a year (252 for daily).

    Returns
    -------
    float
        Annualised compound growth rate.
    """
    r = _as_returns(returns)
    if r.size == 0:
        return float("nan")
    growth = np.prod(1.0 + r)
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / r.size) - 1.0)


def annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised volatility (standard deviation of returns).

    Parameters
    ----------
    returns : numpy.ndarray
        Periodic returns, shape ``(T,)``.
    periods_per_year : int, optional
        Periods in a year.

    Returns
    -------
    float
        Annualised volatility.
    """
    r = _as_returns(returns)
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns : numpy.ndarray
        Periodic returns, shape ``(T,)``.
    risk_free : float, optional
        Per-period risk-free rate.
    periods_per_year : int, optional
        Periods in a year.

    Returns
    -------
    float
        Annualised Sharpe ratio. Returns ``nan`` if volatility is zero.
    """
    r = _as_returns(returns) - risk_free
    if r.size < 2:
        return float("nan")
    sd = np.std(r, ddof=1)
    # Treat a numerically negligible dispersion (e.g. a constant series, whose
    # sample std is ~1e-19 rather than exactly 0) as an undefined Sharpe ratio.
    if sd <= 1e-12:
        return float("nan")
    return float(np.mean(r) / sd * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown of the cumulative return path.

    Parameters
    ----------
    returns : numpy.ndarray
        Periodic simple returns, shape ``(T,)``.

    Returns
    -------
    float
        Maximum drawdown as a non-positive fraction (e.g. ``-0.25`` for a 25%
        peak-to-trough loss).
    """
    r = _as_returns(returns)
    if r.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def hit_rate(returns: np.ndarray) -> float:
    """Fraction of periods with a strictly positive return.

    Parameters
    ----------
    returns : numpy.ndarray
        Periodic returns, shape ``(T,)``.

    Returns
    -------
    float
        Proportion of positive-return periods, in ``[0, 1]``.
    """
    r = _as_returns(returns)
    if r.size == 0:
        return float("nan")
    return float(np.mean(r > 0.0))


def summary(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Bundle of headline performance statistics.

    Parameters
    ----------
    returns : numpy.ndarray
        Periodic returns, shape ``(T,)``.
    risk_free : float, optional
        Per-period risk-free rate.
    periods_per_year : int, optional
        Periods in a year.

    Returns
    -------
    dict of str to float
        Keys: ``ann_return``, ``ann_vol``, ``sharpe``, ``max_drawdown``,
        ``hit_rate``, ``n_periods``.
    """
    r = _as_returns(returns)
    return {
        "ann_return": annualized_return(r, periods_per_year),
        "ann_vol": annualized_volatility(r, periods_per_year),
        "sharpe": sharpe_ratio(r, risk_free, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "hit_rate": hit_rate(r),
        "n_periods": float(r.size),
    }


def jobson_korkie_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
) -> tuple[float, float]:
    """Test the difference between two Sharpe ratios (Jobson-Korkie / Memmel).

    Tests ``H0: SR_a = SR_b`` for two return series measured over the same
    periods, using the Jobson-Korkie statistic with the Memmel (2003)
    correction. The two series must be paired and equal length.

    Parameters
    ----------
    returns_a : numpy.ndarray
        First return series, shape ``(T,)``.
    returns_b : numpy.ndarray
        Second return series, shape ``(T,)``, aligned with ``returns_a``.

    Returns
    -------
    z_stat : float
        Test statistic, asymptotically standard normal under ``H0``.
    p_value : float
        Two-sided p-value.

    Notes
    -----
    Sharpe ratios here are in *per-period* units (the annualisation factor
    cancels in the difference test). References: Jobson & Korkie (1981);
    Memmel (2003), "Performance hypothesis testing with the Sharpe ratio".
    """
    a = _as_returns(returns_a)
    b = _as_returns(returns_b)
    if a.size != b.size:
        raise ValueError("return series must have equal length")
    n = a.size
    if n < 3:
        raise ValueError("need at least 3 observations")

    mu_a, mu_b = np.mean(a), np.mean(b)
    # population (ddof=0) moments, matching the Memmel derivation
    var_a = np.var(a, ddof=0)
    var_b = np.var(b, ddof=0)
    sd_a = np.sqrt(var_a)
    sd_b = np.sqrt(var_b)
    if sd_a == 0 or sd_b == 0:
        raise ValueError("a series has zero variance")

    sr_a = mu_a / sd_a
    sr_b = mu_b / sd_b
    cov_ab = np.mean((a - mu_a) * (b - mu_b))
    corr = cov_ab / (sd_a * sd_b)

    # Memmel (2003) asymptotic variance of the Sharpe-ratio difference
    theta = (
        2.0
        - 2.0 * corr
        + 0.5 * (sr_a**2 + sr_b**2 - 2.0 * sr_a * sr_b * corr**2)
    ) / n
    diff = sr_a - sr_b
    if theta <= 0:
        # Degenerate case: perfectly correlated, identical-Sharpe series give a
        # zero-variance estimate and a zero difference (0/0). The two Sharpe
        # ratios are indistinguishable, so report no evidence against H0.
        if abs(diff) < 1e-12:
            return 0.0, 1.0
        raise ValueError("non-positive variance estimate")

    z_stat = diff / np.sqrt(theta)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    return float(z_stat), float(p_value)
