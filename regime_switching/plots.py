"""Figures for regime-switching analysis.

Every function takes an optional Matplotlib ``ax`` and returns the ``Axes`` it
drew on, so plots compose into larger figures. Nothing is shown or saved here;
callers decide via ``fig.savefig`` / ``plt.show``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "plot_regime_probabilities",
    "plot_price_with_regimes",
    "plot_transition_matrix",
    "plot_returns_by_regime",
    "plot_equity_curves",
]


def _get_ax(ax):
    """Return ``ax`` or a fresh one. Imports Matplotlib lazily."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_regime_probabilities(
    prob: np.ndarray,
    dates: np.ndarray | None = None,
    ax=None,
    labels: list[str] | None = None,
):
    """Stacked area plot of state probabilities over time.

    Parameters
    ----------
    prob : numpy.ndarray
        State probabilities, shape ``(T, K)``. Rows should sum to 1.
    dates : numpy.ndarray, optional
        X-axis labels, shape ``(T,)``. Defaults to an integer index.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new one is created if omitted.
    labels : list of str, optional
        Legend labels per state.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _get_ax(ax)
    prob = np.asarray(prob, dtype=np.float64)
    t, k = prob.shape
    x = np.arange(t) if dates is None else dates
    labels = labels or [f"state {i}" for i in range(k)]
    ax.stackplot(x, prob.T, labels=labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("P(state)")
    ax.set_title("Regime probabilities")
    ax.legend(loc="upper left", ncol=k)
    return ax


def plot_price_with_regimes(
    prices: np.ndarray,
    states: np.ndarray,
    dates: np.ndarray | None = None,
    ax=None,
):
    """Price line shaded by the decoded regime.

    Parameters
    ----------
    prices : numpy.ndarray
        Price level, shape ``(T,)``.
    states : numpy.ndarray
        Integer regime label per period, shape ``(T,)``.
    dates : numpy.ndarray, optional
        X-axis labels, shape ``(T,)``. Defaults to an integer index.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    import matplotlib.pyplot as plt

    ax = _get_ax(ax)
    prices = np.asarray(prices, dtype=np.float64)
    states = np.asarray(states)
    t = prices.shape[0]
    x = np.arange(t) if dates is None else dates

    ax.plot(x, prices, color="black", linewidth=1.0)
    unique = np.unique(states)
    cmap = plt.get_cmap("tab10")
    for s in unique:
        mask = states == s
        ax.fill_between(
            x, prices.min(), prices.max(), where=mask,
            color=cmap(int(s) % 10), alpha=0.15, step="mid",
            label=f"state {int(s)}",
        )
    ax.set_ylabel("price")
    ax.set_title("Price with decoded regimes")
    ax.legend(loc="upper left")
    return ax


def plot_transition_matrix(transmat: np.ndarray, ax=None):
    """Heatmap of a transition matrix.

    Parameters
    ----------
    transmat : numpy.ndarray
        Transition matrix, shape ``(K, K)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _get_ax(ax)
    transmat = np.asarray(transmat, dtype=np.float64)
    k = transmat.shape[0]
    im = ax.imshow(transmat, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xlabel("to state j")
    ax.set_ylabel("from state i")
    ax.set_title("Transition matrix")
    for i in range(k):
        for j in range(k):
            ax.text(
                j, i, f"{transmat[i, j]:.2f}",
                ha="center", va="center",
                color="white" if transmat[i, j] > 0.5 else "black",
            )
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_returns_by_regime(
    returns: np.ndarray,
    states: np.ndarray,
    ax=None,
):
    """Overlaid histograms of returns grouped by regime.

    Parameters
    ----------
    returns : numpy.ndarray
        Per-period returns, shape ``(T,)``.
    states : numpy.ndarray
        Integer regime label per period, shape ``(T,)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    import matplotlib.pyplot as plt

    ax = _get_ax(ax)
    returns = np.asarray(returns, dtype=np.float64)
    states = np.asarray(states)
    cmap = plt.get_cmap("tab10")
    for s in np.unique(states):
        vals = returns[states == s]
        ax.hist(
            vals, bins=40, alpha=0.5, density=True,
            color=cmap(int(s) % 10), label=f"state {int(s)}",
        )
    ax.set_xlabel("return")
    ax.set_ylabel("density")
    ax.set_title("Return distribution by regime")
    ax.legend()
    return ax


def plot_equity_curves(
    curves: dict[str, np.ndarray],
    dates: np.ndarray | None = None,
    ax=None,
):
    """Cumulative equity curves from per-period return series.

    Parameters
    ----------
    curves : dict of str to numpy.ndarray
        Mapping from label to a per-period return series of shape ``(T,)``.
        Each is compounded into an equity curve starting at 1.
    dates : numpy.ndarray, optional
        X-axis labels shared by all curves.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    ax = _get_ax(ax)
    for label, returns in curves.items():
        equity = np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))
        x = np.arange(equity.shape[0]) if dates is None else dates
        ax.plot(x, equity, label=label)
    ax.set_ylabel("equity (start = 1)")
    ax.set_title("Cumulative performance")
    ax.legend()
    return ax
