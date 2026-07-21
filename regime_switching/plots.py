"""Figures for regime-switching analysis (matplotlib only, no seaborn).

Every function accepts an optional ``ax`` and returns the Matplotlib ``Figure``
it drew on, so plots compose into larger layouts or stand alone in a README. A
single colour scheme is used throughout: the low-variance ("calm") regime is
blue and the high-variance ("turbulent") regime is red, with intermediate
states interpolated between them.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .chain import expected_duration

__all__ = [
    "plot_regimes",
    "plot_transition_matrix",
    "plot_regime_distributions",
    "plot_loglik_traces",
    "plot_equity_curves",
    "plot_filtered_vs_smoothed",
]

COLOR_CALM = "#2166ac"        # low-variance regime (state 0)
COLOR_TURBULENT = "#b2182b"   # high-variance regime (last state)
COLOR_STRATEGY = "#1b7837"
COLOR_BENCH = ["#762a83", "#e08214", "#4d4d4d", "#35978f"]


def _new_fig(ax, figsize):
    """Return ``(fig, ax)``, creating them if ``ax`` is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.get_figure(), ax


def _regime_color(k: int, n_states: int) -> tuple:
    """Colour for state ``k``: blue (calm) -> red (turbulent)."""
    if n_states <= 1:
        return plt.matplotlib.colors.to_rgba(COLOR_CALM)
    frac = k / (n_states - 1)
    c0 = np.array(plt.matplotlib.colors.to_rgba(COLOR_CALM))
    c1 = np.array(plt.matplotlib.colors.to_rgba(COLOR_TURBULENT))
    return tuple(c0 + frac * (c1 - c0))


def _as_high_var_prob(probs: np.ndarray) -> np.ndarray:
    """Extract P(high-variance regime): last column of a 2-D array, else 1-D."""
    probs = np.asarray(probs, dtype=np.float64)
    return probs[:, -1] if probs.ndim == 2 else probs.ravel()


def plot_regimes(prices, filtered_probs, dates=None, ax=None):
    """Price series with the high-variance regime shaded by filtered probability.

    The shading is a red overlay whose **opacity is proportional to the filtered
    probability** of the high-variance regime, so turbulent periods read darker.

    Parameters
    ----------
    prices : numpy.ndarray
        Price level, shape ``(T,)``.
    filtered_probs : numpy.ndarray
        Filtered probabilities, shape ``(T, K)`` (the last column, highest
        variance, is used) or ``(T,)`` interpreted as P(high-variance).
    dates : numpy.ndarray, optional
        X-axis labels, shape ``(T,)``. Defaults to an integer index.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _new_fig(ax, (11, 4))
    prices = np.asarray(prices, dtype=np.float64)
    t = prices.shape[0]
    x = np.arange(t) if dates is None else np.asarray(dates)
    p_high = np.clip(_as_high_var_prob(filtered_probs), 0.0, 1.0)

    # RGBA background band: red with per-column alpha == P(high-variance).
    rgba = np.zeros((1, t, 4), dtype=np.float64)
    base = plt.matplotlib.colors.to_rgba(COLOR_TURBULENT)
    rgba[0, :, 0], rgba[0, :, 1], rgba[0, :, 2] = base[0], base[1], base[2]
    rgba[0, :, 3] = p_high * 0.75
    x_num = plt.matplotlib.dates.date2num(x) if not np.issubdtype(
        np.asarray(x).dtype, np.number
    ) else np.asarray(x, dtype=float)
    ax.imshow(
        rgba,
        aspect="auto",
        origin="lower",
        extent=[x_num[0], x_num[-1], float(prices.min()), float(prices.max())],
        zorder=0,
    )
    ax.plot(x, prices, color="black", linewidth=1.0, zorder=2)
    ax.set_ylabel("price")
    ax.set_title("Price with high-variance regime shaded by filtered probability")

    # a thin proxy colourbar for the opacity scale
    sm = plt.cm.ScalarMappable(cmap=_turbulent_cmap(), norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="P(high-variance)")
    return fig


def _turbulent_cmap():
    """A white->turbulent-red colormap for the P(high-variance) colourbar."""
    return plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "turbulent", ["white", COLOR_TURBULENT]
    )


def plot_transition_matrix(P, ax=None):
    """Annotated transition-matrix heatmap with expected durations in the labels.

    Parameters
    ----------
    P : numpy.ndarray
        Transition matrix, shape ``(K, K)``.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _new_fig(ax, (5.5, 4.5))
    P = np.asarray(P, dtype=np.float64)
    k = P.shape[0]
    durations = expected_duration(P)

    im = ax.imshow(P, cmap="Blues", vmin=0.0, vmax=1.0)
    labels = [f"S{i}\n(dur {durations[i]:.0f}d)" for i in range(k)]
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([f"S{j}" for j in range(k)])
    ax.set_yticklabels(labels)
    ax.set_xlabel("to state j")
    ax.set_ylabel("from state i (expected duration)")
    ax.set_title("Transition matrix  P(S_t = j | S_{t-1} = i)")
    for i in range(k):
        for j in range(k):
            ax.text(
                j, i, f"{P[i, j]:.2f}",
                ha="center", va="center",
                color="white" if P[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def plot_regime_distributions(returns, states, ax=None):
    """Return histograms by regime with the fitted normal densities overlaid.

    Parameters
    ----------
    returns : numpy.ndarray
        Per-period returns, shape ``(T,)``.
    states : numpy.ndarray
        Integer regime label per period, shape ``(T,)``.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _new_fig(ax, (8, 4.5))
    returns = np.asarray(returns, dtype=np.float64)
    states = np.asarray(states)
    unique = np.unique(states[~np.isnan(states)]) if states.dtype.kind == "f" else np.unique(states)
    n_states = len(unique)
    grid = np.linspace(returns.min(), returns.max(), 400)

    for idx, s in enumerate(unique):
        s_int = int(s)
        vals = returns[states == s]
        if vals.size == 0:
            continue
        color = _regime_color(idx, n_states)
        ax.hist(vals, bins=60, density=True, alpha=0.4, color=color, label=f"state {s_int}")
        mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        if sd > 0:
            pdf = np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            ax.plot(grid, pdf, color=color, linewidth=2.0)
    ax.set_xlabel("return")
    ax.set_ylabel("density")
    ax.set_title("Return distribution by regime (fitted normals overlaid)")
    ax.legend()
    return fig


def plot_loglik_traces(all_traces, ax=None):
    """Every multi-start EM run's convergence, with the best run highlighted.

    Makes the local-optimum problem visible: scattered ending log-likelihoods
    mean EM found different optima from different starts.

    Parameters
    ----------
    all_traces : list of list of float
        Per-start log-likelihood traces (one list per initialisation).
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _new_fig(ax, (8, 4.5))
    if not all_traces:
        return fig
    finals = [t[-1] if len(t) else -np.inf for t in all_traces]
    best = int(np.argmax(finals))
    for i, trace in enumerate(all_traces):
        if i == best:
            continue
        ax.plot(range(1, len(trace) + 1), trace, color="0.7", linewidth=1.0, zorder=1)
    best_trace = all_traces[best]
    ax.plot(
        range(1, len(best_trace) + 1), best_trace,
        color=COLOR_STRATEGY, linewidth=2.5, zorder=3,
        label=f"best start (loglik {finals[best]:.1f})",
    )
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("log-likelihood")
    ax.set_title(f"Multi-start EM convergence ({len(all_traces)} starts)")
    ax.legend(loc="lower right")
    return fig


def plot_equity_curves(results, dates=None, ax=None):
    """Strategy vs benchmarks on a log scale, with a drawdown panel beneath.

    Parameters
    ----------
    results : dict of str to numpy.ndarray
        Mapping label -> per-period return series ``(T,)``. Each is compounded
        into an equity curve starting at 1.
    dates : numpy.ndarray, optional
        Shared x-axis labels.
    ax : matplotlib.axes.Axes, optional
        Equity axis. A drawdown axis is appended beneath it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _new_fig(ax, (11, 6))
    divider = make_axes_locatable(ax)
    dax = divider.append_axes("bottom", size="35%", pad=0.35, sharex=ax)

    for i, (label, returns) in enumerate(results.items()):
        r = np.asarray(returns, dtype=np.float64)
        equity = np.cumprod(1.0 + r)
        x = np.arange(equity.shape[0]) if dates is None else np.asarray(dates)
        color = COLOR_STRATEGY if i == 0 else COLOR_BENCH[(i - 1) % len(COLOR_BENCH)]
        lw = 2.2 if i == 0 else 1.4
        ax.plot(x, equity, label=label, color=color, linewidth=lw)
        drawdown = equity / np.maximum.accumulate(equity) - 1.0
        dax.plot(x, drawdown, color=color, linewidth=lw)

    ax.set_yscale("log")
    ax.set_ylabel("equity (start = 1, log)")
    ax.set_title("Cumulative performance")
    ax.legend(loc="upper left")
    ax.tick_params(labelbottom=False)
    dax.set_ylabel("drawdown")
    dax.axhline(0.0, color="black", linewidth=0.6)
    return fig


def plot_filtered_vs_smoothed(filtered, smoothed, dates=None, ax=None):
    """Filtered vs smoothed regime probability, annotated where they diverge.

    This is the visual argument for using **filtered** probabilities in the
    backtest: the smoothed estimate conditions on the whole sample (including the
    future) and disagrees with the filtered estimate exactly at regime turns —
    so trading on it would be look-ahead bias. Shaded bands mark where the two
    disagree most.

    Parameters
    ----------
    filtered : numpy.ndarray
        Filtered probability of a chosen state, shape ``(T,)`` or ``(T, K)``
        (last column used).
    smoothed : numpy.ndarray
        Smoothed probability of the same state, same shape.
    dates : numpy.ndarray, optional
        X-axis labels.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _new_fig(ax, (11, 4.5))
    f = _as_high_var_prob(filtered)
    s = _as_high_var_prob(smoothed)
    t = f.shape[0]
    x = np.arange(t) if dates is None else np.asarray(dates)

    ax.plot(x, s, color="#4575b4", linewidth=1.6, label="smoothed  P(S_t | all data)")
    ax.plot(x, f, color="#d73027", linewidth=1.4, label="filtered  P(S_t | data <= t)")

    diff = np.abs(f - s)
    thresh = max(0.15, float(np.nanpercentile(diff, 95)))
    diverge = diff >= thresh
    ax.fill_between(
        x, 0, 1, where=diverge, color="0.6", alpha=0.25, step="mid",
        label="largest divergence",
    )
    if diverge.any():
        j = int(np.nanargmax(diff))
        ax.annotate(
            "filtered and smoothed disagree at regime turns\n"
            "(smoothed peeks at the future — unusable live)",
            xy=(x[j], f[j]),
            xytext=(0.02, 0.08),
            textcoords="axes fraction",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"),
        )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("P(regime)")
    ax.set_title("Filtered vs smoothed regime probability")
    ax.legend(loc="upper right", fontsize=9)
    return fig
