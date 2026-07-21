"""One reproducible, end-to-end regime-switching run for a single market.

Pipeline:
  1. acquire and cache prices, build a DuckDB store, engineer features in SQL
  2. fit the HMM on the full sample and print its summary
  3. Markov-chain analytics on the fitted transition matrix
  4. model selection: BIC for K in 2..4
  5. walk-forward backtest with benchmarks and the comparison table
  6. write every figure to figures/{market}/ and results to results/{market}_summary.csv
  7. print a short plain-text report

The run is fully deterministic given the seed: running it twice produces
identical numbers (see tests/test_run_analysis.py).

Yahoo Finance is the default data source. When it is unreachable (or for a
fully offline, reproducible run) pass ``--synthetic`` to generate a deterministic
two-regime series instead; the rest of the pipeline is identical.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from regime_switching.backtest import (  # noqa: E402
    benchmarks,
    compare,
    regime_strategy,
    walk_forward_regimes,
)
from regime_switching.chain import (  # noqa: E402
    expected_duration,
    expected_return_time,
    relaxation_time,
    second_eigenvalue_modulus,
    stationary_distribution,
)
from regime_switching.data import (  # noqa: E402
    MARKETS,
    build_database,
    download_prices,
    engineer_features,
)
from regime_switching.hmm import GaussianHMM  # noqa: E402
from regime_switching.plots import (  # noqa: E402
    plot_equity_curves,
    plot_filtered_vs_smoothed,
    plot_loglik_traces,
    plot_regime_distributions,
    plot_regimes,
    plot_transition_matrix,
)

logger = logging.getLogger("run_analysis")


# --------------------------------------------------------------------------- #
# data acquisition
# --------------------------------------------------------------------------- #
def build_synthetic_prices(market: str, dates: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Deterministic two-regime synthetic prices for an offline/reproducible run.

    Generates an equity index from a known 2-state Gaussian HMM and a paired
    volatility-index series that is elevated in the turbulent regime.
    """
    cfg = MARKETS[market]
    rng = np.random.default_rng(seed)
    n = len(dates)
    means = np.array([0.0005, -0.0007])
    sds = np.array([0.007, 0.022])
    transmat = np.array([[0.985, 0.015], [0.040, 0.960]])

    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(2, p=transmat[states[t - 1]])
    rets = rng.normal(means[states], sds[states])
    close = 1000.0 * np.cumprod(1.0 + rets)
    vol = 12.0 + 8.0 * states + rng.uniform(0.0, 2.0, size=n)  # calm ~12, turbulent ~20

    idx = pd.DataFrame({"date": dates, "ticker": cfg["index"], "close": close})
    volser = pd.DataFrame({"date": dates, "ticker": cfg["vol_index"], "close": vol})
    return pd.concat([idx, volser], ignore_index=True)


def acquire_data(
    market: str,
    start: str,
    end: str,
    cache_dir: str | Path,
    db_path: str | Path,
    synthetic: bool,
    seed: int,
) -> str:
    """Acquire prices (download or synthetic), build the DuckDB store."""
    cfg = MARKETS[market]
    if synthetic:
        dates = pd.bdate_range(start, end)
        prices = build_synthetic_prices(market, dates, seed)
        logger.info("using synthetic data (%d rows, %d dates)", len(prices), len(dates))
    else:
        prices = download_prices(
            [cfg["index"], cfg["vol_index"]], start, end, cache_dir=cache_dir
        )
    build_database(prices, db_path)
    return str(db_path)


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(
    db_path: str | Path,
    market: str,
    n_states: int = 2,
    seed: int = 20260721,
    outdir: str | Path = ".",
    *,
    vol_window: int = 21,
    z_window: int = 252,
    n_init: int = 10,
    train_window: int = 756,
    refit_every: int = 63,
    bic_k: tuple[int, ...] = (2, 3, 4),
    periods_per_year: int = 252,
    make_figures: bool = True,
) -> dict:
    """Run steps 2-7 on an already-built database. Deterministic given ``seed``.

    Returns
    -------
    dict
        ``summary_table`` (the strategy-vs-benchmark comparison), ``bic`` (dict
        K -> BIC), ``chain`` (analytics dict), and ``model`` (the fitted HMM).
    """
    feats = engineer_features(db_path, market=market, vol_window=vol_window, z_window=z_window)
    log_returns = feats["log_return"].to_numpy()
    dates = feats.index.to_numpy()
    prices = feats["close"].to_numpy()

    # 2. full-sample fit
    model = GaussianHMM(n_states=n_states, n_init=n_init, random_state=seed).fit(log_returns)
    logger.info("full-sample fit:\n%s", model.summary())

    # 3. chain analytics
    P = model.transmat_
    chain_stats = {
        "stationary": stationary_distribution(P),
        "expected_duration": expected_duration(P),
        "expected_return_time": expected_return_time(P),
        "second_eigenvalue_modulus": second_eigenvalue_modulus(P),
        "relaxation_time": relaxation_time(P),
    }

    # 4. model selection via BIC
    bic = {}
    for k in bic_k:
        m_k = GaussianHMM(n_states=k, n_init=n_init, random_state=seed).fit(log_returns)
        bic[k] = m_k.bic(log_returns)
    best_k = min(bic, key=bic.get)

    # 5. walk-forward backtest
    wf = walk_forward_regimes(
        log_returns, dates, n_states=n_states, train_window=train_window,
        refit_every=refit_every, n_init=n_init, random_state=seed,
    )
    prob_cols = [f"p_state_{k}" for k in range(n_states)]
    strat = regime_strategy(feats, wf[prob_cols], mode="probability")
    bench = benchmarks(
        log_returns, feats, avg_exposure=float(strat["position"].mean())
    )
    table = compare(strat, bench, periods_per_year=periods_per_year)

    # 6. figures + results CSV
    outdir = Path(outdir)
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / f"{market}_summary.csv")

    if make_figures:
        figdir = outdir / "figures" / market
        figdir.mkdir(parents=True, exist_ok=True)
        filtered = model.filter(log_returns)
        smoothed = model.predict_proba(log_returns)
        viterbi_states = model.predict(log_returns)

        _save(plot_regimes(prices, filtered, dates), figdir / "regimes.png")
        _save(plot_transition_matrix(P), figdir / "transition_matrix.png")
        _save(
            plot_regime_distributions(log_returns, viterbi_states),
            figdir / "regime_distributions.png",
        )
        _save(plot_loglik_traces(model.all_traces_), figdir / "loglik_traces.png")
        _save(
            plot_equity_curves(
                {
                    "strategy": strat["net"].to_numpy(),
                    "buy & hold": bench["buy_and_hold"].to_numpy(),
                    "vol target": bench["vol_target"].to_numpy(),
                    "constant exposure": bench["constant_exposure"].to_numpy(),
                },
                dates=wf.index.to_numpy(),
            ),
            figdir / "equity_curves.png",
        )
        _save(
            plot_filtered_vs_smoothed(filtered[:, -1], smoothed[:, -1], dates),
            figdir / "filtered_vs_smoothed.png",
        )

    # 7. report
    report = format_report(market, feats, model, chain_stats, bic, best_k, table)
    logger.info("\n%s", report)

    return {"summary_table": table, "bic": bic, "chain": chain_stats, "model": model}


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def format_report(market, feats, model, chain_stats, bic, best_k, table) -> str:
    """A short, plain-text summary of the run."""
    lines = [
        "=" * 70,
        f"REGIME-SWITCHING REPORT — market: {market}",
        "=" * 70,
        f"sample: {len(feats)} observations, {feats.index.min()} .. {feats.index.max()}",
        "",
        "Regimes (sorted by ascending variance):",
    ]
    for i in range(model.n_states):
        lines.append(
            f"  state {i}: mean={model.means_[i]:+.5f}  "
            f"std={np.sqrt(model.variances_[i]):.5f}  "
            f"exp.duration={chain_stats['expected_duration'][i]:.0f}d  "
            f"stationary={chain_stats['stationary'][i]:.2f}"
        )
    lines += [
        "",
        f"|lambda_2| = {chain_stats['second_eigenvalue_modulus']:.4f}  "
        f"=> relaxation time = {chain_stats['relaxation_time']:.1f} days",
        "BIC by K: " + ", ".join(f"K={k}:{v:.0f}" for k, v in bic.items())
        + f"  => selects K={best_k}",
        "",
        "Walk-forward strategy vs benchmarks (net of costs):",
    ]
    for col in table.columns:
        row = table[col]
        extra = ""
        if col != "strategy" and not np.isnan(row.get("jk_p", np.nan)):
            extra = f"  JK-Memmel vs strategy: z={row['jk_z']:+.2f}, p={row['jk_p']:.3f}"
        lines.append(
            f"  {col:<18} CAGR={row['cagr']:+.2%}  Sharpe={row['sharpe']:.2f}  "
            f"maxDD={row['max_drawdown']:.1%}{extra}"
        )
    lines += [
        "",
        "NOTE: buy_and_hold understates true holding return — the equity indices",
        "are PRICE indices and exclude dividends (~1.5%/yr S&P, ~3%/yr EURO STOXX",
        "50), so the distortion is larger for the European configuration.",
        "NOTE: vol_target is the benchmark that matters; a regime model that does",
        "not beat inverse-vol scaling is not adding anything.",
        "=" * 70,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="us", choices=list(MARKETS))
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--n-states", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--synthetic", action="store_true", help="use offline synthetic data")
    parser.add_argument("--db", default=None, help="DuckDB path (default: outdir/data)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args = parse_args()

    outdir = Path(args.outdir)
    db_path = args.db or (outdir / "data" / f"{args.market}.duckdb")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    acquire_data(
        args.market, args.start, args.end,
        cache_dir=outdir / "data", db_path=db_path,
        synthetic=args.synthetic, seed=args.seed,
    )
    run_pipeline(db_path, args.market, n_states=args.n_states, seed=args.seed, outdir=outdir)


if __name__ == "__main__":
    main()
