"""End-to-end example: fit a regime-switching model and backtest it.

This script downloads prices, engineers causal features, fits a Gaussian HMM by
multi-start Baum-Welch, runs a walk-forward backtest on the filtered
probabilities, and writes figures to ``figures/``.

Run with::

    python scripts/run_analysis.py --ticker SPY --start 2010-01-01 --end 2024-01-01

Network access is required for the price download.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from regime_switching.backtest import (  # noqa: E402
    benchmarks,
    compare,
    regime_strategy,
    walk_forward_regimes,
)
from regime_switching.chain import (  # noqa: E402
    expected_duration,
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
    plot_regime_probabilities,
    plot_transition_matrix,
)

logger = logging.getLogger("run_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="us", choices=list(MARKETS))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=756)
    parser.add_argument("--step", type=int, default=63)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--db", default="data/market.duckdb")
    parser.add_argument("--figdir", default="figures")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args = parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    # 1. download the market's equity/vol pair, store, engineer features
    tickers = [MARKETS[args.market]["index"], MARKETS[args.market]["vol_index"]]
    prices = download_prices(
        tickers, args.start, args.end, cache_dir=Path(args.db).parent
    )
    build_database(prices, args.db)
    feats = engineer_features(args.db, market=args.market)

    log_returns = feats["log_return"].to_numpy()
    dates = feats.index.to_numpy()

    # 2. fit on the full sample for interpretation (in-sample decode)
    model = GaussianHMM(
        n_states=args.states, n_init=10, random_state=args.seed
    ).fit(log_returns)
    logger.info(
        "fitted loglik=%.2f, converged=%s", model.loglik_, model.converged_
    )
    logger.info("\n%s", model.summary())

    pi = stationary_distribution(model.transmat_)
    dur = expected_duration(model.transmat_)
    logger.info("stationary distribution: %s", np.round(pi, 3))
    logger.info("expected durations (days): %s", np.round(dur, 1))

    smoothed = model.predict_proba(log_returns)  # non-causal, for plotting only

    # 3. walk-forward regime detection (causal) and strategy vs benchmarks
    wf = walk_forward_regimes(
        log_returns,
        dates,
        n_states=args.states,
        train_window=args.train_size,
        refit_every=args.step,
        random_state=args.seed,
    )
    prob_cols = [f"p_state_{k}" for k in range(args.states)]
    strat = regime_strategy(feats, wf[prob_cols], mode="probability")
    bench = benchmarks(log_returns, feats, avg_exposure=float(strat["position"].mean()))
    table = compare(strat, bench)
    logger.info("performance comparison:\n%s", table.round(4).to_string())

    # 4. figures
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_regime_probabilities(smoothed, dates=dates, ax=ax)
    fig.savefig(figdir / "regime_probabilities.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_transition_matrix(model.transmat_, ax=ax)
    fig.savefig(figdir / "transition_matrix.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(11, 4))
    plot_equity_curves(
        {
            "strategy": strat["net"].to_numpy(),
            "buy & hold": bench["buy_and_hold"].to_numpy(),
            "vol target": bench["vol_target"].to_numpy(),
        },
        dates=wf.index.to_numpy(),
        ax=ax,
    )
    fig.savefig(figdir / "equity_curves.png", dpi=120, bbox_inches="tight")
    logger.info("wrote figures to %s", figdir)


if __name__ == "__main__":
    main()
