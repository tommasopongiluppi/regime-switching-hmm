"""Run the full pipeline for both markets and compare the regime structure.

The question this answers is empirical: does the two-regime structure **replicate**
across two independent equity markets (US: S&P 500 / VIX; Europe: EURO STOXX 50 /
VSTOXX), or is it an artefact of one sample? Agreement and disagreement are
reported equally plainly — a structure that fails to replicate is a finding, not
a failure.

Runs offline with ``--synthetic`` (the default here, since live data may be
blocked); each market is drawn independently (different seeds) from the same
generating process, so under synthetic data replication is expected *by
construction* and this script mainly demonstrates the comparison. On real data
the replication question is genuine.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

# scripts/ is on sys.path[0] when run directly, so this imports the sibling script
import run_analysis as ra  # noqa: E402

from regime_switching.data import MARKETS  # noqa: E402

logger = logging.getLogger("compare_markets")


def _run_one(market: str, args, seed: int) -> dict:
    outdir = Path(args.outdir)
    db_path = outdir / "data" / f"{market}.duckdb"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    ra.acquire_data(
        market, args.start, args.end,
        cache_dir=outdir / "data", db_path=db_path,
        synthetic=args.synthetic, seed=seed,
    )
    return ra.run_pipeline(
        db_path, market, n_states=args.n_states, seed=seed, outdir=outdir
    )


def _fmt_vec(v) -> str:
    return "[" + ", ".join(f"{x:.3f}" for x in np.asarray(v)) + "]"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--n-states", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--outdir", default=".")
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="offline synthetic data (default; live data may be blocked)",
    )
    args = parser.parse_args()

    markets = list(MARKETS)  # ["us", "europe"]
    results = {}
    for i, market in enumerate(markets):
        logger.info("running pipeline for market=%s", market)
        results[market] = _run_one(market, args, seed=args.seed + i)

    # ---- side-by-side comparison -----------------------------------------
    print("\n" + "#" * 74)
    print("# CROSS-MARKET COMPARISON — does the regime structure replicate?")
    print("#" * 74)

    for market in markets:
        m = results[market]["model"]
        c = results[market]["chain"]
        print(f"\n--- {market.upper()} ---")
        print("transition matrix:")
        for row in m.transmat_:
            print("   [" + "  ".join(f"{x:.3f}" for x in row) + "]")
        print(f"regime std      : {_fmt_vec(np.sqrt(m.variances_))}")
        print(f"expected durations (days): {_fmt_vec(c['expected_duration'])}")
        print(f"stationary dist : {_fmt_vec(c['stationary'])}")
        print(f"relaxation time : {c['relaxation_time']:.1f} days")

    # ---- performance vs benchmarks, side by side -------------------------
    print("\n--- WALK-FORWARD PERFORMANCE (Sharpe, net) ---")
    header = f"{'benchmark':<20}" + "".join(f"{mk:>12}" for mk in markets)
    print(header)
    first_cols = results[markets[0]]["summary_table"].columns
    cols = ["strategy"] + [c for c in first_cols if c != "strategy"]
    for col in cols:
        line = f"{col:<20}"
        for mk in markets:
            tbl = results[mk]["summary_table"]
            line += f"{tbl[col]['sharpe']:>12.2f}" if col in tbl.columns else f"{'-':>12}"
        print(line)

    # ---- plain-language verdict ------------------------------------------
    print("\n--- REPLICATION READOUT ---")
    a, b = markets
    ma, mb = results[a]["model"], results[b]["model"]
    ca, cb = results[a]["chain"], results[b]["chain"]

    sd_a, sd_b = np.sqrt(ma.variances_), np.sqrt(mb.variances_)
    dur_a, dur_b = ca["expected_duration"], cb["expected_duration"]

    def _agree(x, y, rel=0.30):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return bool(np.all(np.abs(x - y) <= rel * np.maximum(np.abs(x), np.abs(y))))

    vol_ratio_a = sd_a[-1] / sd_a[0]
    vol_ratio_b = sd_b[-1] / sd_b[0]
    print(
        f"  high/low volatility ratio: {a}={vol_ratio_a:.2f}, {b}={vol_ratio_b:.2f}"
        f"  -> {'AGREE' if _agree(vol_ratio_a, vol_ratio_b) else 'DISAGREE'}"
    )
    print(
        f"  regime durations:  {a}={_fmt_vec(dur_a)}, {b}={_fmt_vec(dur_b)}"
        f"  -> {'AGREE' if _agree(dur_a, dur_b) else 'DISAGREE'}"
    )
    print(
        f"  stationary weights:  {a}={_fmt_vec(ca['stationary'])}, "
        f"{b}={_fmt_vec(cb['stationary'])}"
        f"  -> {'AGREE' if _agree(ca['stationary'], cb['stationary']) else 'DISAGREE'}"
    )
    print(
        "\n  Reading: agreement across two independent samples is evidence the"
        "\n  two-regime structure is real; disagreement would mean it does not"
        "\n  replicate and is likely sample-specific. Under --synthetic both markets"
        "\n  share one generating process, so agreement here is by construction;"
        "\n  the genuine test needs live S&P 500 and EURO STOXX 50 data."
    )
    print("#" * 74)


if __name__ == "__main__":
    main()
