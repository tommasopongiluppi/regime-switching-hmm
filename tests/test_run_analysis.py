"""Determinism test for the end-to-end run_analysis pipeline.

Loads the script module directly (scripts/ is not a package) and asserts that
two runs with the same seed produce byte-identical results.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
import pytest

from regime_switching.data import build_database

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_run_analysis():
    spec = importlib.util.spec_from_file_location(
        "run_analysis", _ROOT / "scripts" / "run_analysis.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_run_analysis_is_deterministic(tmp_path):
    ra = _load_run_analysis()

    # deterministic synthetic data -> DuckDB
    dates = pd.bdate_range("2015-01-01", periods=500)
    prices = ra.build_synthetic_prices("us", dates, seed=123)
    db = tmp_path / "market.duckdb"
    build_database(prices, db)

    kw = dict(
        n_states=2, seed=7, vol_window=10, z_window=40,
        n_init=2, train_window=150, refit_every=50, bic_k=(2, 3),
        make_figures=False,
    )
    r1 = ra.run_pipeline(db, "us", outdir=tmp_path / "o1", **kw)
    r2 = ra.run_pipeline(db, "us", outdir=tmp_path / "o2", **kw)

    # identical numbers: comparison table, BIC, and the written CSV
    pd.testing.assert_frame_equal(r1["summary_table"], r2["summary_table"])
    assert r1["bic"] == r2["bic"]
    csv1 = (tmp_path / "o1" / "results" / "us_summary.csv").read_text()
    csv2 = (tmp_path / "o2" / "results" / "us_summary.csv").read_text()
    assert csv1 == csv2
