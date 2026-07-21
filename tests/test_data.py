"""Tests for data acquisition and SQL feature engineering (no network).

The synthetic-price tests confirm that the DuckDB window-function features match
independent pandas computations and, crucially, that every window frame is
causal: a feature at row ``t`` never changes when future rows are removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_switching.data import (
    MARKETS,
    build_database,
    download_prices,
    engineer_features,
    load_returns,
)

VOL_WINDOW = 5
Z_WINDOW = 20


def _synthetic_prices(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Long price frame for the US index/vol pair, all dates aligned."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2005-01-03", periods=n)
    rets = rng.normal(0.0003, 0.011, size=n)
    close = 1000.0 * np.cumprod(1.0 + rets)
    vix = 15.0 + 5.0 * np.abs(rng.normal(size=n))  # a plausible vol series

    idx = pd.DataFrame(
        {"date": dates, "ticker": MARKETS["us"]["index"], "close": close}
    )
    vol = pd.DataFrame(
        {"date": dates, "ticker": MARKETS["us"]["vol_index"], "close": vix}
    )
    return pd.concat([idx, vol], ignore_index=True)


def _build(tmp_path, prices, name="market.duckdb"):
    db = tmp_path / name
    build_database(prices, db)
    return db


# --------------------------------------------------------------------------- #
# feature correctness vs pandas
# --------------------------------------------------------------------------- #
def test_realised_vol_matches_pandas_rolling(tmp_path):
    prices = _synthetic_prices(300)
    db = _build(tmp_path, prices)
    feats = engineer_features(db, market="us", vol_window=VOL_WINDOW, z_window=Z_WINDOW)

    close = prices.loc[
        prices["ticker"] == MARKETS["us"]["index"], ["date", "close"]
    ].set_index("date")["close"]
    log_ret = np.log(close / close.shift(1))
    # rolling features are validated against pandas .rolling here ONLY as an
    # independent check; the package itself computes them in SQL.
    pandas_rv = log_ret.rolling(VOL_WINDOW).std(ddof=1) * np.sqrt(252)

    ref = pandas_rv.loc[feats.index]
    mask = ref.notna()  # compare on full-window rows (pandas min_periods=window)
    assert mask.sum() > 0
    assert np.allclose(
        feats["realised_vol"][mask].to_numpy(),
        ref[mask].to_numpy(),
        atol=1e-10,
        rtol=0,
    )


def test_drawdown_matches_pandas_cummax(tmp_path):
    prices = _synthetic_prices(300)
    db = _build(tmp_path, prices)
    feats = engineer_features(db, market="us", vol_window=VOL_WINDOW, z_window=Z_WINDOW)

    close = prices.loc[
        prices["ticker"] == MARKETS["us"]["index"], ["date", "close"]
    ].set_index("date")["close"]
    pandas_dd = close / close.cummax() - 1.0

    ref = pandas_dd.loc[feats.index]
    assert np.allclose(
        feats["drawdown"].to_numpy(), ref.to_numpy(), atol=1e-10, rtol=0
    )


def test_vol_index_change_matches_diff(tmp_path):
    prices = _synthetic_prices(300)
    db = _build(tmp_path, prices)
    feats = engineer_features(db, market="us", vol_window=VOL_WINDOW, z_window=Z_WINDOW)

    vix = prices.loc[
        prices["ticker"] == MARKETS["us"]["vol_index"], ["date", "close"]
    ].set_index("date")["close"]
    ref = vix.diff().loc[feats.index]
    assert np.allclose(
        feats["vol_index_change"].to_numpy(), ref.to_numpy(), atol=1e-10, rtol=0
    )


# --------------------------------------------------------------------------- #
# causality guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t", [120, 180, 240, 299])
def test_features_are_causal_under_truncation(tmp_path, t):
    """A feature at row t must be bit-identical when future rows are removed.

    This proves the SQL window frames are causal: truncating the input at row t
    cannot change any feature value at rows 0..t.
    """
    prices = _synthetic_prices(300)
    full_db = _build(tmp_path, prices, name="full.duckdb")
    full = engineer_features(full_db, "us", vol_window=VOL_WINDOW, z_window=Z_WINDOW)

    # truncate BOTH series at the date of row t (inclusive)
    cutoff = pd.bdate_range("2005-01-03", periods=300)[t]
    trunc_prices = prices[prices["date"] <= cutoff].copy()
    trunc_db = _build(tmp_path, trunc_prices, name=f"trunc_{t}.duckdb")
    trunc = engineer_features(trunc_db, "us", vol_window=VOL_WINDOW, z_window=Z_WINDOW)

    common = trunc.index  # every truncated row must appear identically in full
    cols = [
        "close",
        "log_return",
        "realised_vol",
        "vol_z",
        "drawdown",
        "vol_index",
        "vol_index_change",
    ]
    for col in cols:
        np.testing.assert_array_equal(
            full.loc[common, col].to_numpy(),
            trunc[col].to_numpy(),
            err_msg=f"{col} changed under truncation at t={t}",
        )


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #
def test_build_database_is_idempotent(tmp_path):
    prices = _synthetic_prices(50)
    db = tmp_path / "m.duckdb"
    build_database(prices, db)
    build_database(prices, db)  # second write must not duplicate rows
    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    n = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    con.close()
    assert n == len(prices)


def test_load_returns_returns_1d_array(tmp_path):
    prices = _synthetic_prices(300)
    db = _build(tmp_path, prices)
    r = load_returns(db, market="us", vol_window=VOL_WINDOW, z_window=Z_WINDOW)
    feats = engineer_features(db, "us", vol_window=VOL_WINDOW, z_window=Z_WINDOW)
    assert r.ndim == 1
    assert r.shape[0] == len(feats)
    assert r.dtype == np.float64


def test_unknown_market_raises(tmp_path):
    prices = _synthetic_prices(50)
    db = _build(tmp_path, prices)
    with pytest.raises(ValueError, match="unknown market"):
        engineer_features(db, market="asia")


@pytest.mark.network
def test_download_prices_live(tmp_path):
    prices = download_prices(
        [MARKETS["us"]["index"], MARKETS["us"]["vol_index"]],
        start="2023-01-01",
        end="2023-03-01",
        cache_dir=tmp_path,
    )
    assert not prices.empty
    assert set(prices["ticker"].unique()) == {
        MARKETS["us"]["index"],
        MARKETS["us"]["vol_index"],
    }
