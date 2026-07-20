"""Tests for the DuckDB store and SQL feature engineering.

The network-dependent download path is marked ``network`` and deselected by
default; the causal feature logic is tested on synthetic in-memory prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_switching.data import PriceStore, download_prices, engineer_features


def _synthetic_prices(n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=n)
    rets = rng.normal(0.0003, 0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "TEST",
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        }
    )


def test_write_read_roundtrip():
    frame = _synthetic_prices(50)
    with PriceStore(":memory:") as store:
        n = store.write_prices(frame)
        assert n == 50
        back = store.read_prices("TEST")
        assert len(back) == 50
        assert np.allclose(back["close"].to_numpy(), frame["close"].to_numpy())


def test_write_is_idempotent_upsert():
    frame = _synthetic_prices(30)
    with PriceStore(":memory:") as store:
        store.write_prices(frame)
        store.write_prices(frame)  # second write should not duplicate
        back = store.read_prices("TEST")
        assert len(back) == 30


def test_features_are_causal():
    frame = _synthetic_prices(120)
    with PriceStore(":memory:") as store:
        store.write_prices(frame)
        feats = engineer_features(store.connection, "TEST", vol_window=10, mom_window=10)

    # log returns match a manual causal computation
    close = frame["close"].to_numpy()
    manual_lr = np.diff(np.log(close))
    assert np.allclose(feats["log_return"].to_numpy(), manual_lr, atol=1e-12)

    # rolling vol at position i uses only returns up to i (causal): recompute
    # the last row's vol from the trailing window and compare.
    lr = feats["log_return"].to_numpy()
    window = 10
    last_window = lr[-window:]
    manual_vol = np.std(last_window, ddof=1)
    assert np.isclose(feats["roll_vol"].to_numpy()[-1], manual_vol, atol=1e-10)


def test_features_no_lookahead_on_truncation():
    """Feature at row t must not change when future rows are removed."""
    frame = _synthetic_prices(120)
    with PriceStore(":memory:") as full, PriceStore(":memory:") as truncated:
        full.write_prices(frame)
        truncated.write_prices(frame.iloc[:80].copy())
        f_full = engineer_features(full.connection, "TEST", vol_window=10, mom_window=10)
        f_trunc = engineer_features(
            truncated.connection, "TEST", vol_window=10, mom_window=10
        )
    # First 79 feature rows (row 0 dropped) must be identical.
    common = min(len(f_trunc), len(f_full))
    assert np.allclose(
        f_full["roll_vol"].to_numpy()[:common],
        f_trunc["roll_vol"].to_numpy()[:common],
        atol=1e-12,
        equal_nan=True,
    )


@pytest.mark.network
def test_download_prices_live():
    frame = download_prices("SPY", "2023-01-01", "2023-02-01")
    assert not frame.empty
    assert {"date", "ticker", "close"}.issubset(frame.columns)
