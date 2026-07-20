"""Price download, DuckDB storage, and SQL feature engineering.

Prices are fetched with :mod:`yfinance`, persisted to a DuckDB database, and
turned into model features (log returns, rolling volatility, ...) with SQL.
Every windowed feature computed here is **causal**: window frames use only rows
up to and including the current one (``ROWS BETWEEN ... AND CURRENT ROW``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

__all__ = [
    "download_prices",
    "PriceStore",
    "engineer_features",
]

logger = logging.getLogger(__name__)


def download_prices(
    ticker: str,
    start: str,
    end: str,
    *,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download OHLCV prices for a single ticker with yfinance.

    Parameters
    ----------
    ticker : str
        Symbol to download, e.g. ``"SPY"``.
    start : str
        Start date, ``"YYYY-MM-DD"`` (inclusive).
    end : str
        End date, ``"YYYY-MM-DD"`` (exclusive, per yfinance semantics).
    interval : str, optional
        Sampling interval understood by yfinance (``"1d"``, ``"1wk"``, ...).
    auto_adjust : bool, optional
        Whether to return split/dividend-adjusted prices.

    Returns
    -------
    pandas.DataFrame
        Columns ``date``, ``ticker``, ``open``, ``high``, ``low``, ``close``,
        ``volume``. One row per period, sorted ascending by date.

    Notes
    -----
    Requires network access. Tests exercising this function are marked
    ``network`` and deselected by default.
    """
    import yfinance as yf

    logger.info("downloading %s %s..%s (%s)", ticker, start, end, interval)
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if raw.empty:
        raise ValueError(f"no data returned for {ticker}")

    # yfinance may return a MultiIndex column frame for a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    frame = raw.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    frame["ticker"] = ticker
    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    frame = frame[cols].sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


class PriceStore:
    """Thin DuckDB-backed store for OHLCV price frames.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the DuckDB database file. Use ``":memory:"`` for an ephemeral
        in-process database.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS prices (
            date    TIMESTAMP,
            ticker  VARCHAR,
            open    DOUBLE,
            high    DOUBLE,
            low     DOUBLE,
            close   DOUBLE,
            volume  DOUBLE,
            PRIMARY KEY (date, ticker)
        );
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._con = duckdb.connect(self.path)
        self._con.execute(self._SCHEMA)

    def __enter__(self) -> PriceStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._con.close()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """The live DuckDB connection."""
        return self._con

    def write_prices(self, frame: pd.DataFrame) -> int:
        """Insert or replace price rows.

        Parameters
        ----------
        frame : pandas.DataFrame
            Frame in the schema returned by :func:`download_prices`.

        Returns
        -------
        int
            Number of rows written.
        """
        required = {"date", "ticker", "open", "high", "low", "close", "volume"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"frame is missing columns: {sorted(missing)}")

        self._con.register("_incoming", frame)
        # DELETE overlapping keys then INSERT to emulate upsert semantics.
        self._con.execute(
            """
            DELETE FROM prices
            USING _incoming
            WHERE prices.date = _incoming.date
              AND prices.ticker = _incoming.ticker;
            """
        )
        self._con.execute(
            """
            INSERT INTO prices
            SELECT date, ticker, open, high, low, close, volume FROM _incoming;
            """
        )
        self._con.unregister("_incoming")
        return len(frame)

    def read_prices(self, ticker: str) -> pd.DataFrame:
        """Read all stored rows for a ticker, ascending by date.

        Parameters
        ----------
        ticker : str
            Symbol to read.

        Returns
        -------
        pandas.DataFrame
            Stored OHLCV rows for the ticker.
        """
        return self._con.execute(
            "SELECT * FROM prices WHERE ticker = ? ORDER BY date;",
            [ticker],
        ).df()


def engineer_features(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    *,
    vol_window: int = 21,
    mom_window: int = 21,
) -> pd.DataFrame:
    """Compute causal model features from stored prices via SQL.

    All window frames use ``ROWS BETWEEN (window - 1) PRECEDING AND CURRENT
    ROW`` so that every feature at row ``t`` depends only on data available up
    to and including ``t``. No feature looks into the future.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection to a store holding a ``prices`` table.
    ticker : str
        Symbol to build features for.
    vol_window : int, optional
        Look-back length for rolling volatility (in periods).
    mom_window : int, optional
        Look-back length for cumulative momentum (in periods).

    Returns
    -------
    pandas.DataFrame
        Columns ``date``, ``ticker``, ``close``, ``log_return``,
        ``roll_vol``, ``momentum``. The first row (no prior close) has a null
        ``log_return`` and is dropped.
    """
    vol_preceding = vol_window - 1
    mom_preceding = mom_window - 1
    query = f"""
        WITH base AS (
            SELECT
                date,
                ticker,
                close,
                LN(close / LAG(close) OVER w) AS log_return
            FROM prices
            WHERE ticker = ?
            WINDOW w AS (ORDER BY date)
        ),
        feat AS (
            SELECT
                date,
                ticker,
                close,
                log_return,
                -- causal rolling volatility: current row and the prior
                -- (vol_window - 1) rows only
                STDDEV_SAMP(log_return) OVER (
                    ORDER BY date
                    ROWS BETWEEN {vol_preceding} PRECEDING AND CURRENT ROW
                ) AS roll_vol,
                -- causal cumulative momentum over the trailing window
                SUM(log_return) OVER (
                    ORDER BY date
                    ROWS BETWEEN {mom_preceding} PRECEDING AND CURRENT ROW
                ) AS momentum
            FROM base
        )
        SELECT * FROM feat
        WHERE log_return IS NOT NULL
        ORDER BY date;
    """

    return con.execute(query, [ticker]).df()
