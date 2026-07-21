"""Data acquisition and feature engineering.

Rolling features are computed **in SQL with DuckDB window functions**, not with
pandas ``.rolling()``. This is deliberate: the window frames live in one
reviewable SQL statement (:data:`FEATURE_SQL`), every frame is explicitly
causal (``ROWS BETWEEN n PRECEDING AND CURRENT ROW`` or ``UNBOUNDED PRECEDING
AND CURRENT ROW``), and the same query serves every market configuration.

The project runs on two market configurations (see :data:`MARKETS`): each pairs
an equity index with the implied-volatility index derived from options on that
same index (S&P 500 / VIX; EURO STOXX 50 / VSTOXX). US and European calendars
differ, so the two configurations are never joined to each other on date.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

__all__ = [
    "MARKETS",
    "FEATURE_SQL",
    "DEFAULT_START",
    "download_prices",
    "load_vol_index_from_file",
    "validate_vol_coverage",
    "build_database",
    "engineer_features",
    "load_returns",
]

logger = logging.getLogger(__name__)

DEFAULT_START = "2000-01-01"

# Each market pairs an equity index with the implied-volatility index derived
# from options on that same equity index, so the pairing is not arbitrary.
MARKETS: dict[str, dict[str, str]] = {
    "us": {"index": "^GSPC", "vol_index": "^VIX"},  # S&P 500 and the VIX (default)
    "europe": {"index": "^STOXX50E", "vol_index": "V2TX.DE"},  # EURO STOXX 50 / VSTOXX
}

# ---------------------------------------------------------------------------
# Feature query. Rolling features are computed here with DuckDB window
# functions rather than pandas. Every frame below is ROWS BETWEEN n PRECEDING
# AND CURRENT ROW or UNBOUNDED PRECEDING AND CURRENT ROW -- strictly causal,
# using only information available up to and including the current row. A
# future-looking frame (UNBOUNDED FOLLOWING, or a centred window) would be
# look-ahead bias and must never be used here. The query is market-agnostic:
# the equity-index and volatility-index tickers are bound as parameters, and
# only the (structural) window widths are formatted in.
# ---------------------------------------------------------------------------
FEATURE_SQL = """
WITH idx AS (
    SELECT date, close
    FROM prices
    WHERE ticker = ?
),
volidx AS (
    SELECT
        date,
        close AS vol_index,
        close - lag(close) OVER (ORDER BY date) AS vol_index_change
    FROM prices
    WHERE ticker = ?
),
ret AS (
    SELECT
        date,
        close,
        ln(close / lag(close) OVER (ORDER BY date)) AS log_return
    FROM idx
),
rolled AS (
    SELECT
        date,
        close,
        log_return,
        stddev_samp(log_return) OVER (
            ORDER BY date
            ROWS BETWEEN {vol_frame} PRECEDING AND CURRENT ROW
        ) * sqrt(252) AS realised_vol,
        close / max(close) OVER (
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) - 1 AS drawdown
    FROM ret
),
zscored AS (
    SELECT
        date,
        close,
        log_return,
        realised_vol,
        (realised_vol - avg(realised_vol) OVER w_z)
            / stddev_samp(realised_vol) OVER w_z AS vol_z,
        drawdown
    FROM rolled
    WINDOW w_z AS (
        ORDER BY date
        ROWS BETWEEN {z_frame} PRECEDING AND CURRENT ROW
    )
)
SELECT
    z.date,
    z.close,
    z.log_return,
    z.realised_vol,
    z.vol_z,
    z.drawdown,
    v.vol_index,
    v.vol_index_change
FROM zscored z
LEFT JOIN volidx v USING (date)
ORDER BY z.date
"""


# ---------------------------------------------------------------------------
# download + cache
# ---------------------------------------------------------------------------
def _safe(name: str) -> str:
    """Filesystem-safe token for a ticker symbol."""
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def _cache_path(cache_dir: str | Path, ticker: str, start: str, end: str) -> Path:
    return Path(cache_dir) / f"{_safe(ticker)}__{start}__{end}.parquet"


def _download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download a single ticker's adjusted close via yfinance."""
    import yfinance as yf

    raw = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False
    )
    if raw.empty:
        raise ValueError(f"no data returned for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw.reset_index()[["Date", "Close"]].rename(
        columns={"Date": "date", "Close": "close"}
    )
    out["ticker"] = ticker
    out["date"] = pd.to_datetime(out["date"])
    return out[["date", "ticker", "close"]]


def download_prices(
    tickers: str | list[str],
    start: str = DEFAULT_START,
    end: str | None = None,
    cache_dir: str | Path = "data",
) -> pd.DataFrame:
    """Download adjusted-close prices, caching each series to parquet.

    Each ticker is cached to a parquet file keyed by ticker and date range, so a
    second call reads the cache instead of re-downloading. If a download fails
    and a cache for that ticker exists it is used; a clear error is raised only
    when neither the network nor a cache can supply the data.

    When both an equity index and its paired volatility index (per
    :data:`MARKETS`) are requested, the volatility series' coverage is validated
    against the equity index's trading days (see :func:`validate_vol_coverage`).

    Parameters
    ----------
    tickers : str or list of str
        One or more Yahoo Finance symbols.
    start : str, optional
        Start date ``"YYYY-MM-DD"`` (inclusive). Defaults to ``"2000-01-01"``.
    end : str, optional
        End date ``"YYYY-MM-DD"`` (exclusive). Defaults to today.
    cache_dir : str or pathlib.Path, optional
        Directory for the parquet cache.

    Returns
    -------
    pandas.DataFrame
        Long frame with columns ``date``, ``ticker``, ``close``.

    Notes
    -----
    Requires network access on a cache miss. The live path is exercised only by
    the ``network``-marked test.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        cache = _cache_path(cache_dir, ticker, start, end)
        if cache.exists():
            logger.info("cache hit for %s (%s)", ticker, cache.name)
            frames.append(pd.read_parquet(cache))
            continue
        try:
            frame = _download_one(ticker, start, end)
        except Exception as exc:  # noqa: BLE001 - fall back to any cache
            fallback = sorted(Path(cache_dir).glob(f"{_safe(ticker)}__*.parquet"))
            if fallback:
                logger.warning(
                    "download failed for %s (%s); using cache %s",
                    ticker,
                    exc,
                    fallback[-1].name,
                )
                frames.append(pd.read_parquet(fallback[-1]))
                continue
            raise RuntimeError(
                f"could not download {ticker} and no cache is available"
            ) from exc
        frame.to_parquet(cache)
        frames.append(frame)

    prices = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    _validate_market_pairs(prices)
    return prices


def load_vol_index_from_file(
    path: str | Path,
    value_column: str | int | None = None,
    decimal: str = ".",
) -> pd.DataFrame:
    """Load a volatility index series from a semicolon-delimited history file.

    A documented fallback for when Yahoo's volatility series is unusable (its
    coverage of ``V2TX.DE`` is unreliable). STOXX publishes VSTOXX history back
    to 1999 as a semicolon-delimited file with the date in the first column.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``;``-delimited file. The first column is the date.
    value_column : str, int, or None, optional
        Column holding the index level. If ``None``, the last numeric column is
        used.
    decimal : str, optional
        Decimal separator (``"."`` or ``","``).

    Returns
    -------
    pandas.DataFrame
        Columns ``date`` and ``vol_index``, sorted ascending by date.
    """
    frame = pd.read_csv(path, sep=";", decimal=decimal)
    if frame.shape[1] < 2:
        raise ValueError("file must have a date column and at least one value column")

    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", dayfirst=True)
    frame = frame.dropna(subset=["date"])

    if value_column is None:
        numeric = frame.select_dtypes(include="number").columns
        if len(numeric) == 0:
            raise ValueError("no numeric value column found")
        value_column = numeric[-1]
    elif isinstance(value_column, int):
        value_column = frame.columns[value_column]

    out = frame[["date", value_column]].rename(columns={value_column: "vol_index"})
    return out.sort_values("date").reset_index(drop=True)


def validate_vol_coverage(
    index_dates: object,
    vol_dates: object,
    min_coverage: float = 0.8,
    name: str = "volatility index",
) -> float:
    """Validate a volatility series' coverage against the equity calendar.

    Logs the number of missing trading days and raises if coverage falls below
    ``min_coverage``, rather than silently returning a sparse column.

    Parameters
    ----------
    index_dates : iterable
        Trading days of the equity index.
    vol_dates : iterable
        Trading days on which the volatility index has a value.
    min_coverage : float, optional
        Minimum acceptable fraction of equity trading days covered.
    name : str, optional
        Name used in log and error messages.

    Returns
    -------
    float
        The coverage fraction in ``[0, 1]``.
    """
    index_set = set(index_dates)
    vol_set = set(vol_dates)
    n_index = len(index_set)
    if n_index == 0:
        return 1.0
    covered = len(index_set & vol_set)
    missing = n_index - covered
    coverage = covered / n_index
    logger.info(
        "%s: %d/%d equity trading days covered, %d missing (%.1f%%)",
        name,
        covered,
        n_index,
        missing,
        100.0 * coverage,
    )
    if coverage < min_coverage:
        raise ValueError(
            f"{name} covers only {coverage:.1%} of the equity index trading days "
            f"(< {min_coverage:.0%}); series is too sparse to use"
        )
    return coverage


def _validate_market_pairs(prices: pd.DataFrame) -> None:
    """Coverage-check any equity/volatility pair fully present in ``prices``."""
    present = set(prices["ticker"].unique())
    for cfg in MARKETS.values():
        idx_t, vol_t = cfg["index"], cfg["vol_index"]
        if idx_t in present and vol_t in present:
            idx_dates = prices.loc[prices["ticker"] == idx_t, "date"]
            vol_dates = prices.loc[
                (prices["ticker"] == vol_t) & prices["close"].notna(), "date"
            ]
            validate_vol_coverage(idx_dates, vol_dates, name=vol_t)


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------
_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS prices (
        date   DATE,
        ticker VARCHAR,
        close  DOUBLE,
        PRIMARY KEY (date, ticker)
    );
"""


def build_database(prices: pd.DataFrame, db_path: str | Path = "data/market.duckdb") -> None:
    """Write a long price frame into a DuckDB ``prices`` table.

    The table is ``prices(date DATE, ticker VARCHAR, close DOUBLE)`` with a
    primary key on ``(date, ticker)``. Existing rows with the same key are
    replaced, so the call is idempotent.

    Parameters
    ----------
    prices : pandas.DataFrame
        Frame with columns ``date``, ``ticker``, ``close``.
    db_path : str or pathlib.Path, optional
        DuckDB database file to create or update.
    """
    required = {"date", "ticker", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing columns: {sorted(missing)}")

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(_CREATE_TABLE)
        con.register("_incoming", prices[["date", "ticker", "close"]])
        # upsert: drop overlapping keys, then insert
        con.execute(
            """
            DELETE FROM prices
            USING _incoming inc
            WHERE prices.date = inc.date::DATE AND prices.ticker = inc.ticker;
            """
        )
        con.execute(
            "INSERT INTO prices SELECT date::DATE, ticker, close FROM _incoming;"
        )
        con.unregister("_incoming")
    finally:
        con.close()


# ---------------------------------------------------------------------------
# feature engineering
# ---------------------------------------------------------------------------
def engineer_features(
    db_path: str | Path,
    market: str = "us",
    vol_window: int = 21,
    z_window: int = 252,
    forward_fill_vol: bool = False,
) -> pd.DataFrame:
    """Compute causal features for a market via the DuckDB window-function query.

    Resolves the equity/volatility ticker pair through :data:`MARKETS` and runs
    :data:`FEATURE_SQL`. All rolling features are causal (trailing frames only).
    The volatility column is left ``NULL`` on calendar gaps and its missing
    count is logged; it is **not** forward-filled by default, since that would
    fabricate observations. Forward-fill is opt-in and strictly backward-looking.

    Parameters
    ----------
    db_path : str or pathlib.Path
        DuckDB database built by :func:`build_database`.
    market : str, optional
        Market key in :data:`MARKETS` (``"us"`` or ``"europe"``).
    vol_window : int, optional
        Look-back length (rows) for realised volatility.
    z_window : int, optional
        Look-back length (rows) for the volatility z-score.
    forward_fill_vol : bool, optional
        If ``True``, forward-fill the joined volatility columns (strictly
        backward-looking). Off by default.

    Returns
    -------
    pandas.DataFrame
        Features indexed by ``date`` with warm-up ``NaN`` rows dropped. Columns:
        ``close``, ``log_return``, ``realised_vol``, ``vol_z``, ``drawdown``,
        ``vol_index``, ``vol_index_change``.
    """
    if market not in MARKETS:
        raise ValueError(f"unknown market {market!r}; choose from {list(MARKETS)}")
    cfg = MARKETS[market]

    sql = FEATURE_SQL.format(vol_frame=vol_window - 1, z_frame=z_window - 1)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        frame = con.execute(sql, [cfg["index"], cfg["vol_index"]]).df()
    finally:
        con.close()

    n_missing = int(frame["vol_index"].isna().sum())
    logger.info(
        "market=%s vol index %s: %d missing trading days after the calendar join",
        market,
        cfg["vol_index"],
        n_missing,
    )
    if forward_fill_vol:
        # Opt-in, strictly backward-looking fill (past values only).
        frame["vol_index"] = frame["vol_index"].ffill()
        frame["vol_index_change"] = frame["vol_index_change"].ffill()

    # Drop warm-up rows where the trailing windows have no value yet. The
    # volatility-index column is intentionally left NaN on gaps and preserved.
    frame = frame.dropna(subset=["log_return", "realised_vol", "vol_z"])
    return frame.set_index("date")


def load_returns(
    db_path: str | Path,
    market: str = "us",
    vol_window: int = 21,
    z_window: int = 252,
) -> np.ndarray:
    """Convenience wrapper returning the ``(T,)`` return array the model consumes.

    Parameters
    ----------
    db_path : str or pathlib.Path
        DuckDB database built by :func:`build_database`.
    market : str, optional
        Market key in :data:`MARKETS`.
    vol_window, z_window : int, optional
        Feature window lengths passed through to :func:`engineer_features`.

    Returns
    -------
    numpy.ndarray
        Log-return series, shape ``(T,)``, float64.
    """
    feats = engineer_features(db_path, market, vol_window, z_window)
    return feats["log_return"].to_numpy(dtype=np.float64)
