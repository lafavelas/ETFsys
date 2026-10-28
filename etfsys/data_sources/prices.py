"""ETF prices from yfinance: raw download, then adjusted and cleaned OHLCV.

yfinance is not survivorship-free: it only knows funds that still trade. Fine for a
fixed list of large ETFs; flag it before any broad-universe study.
"""

from __future__ import annotations

import pandas as pd

FIELDS = ["open", "high", "low", "close", "adj_close", "volume"]


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Long table as downloaded: date, ticker, open, high, low, close, adj_close, volume.

    Prices are unadjusted except adj_close (splits and dividends). Rows before a
    fund's launch come back empty and are dropped.
    """
    import yfinance as yf

    wide = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if wide.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}")
    return wide_to_long(wide)


def wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """yfinance (field, ticker) columns -> one row per date and ticker."""
    long = wide.stack(level=1).reset_index()
    long.columns = [str(c).lower().replace(" ", "_") for c in long.columns]
    long = long.rename(columns={long.columns[0]: "date", long.columns[1]: "ticker"})
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    long = long.dropna(subset=["open", "high", "low", "close", "adj_close"], how="all")
    return long[["date", "ticker", *FIELDS]].sort_values(["ticker", "date"]).reset_index(drop=True)


def clean_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Adjusted OHLCV and a report of what was changed.

    close = adjusted close (total return), and open/high/low get the same daily
    adjustment factor so intraday legs stay consistent with it. Bars whose high/low
    do not bracket open and close (vendor rounding) are widened, and counted.
    """
    df = raw.dropna(subset=["open", "high", "low", "close", "adj_close"]).copy()
    dropped = len(raw) - len(df)
    factor = df["adj_close"] / df["close"]
    out = pd.DataFrame(
        {
            "date": df["date"],
            "ticker": df["ticker"],
            "open": df["open"] * factor,
            "high": df["high"] * factor,
            "low": df["low"] * factor,
            "close": df["adj_close"],
            "volume": df["volume"].fillna(0).astype("int64"),
        }
    )
    hi = out[["open", "high", "close"]].max(axis=1)
    lo = out[["open", "low", "close"]].min(axis=1)
    fixed = int(((hi > out["high"]) | (lo < out["low"])).sum())
    out["high"], out["low"] = hi, lo
    report = {"rows": len(out), "dropped_incomplete_rows": dropped, "widened_bars": fixed}
    return out.reset_index(drop=True), report


def to_wide(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """field -> date x ticker on the union calendar. NaN before a fund's first bar."""
    tables = {}
    for field in ("open", "high", "low", "close", "volume"):
        t = prices.pivot(index="date", columns="ticker", values=field).sort_index()
        t.columns.name = None
        tables[field] = t.astype(float)
    return tables
