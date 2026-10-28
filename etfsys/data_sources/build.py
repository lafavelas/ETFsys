"""Processed Parquet -> validated `MarketData`, plus a data-quality check on SPY."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import MarketData
from .config import HOLDOUT_START, UNIVERSE
from .fred import macro_wide
from .prices import to_wide
from .schemas import MACRO_LONG, PRICES

SPY_EXPENSE_RATIO = 0.000945


def processed_dir(root: str | Path = "data") -> Path:
    return Path(root) / "processed"


def load_market_data(
    root: str | Path = "data",
    tickers: list[str] | None = None,
    unlock_holdout: bool = False,
) -> MarketData:
    """Everything `refresh` stored, validated, as one point-in-time MarketData.

    By default every table stops the day before HOLDOUT_START, so research code cannot
    see the holdout even by accident. `unlock_holdout=True` is for the final run only.
    """
    d = processed_dir(root)
    prices = PRICES.validate(pd.read_parquet(d / "prices.parquet"))
    if tickers is not None:
        prices = prices[prices["ticker"].isin(tickers)]
    tables = to_wide(prices)
    macro = macro_wide(MACRO_LONG.validate(pd.read_parquet(d / "macro.parquet")))
    close = tables["close"]
    meta = pd.DataFrame(
        {
            "category": [UNIVERSE.get(t, ("Other", "Other"))[0] for t in close.columns],
            "asset_class": [UNIVERSE.get(t, ("Other", "Other"))[1] for t in close.columns],
            "leverage": 1.0,
            "inception": [close[t].first_valid_index() for t in close.columns],
        },
        index=pd.Index(close.columns, name="ticker"),
    )
    data = MarketData(
        close=close,
        open=tables["open"],
        high=tables["high"],
        low=tables["low"],
        volume=tables["volume"],
        macro=macro,
        meta=meta,
        holdout_start=pd.Timestamp(HOLDOUT_START),
    )
    return (data if unlock_holdout else data.lock_holdout()).validate()


def load_reference(root: str | Path = "data") -> pd.DataFrame:
    """date x reference index close (never part of the tradable universe)."""
    ref = PRICES.validate(pd.read_parquet(processed_dir(root) / "reference.parquet"))
    return to_wide(ref)["close"]


def tracking_gap(etf: pd.Series, index: pd.Series, expense_ratio: float = 0.0) -> float:
    """Annual return gap of an ETF vs. its total-return index after the expense ratio.

    Near 0 means the adjusted closes carry dividends correctly. Measured on common dates.
    """
    both = pd.concat([etf, index], axis=1).dropna()
    years = (both.index[-1] - both.index[0]).days / 365.25
    growth = both.iloc[-1] / both.iloc[0]
    cagr_etf, cagr_idx = (growth ** (1 / years) - 1).to_numpy()
    return float(cagr_etf - (cagr_idx - expense_ratio)) if years > 0 else np.nan
