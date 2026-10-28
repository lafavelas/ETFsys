"""State handed to portfolio construction and risk rules on each date."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import pandas as pd

from .data import MarketData


@dataclass
class Context:
    t: pd.Timestamp
    data: MarketData                 # FULL data; use the truncated views below
    weights: pd.Series               # current (drifted) weights before today's trade
    equity: float                    # portfolio value before today's trade
    peak_equity: float
    entry_price: dict = field(default_factory=dict)  # ticker -> price when position opened

    @property
    def tickers(self) -> list[str]:
        return self.data.tickers

    @cached_property
    def history(self) -> MarketData:
        return self.data.truncate(self.t)

    @cached_property
    def returns(self) -> pd.DataFrame:
        return self.history.close.pct_change(fill_method=None).iloc[1:]

    @property
    def price(self) -> pd.Series:
        return self.data.close.loc[self.t]

    @property
    def drawdown(self) -> float:
        return self.equity / self.peak_equity - 1.0
