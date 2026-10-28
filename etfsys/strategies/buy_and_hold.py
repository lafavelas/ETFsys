"""Benchmark plug-in: hold fixed weights. Every report compares against it."""

from __future__ import annotations

import pandas as pd

from ..data import MarketData
from ..strategy import Strategy


class BuyAndHold(Strategy):
    frequency = "M"
    required_data = ("close",)

    def __init__(self, weights: dict[str, float] | None = None, name: str = "buy_and_hold"):
        self.weights = weights or {"SPY": 1.0}
        self.name = name

    def generate(self, data: MarketData, t: pd.Timestamp) -> pd.Series:
        return pd.Series(self.weights, dtype=float)
