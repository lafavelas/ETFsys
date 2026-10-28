"""Copy this file to add a strategy. Nothing else in the framework needs to change.

Checklist
  1. Set `frequency` and `required_data`.
  2. Put anything learned from data in `fit` (it only ever sees data up to train_end).
  3. Return target weights from `generate`; leave sizing limits to the portfolio
     constructor and exits to the risk overlay.
  4. Expose hyperparameters as __init__ args so `params()` logs them with each run.
  5. Add it to strategies/__init__.py and run it through run_example.py.
"""
from __future__ import annotations

import pandas as pd

from ..data import MarketData
from ..strategy import Strategy


class MyStrategy(Strategy):
    frequency = "W"
    required_data = ("close",)

    def __init__(self, lookback: int = 12, name: str = "my_strategy"):
        self.lookback = lookback
        self.name = name

    def fit(self, data: MarketData, train_end: pd.Timestamp) -> None:
        pass  # e.g. estimate parameters, train a model

    def generate(self, data: MarketData, t: pd.Timestamp) -> pd.Series:
        weekly = data.resample(self.frequency).close
        if len(weekly) <= self.lookback:
            return pd.Series(dtype=float)
        trend = weekly.iloc[-1] / weekly.iloc[-1 - self.lookback] - 1
        winners = trend[trend > 0].index
        return pd.Series(1.0 / len(winners), index=winners) if len(winners) else pd.Series(dtype=float)
