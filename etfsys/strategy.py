"""The one contract every strategy plug-in implements."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .data import MarketData


class Strategy(ABC):
    """A strategy turns point-in-time data into TARGET WEIGHTS.

    - `fit` is called by the backtest engine at each refit date with data truncated
      at `train_end`. Stateless strategies can leave it empty.
    - `generate` is called on each rebalance date t with data truncated at t and must
      return a Series of weights indexed by ticker. Missing tickers mean 0. Positive =
      long, negative = short. Portfolio construction and the risk overlay enforce caps
      afterwards, so a strategy states intent, not final sizing.

    A +1/-1 signal (Baek et al. 2020) is the special case {ticker: +1.0} or {ticker: -1.0}.
    """

    name: str = "strategy"
    frequency: str = "M"  # "D", "W" or "M": how often the engine calls generate
    required_data: tuple[str, ...] = ("close",)

    def check_data(self, data: MarketData) -> None:
        missing = [t for t in self.required_data if not data.has(t)]
        if missing:
            raise ValueError(f"{self.name} needs data tables {missing}")

    def fit(self, data: MarketData, train_end: pd.Timestamp) -> None:  # noqa: B027
        """Optional training step. Default: nothing to learn."""

    @abstractmethod
    def generate(self, data: MarketData, t: pd.Timestamp) -> pd.Series:
        """Target weights per ticker at date t."""

    def params(self) -> dict:
        """Config logged with every run. Override to expose hyperparameters."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_") and _jsonable(v)}

    def diagnostics(self) -> dict:
        """Optional model diagnostics logged with the run (CV scores, loadings, ...)."""
        return {}


def _jsonable(v) -> bool:
    return isinstance(v, (int, float, str, bool, type(None), list, tuple, dict))
