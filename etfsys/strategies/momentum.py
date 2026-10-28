"""Plug-in 2: multi-horizon ETF momentum with diversification caps.

A simplified version of the rules in Magner & Sanhueza (2026): rank the universe on
an average of several lookbacks, hold the top N with at most `max_per_category`
funds from one category, equal weight. Exits (stop-loss / take-profit) are NOT coded
here: they live in the shared risk overlay, so any strategy can use them.
"""
from __future__ import annotations

import pandas as pd

from ..data import MarketData
from ..features import momentum_score
from ..strategy import Strategy


class MultiHorizonMomentum(Strategy):
    frequency = "W"
    required_data = ("close", "meta")

    def __init__(
        self,
        horizons: tuple = (4, 12, 26),
        top_n: int = 4,
        max_per_category: int = 2,
        min_score: float = 0.0,
        universe: list[str] | None = None,
        name: str = "momentum_multi_horizon",
    ):
        self.horizons = list(horizons)
        self.top_n = top_n
        self.max_per_category = max_per_category
        self.min_score = min_score
        self.universe = universe
        self.name = name

    def generate(self, data: MarketData, t: pd.Timestamp) -> pd.Series:
        self.check_data(data)
        weekly = data.resample(self.frequency).close
        if self.universe:
            weekly = weekly[self.universe]
        if len(weekly) <= max(self.horizons):
            return pd.Series(dtype=float)
        score = momentum_score(weekly, self.horizons).iloc[-1].dropna()
        score = score[score > self.min_score].sort_values(ascending=False)  # absolute filter: cash if nothing trends up

        picks, per_cat = [], {}
        for tk in score.index:
            cat = data.meta.loc[tk, "category"]
            if per_cat.get(cat, 0) >= self.max_per_category:
                continue
            picks.append(tk)
            per_cat[cat] = per_cat.get(cat, 0) + 1
            if len(picks) == self.top_n:
                break
        if not picks:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(picks), index=picks)
