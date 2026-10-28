"""Portfolio construction: strategy intent -> valid, constrained weights."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .context import Context


@dataclass
class PortfolioConstructor:
    max_gross: float = 1.0      # sum of |w|; 1.0 = no leverage
    max_weight: float = 1.0     # cap per position
    long_only: bool = True
    inverse_vol: bool = False   # rescale picks by 1/vol, keeping the strategy's gross
    vol_window: int = 63        # trading days

    def apply(self, target: pd.Series, ctx: Context) -> pd.Series:
        w = target.reindex(ctx.tickers).fillna(0.0).astype(float)
        w[ctx.price.isna()] = 0.0  # fund not launched yet: nothing to buy
        if self.long_only:
            w = w.clip(lower=0.0)
        if self.inverse_vol and (w != 0).any():
            gross = w.abs().sum()
            vol = ctx.returns.iloc[-self.vol_window :].std().replace(0, np.nan)
            raw = np.sign(w) * (w.abs() / vol).fillna(0.0)
            if raw.abs().sum() > 0:
                w = raw * gross / raw.abs().sum()
        w = w.clip(-self.max_weight, self.max_weight)
        gross = w.abs().sum()
        if gross > self.max_gross:
            w *= self.max_gross / gross
        return w
