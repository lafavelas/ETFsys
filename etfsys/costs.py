"""Transaction cost model. Kept simple and linear so results can be re-priced at any level."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CostModel:
    commission_bps: float = 5.0  # per unit of traded notional
    spread_bps: float = 2.0  # full bid-ask spread; half is paid per trade

    @property
    def bps_per_turnover(self) -> float:
        return self.commission_bps + self.spread_bps / 2

    def turnover(self, old: pd.Series, new: pd.Series) -> float:
        idx = old.index.union(new.index)
        return float(
            (new.reindex(idx, fill_value=0.0) - old.reindex(idx, fill_value=0.0)).abs().sum()
        )

    def cost(self, old: pd.Series, new: pd.Series) -> float:
        """Cost as a fraction of portfolio value."""
        return self.turnover(old, new) * self.bps_per_turnover / 1e4
