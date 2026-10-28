"""etfsys: a multi-strategy ETF research and backtest skeleton.

Generalizes the 7-module robo-advisor of Baek et al. (2020, Sustainability 12:6399)
so that any strategy plugs into shared data, risk, cost and backtest layers.
"""

from .backtest import BacktestConfig, Backtester, BacktestResult
from .context import Context
from .costs import CostModel
from .data import MarketData, make_synthetic_data
from .portfolio import PortfolioConstructor
from .risk import DrawdownBrake, RegimeFilter, RiskOverlay, StopLossTakeProfit, VolTarget
from .strategy import Strategy

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
    "Context",
    "CostModel",
    "DrawdownBrake",
    "MarketData",
    "PortfolioConstructor",
    "RegimeFilter",
    "RiskOverlay",
    "StopLossTakeProfit",
    "Strategy",
    "VolTarget",
    "make_synthetic_data",
]
