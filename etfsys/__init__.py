"""etfsys: a multi-strategy ETF research and backtest skeleton.

Generalizes the 7-module robo-advisor of Baek et al. (2020, Sustainability 12:6399)
so that any strategy plugs into shared data, risk, cost and backtest layers.
"""
from .backtest import Backtester, BacktestConfig, BacktestResult
from .context import Context
from .costs import CostModel
from .data import MarketData, make_synthetic_data
from .portfolio import PortfolioConstructor
from .risk import DrawdownBrake, RegimeFilter, RiskOverlay, StopLossTakeProfit, VolTarget
from .strategy import Strategy

__all__ = [
    "Backtester", "BacktestConfig", "BacktestResult", "Context", "CostModel",
    "MarketData", "make_synthetic_data", "PortfolioConstructor", "RiskOverlay",
    "VolTarget", "StopLossTakeProfit", "RegimeFilter", "DrawdownBrake", "Strategy",
]
