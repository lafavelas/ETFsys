"""Event-driven, walk-forward backtest engine shared by every strategy.

Daily clock. On each day:
  1. positions earn that day's close-to-close return and drift;
  2. daily risk rules (stops) may close positions;
  3. on a rebalance date: refit if due (expanding window, data up to today only),
     ask the strategy for targets, run portfolio construction then the risk overlay,
     trade at today's close and pay costs.
Decisions at close t therefore earn returns from t+1 on. No future data reaches the
strategy: it only ever sees `data.truncate(t)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .context import Context
from .costs import CostModel
from .data import MarketData, period_last
from .metrics import summarize
from .portfolio import PortfolioConstructor
from .risk import RiskOverlay
from .strategy import Strategy


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "D":
        return index
    marker = pd.DataFrame({"x": 0}, index=index)
    return period_last(marker, freq).index


@dataclass
class BacktestConfig:
    start: str  # first out-of-sample trading date
    end: str | None = None
    refit_every_months: int = 12  # walk-forward refit, expanding window
    cost: CostModel = field(default_factory=CostModel)


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series  # daily net returns
    gross_returns: pd.Series  # before costs
    turnover: pd.Series  # sum |dw| traded each day
    costs: pd.Series  # cost paid each day (fraction of equity)
    weights: pd.DataFrame  # end-of-day weights after trading
    fit_log: list = field(default_factory=list)
    trades: list = field(default_factory=list)

    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    def metrics(self) -> dict:
        return summarize(self.returns, self.turnover, self.costs)

    def reprice(self, bps_per_turnover: float) -> pd.Series:
        """Net returns at another cost level. Exact unless a rule depends on equity
        (DrawdownBrake), in which case it is an approximation."""
        cost = self.turnover * bps_per_turnover / 1e4
        return (1 + self.gross_returns) * (1 - cost) - 1

    def cost_sensitivity(self, levels=(0, 5, 10, 20, 50)) -> pd.DataFrame:
        rows = {}
        for bps in levels:
            s = summarize(self.reprice(bps))
            rows[f"{bps} bps"] = {k: s[k] for k in ("cagr", "sharpe", "max_drawdown")}
        return pd.DataFrame(rows).T


class Backtester:
    def __init__(
        self,
        data: MarketData,
        config: BacktestConfig,
        portfolio: PortfolioConstructor | None = None,
        risk: RiskOverlay | None = None,
    ):
        self.data = data.validate()
        self.config = config
        self.portfolio = portfolio or PortfolioConstructor()
        self.risk = risk or RiskOverlay()

    def run(self, strategy: Strategy, verbose: bool = False) -> BacktestResult:
        strategy.check_data(self.data)
        close = self.data.close
        tickers = list(close.columns)
        rets = close.pct_change(fill_method=None).fillna(0.0)
        days = close.loc[self.config.start : self.config.end].index
        rebal = set(rebalance_dates(close.index, strategy.frequency).intersection(days))

        w = pd.Series(0.0, index=tickers)
        equity, peak = 1.0, 1.0
        entry: dict[str, float] = {}
        last_fit: pd.Timestamp | None = None
        out_ret, out_gross, out_turn, out_cost, out_w = [], [], [], [], []
        fit_log, trades = [], []

        for i, t in enumerate(days):
            # 1. earn today's return on yesterday's positions, then drift
            r = rets.loc[t]
            gross = float((w * r).sum()) if i > 0 else 0.0
            if i > 0:
                w = w * (1 + r) / (1 + gross)
                equity *= 1 + gross
                peak = max(peak, equity)

            new_w = w
            ctx = Context(
                t=t, data=self.data, weights=w, equity=equity, peak_equity=peak, entry_price=entry
            )

            # 2. daily rules (stops)
            if self.risk.has_daily:
                new_w = self.risk.apply(new_w, ctx, daily_only=True)

            # 3. rebalance
            if t in rebal:
                due = last_fit is None or (
                    (t.to_period("M") - last_fit.to_period("M")).n >= self.config.refit_every_months
                )
                if due:
                    strategy.fit(self.data.truncate(t), t)
                    last_fit = t
                    diag = strategy.diagnostics()
                    if diag:
                        fit_log.append(diag)
                    if verbose:
                        print(
                            f"[{strategy.name}] refit at {t.date()} {diag.get('cv_balanced_accuracy', '')}"
                        )
                target = strategy.generate(self.data.truncate(t), t)
                new_w = self.portfolio.apply(target, ctx)
                new_w = self.risk.apply(new_w, ctx)

            turnover = self.config.cost.turnover(w, new_w)
            cost = self.config.cost.cost(w, new_w)
            if turnover > 1e-12:
                trades.append({"date": t, "turnover": turnover, "cost": cost})
                price = close.loc[t]
                for tk in tickers:
                    old, new = w[tk], new_w[tk]
                    if new == 0:
                        entry.pop(tk, None)
                    elif old == 0 or np.sign(old) != np.sign(new):
                        entry[tk] = float(price[tk])
            equity *= 1 - cost
            w = new_w

            out_gross.append(gross)
            out_ret.append((1 + gross) * (1 - cost) - 1)
            out_turn.append(turnover)
            out_cost.append(cost)
            out_w.append(w.values.copy())

        idx = pd.DatetimeIndex(days)
        return BacktestResult(
            name=strategy.name,
            returns=pd.Series(out_ret, index=idx, name=strategy.name),
            gross_returns=pd.Series(out_gross, index=idx),
            turnover=pd.Series(out_turn, index=idx),
            costs=pd.Series(out_cost, index=idx),
            weights=pd.DataFrame(out_w, index=idx, columns=tickers),
            fit_log=fit_log,
            trades=trades,
        )
