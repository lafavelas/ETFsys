"""Risk overlay: a chain of independent rules that can only cut or rescale exposure.

Rules with `daily = True` are checked every trading day (stops); the others only on
rebalance dates. Any strategy gets any rule without changing its own code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .context import Context


class RiskRule:
    daily: bool = False
    name: str = "rule"

    def apply(self, w: pd.Series, ctx: Context) -> pd.Series:  # pragma: no cover
        raise NotImplementedError


@dataclass
class VolTarget(RiskRule):
    """Scale the book so its ex-ante volatility is about `target` per year."""

    target: float = 0.10
    window: int = 63
    max_gross: float = 1.0
    name: str = "vol_target"

    def apply(self, w, ctx):
        live = w[w != 0]
        if live.empty:
            return w
        rets = ctx.returns[live.index].iloc[-self.window :]
        if len(rets) < self.window // 2:
            return w
        vol = float(np.sqrt(live.values @ rets.cov().values @ live.values) * np.sqrt(252))
        if vol <= 0:
            return w
        scale = min(self.target / vol, self.max_gross / live.abs().sum())
        return w * scale


@dataclass
class StopLossTakeProfit(RiskRule):
    """Close a position when its move since entry hits the stop or the target.

    The position stays closed until the strategy's next rebalance re-opens it.
    """

    stop: float | None = 0.08
    take: float | None = None
    daily: bool = True
    name: str = "stop_take"

    def apply(self, w, ctx):
        w = w.copy()
        price = ctx.price
        for tk, entry in ctx.entry_price.items():
            if w.get(tk, 0.0) == 0.0:
                continue
            move = np.sign(w[tk]) * (price[tk] / entry - 1.0)
            if (self.stop is not None and move <= -self.stop) or (
                self.take is not None and move >= self.take
            ):
                w[tk] = 0.0
        return w


@dataclass
class RegimeFilter(RiskRule):
    """Scale exposure down when a stress series is above its point-in-time threshold.

    `threshold` is a fixed level, or None to use the expanding `quantile` of the
    series' own history up to t (no look-ahead).
    """

    series: str = "vix"  # a macro column
    threshold: float | None = None
    quantile: float = 0.8
    scale: float = 0.0  # 0 = go to cash, 0.5 = halve exposure
    min_history: int = 36
    name: str = "regime_filter"

    def apply(self, w, ctx):
        macro = ctx.history.macro
        if macro is None or self.series not in macro or len(macro) < self.min_history:
            return w
        s = macro[self.series].dropna()
        level = (
            self.threshold if self.threshold is not None else s.iloc[:-1].quantile(self.quantile)
        )
        return w * self.scale if s.iloc[-1] > level else w


@dataclass
class DrawdownBrake(RiskRule):
    """Cut exposure while the portfolio is more than `max_drawdown` below its peak."""

    max_drawdown: float = 0.15
    scale: float = 0.5
    name: str = "drawdown_brake"

    def apply(self, w, ctx):
        return w * self.scale if ctx.drawdown <= -self.max_drawdown else w


@dataclass
class RiskOverlay:
    rules: list[RiskRule] = field(default_factory=list)

    def apply(self, w: pd.Series, ctx: Context, daily_only: bool = False) -> pd.Series:
        for rule in self.rules:
            if daily_only and not rule.daily:
                continue
            w = rule.apply(w, ctx)
        return w

    @property
    def has_daily(self) -> bool:
        return any(r.daily for r in self.rules)

    def describe(self) -> list[dict]:
        return [
            {"rule": r.name, **{k: v for k, v in vars(r).items() if k != "name"}}
            for r in self.rules
        ]
