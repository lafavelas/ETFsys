"""Performance metrics computed from daily net returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = 252


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    return float((eq / eq.cummax() - 1).min())


def summarize(returns: pd.Series, turnover: pd.Series | None = None, costs: pd.Series | None = None) -> dict:
    r = returns.dropna()
    if r.empty:
        return {}
    years = len(r) / PERIODS
    total = float((1 + r).prod())
    cagr = total ** (1 / years) - 1 if years > 0 and total > 0 else np.nan
    vol = float(r.std() * np.sqrt(PERIODS))
    downside = float(r[r < 0].std() * np.sqrt(PERIODS)) if (r < 0).any() else np.nan
    mean_ann = float(r.mean() * PERIODS)
    mdd = max_drawdown(r)
    out = {
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
        "cagr": cagr,
        "vol": vol,
        "sharpe": mean_ann / vol if vol > 0 else np.nan,
        "sortino": mean_ann / downside if downside and downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "hit_rate": float((r[r != 0] > 0).mean()) if (r != 0).any() else np.nan,
        "exposure_days": float((r != 0).mean()),
    }
    if turnover is not None:
        out["turnover_per_year"] = float(turnover.sum() / years)
    if costs is not None:
        out["cost_drag_per_year"] = float(costs.sum() / years)
    return out


def regime_split(returns: pd.Series, regime: pd.Series) -> pd.DataFrame:
    """Metrics per regime label (e.g. 'calm'/'stress'), regime aligned to return dates."""
    reg = regime.reindex(returns.index).ffill()
    rows = {}
    for label, grp in returns.groupby(reg):
        s = summarize(grp)
        rows[label] = {k: s.get(k) for k in ("cagr", "vol", "sharpe", "max_drawdown")}
        rows[label]["days"] = len(grp)
    return pd.DataFrame(rows).T
