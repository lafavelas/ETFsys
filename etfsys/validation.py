"""Validation protocol: regime and crisis breakdowns, and selection-bias-adjusted Sharpe.

Every function here is descriptive: it judges finished backtests and feeds no decision
back into a strategy. Regime labels still use only what was known at the time, so a
breakdown never credits a strategy for knowing tomorrow's VIX.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from .data import MarketData
from .metrics import max_drawdown, regime_split

EULER_GAMMA = 0.5772156649015329

# Peak-to-trough windows of the S&P 500 (CLAUDE.md / architecture: 2008, 2020, 2022).
CRISES: dict[str, tuple[str, str]] = {
    "2008 GFC": ("2007-10-09", "2009-03-09"),
    "2020 COVID": ("2020-02-19", "2020-03-23"),
    "2022 bear": ("2022-01-03", "2022-10-12"),
}


# ------------------------------------------------------------------- regimes
def vix_regime(data: MarketData, index: pd.DatetimeIndex, threshold: float = 25.0) -> pd.Series:
    """'calm' / 'stress' per return date, from the VIX known at the PREVIOUS close.

    A day's return is earned on positions held from the prior close, so that close is
    when the regime was observable. Macro is indexed by release date, so the as-of
    value is a forward fill; no later print can reach an earlier date.
    """
    if data.macro is None or "vix" not in data.macro:
        raise ValueError("vix_regime needs a 'vix' column in data.macro")
    vix = data.macro["vix"].dropna()
    asof = vix.reindex(vix.index.union(index)).ffill().reindex(index).shift(1)
    label = pd.Series(np.where(asof >= threshold, "stress", "calm"), index=index)
    return label.where(asof.notna())


def regime_table(
    returns: dict[str, pd.Series], data: MarketData, threshold: float = 25.0
) -> pd.DataFrame:
    """(strategy, regime) x metrics for each return series."""
    parts = {}
    for name, r in returns.items():
        parts[name] = regime_split(r, vix_regime(data, r.index, threshold))
    return pd.concat(parts, names=["strategy", "regime"])


def crisis_table(
    returns: dict[str, pd.Series], crises: dict[str, tuple[str, str]] = CRISES
) -> pd.DataFrame:
    """(strategy, crisis) x total return and max drawdown over the part the run covers.

    `from`/`to` show the covered dates, so a run starting mid-crisis is visible as such.
    """
    rows = {}
    for name, r in returns.items():
        for crisis, (lo, hi) in crises.items():
            window = r.loc[lo:hi]
            if window.empty:
                continue
            rows[(name, crisis)] = {
                "from": str(window.index[0].date()),
                "to": str(window.index[-1].date()),
                "total_return": float((1 + window).prod() - 1),
                "max_drawdown": max_drawdown(window),
                "days": len(window),
            }
    out = pd.DataFrame.from_dict(rows, orient="index")
    if not out.empty:
        out.index.names = ["strategy", "crisis"]
    return out


# ------------------------------------------------------- selection-bias Sharpe
def sharpe_stats(returns: pd.Series) -> dict:
    """Per-period (not annualized) Sharpe and the moments the deflated Sharpe needs."""
    r = returns.dropna()
    sd = r.std()
    return {
        "sharpe_per_period": float(r.mean() / sd) if sd > 0 else 0.0,
        "skew": float(skew(r)) if len(r) > 2 else 0.0,
        "kurtosis": float(kurtosis(r, fisher=False)) if len(r) > 3 else 3.0,  # 3 = normal
        "n_obs": len(r),
    }


def expected_max_sharpe(trial_sharpes: list[float]) -> float:
    """Sharpe the best of N skill-less trials would reach by luck (Bailey & Lopez de Prado 2014).

    Uses the dispersion of the trials' own per-period Sharpes. One trial: no luck to remove.
    """
    n = len(trial_sharpes)
    if n < 2:
        return 0.0
    sd = float(np.std(trial_sharpes, ddof=1))
    return sd * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))
    )


def deflated_sharpe(returns: pd.Series, trial_sharpes: list[float]) -> float:
    """Probability the true Sharpe beats the best-of-N luck benchmark (0..1).

    `trial_sharpes` are the per-period Sharpes of every configuration tried in the
    study, this one included. Above 0.95 is the usual bar; with one trial this is the
    probabilistic Sharpe ratio against zero.
    """
    s = sharpe_stats(returns)
    sr, t = s["sharpe_per_period"], s["n_obs"]
    if t < 3:
        return float("nan")
    sr0 = expected_max_sharpe(trial_sharpes)
    denom = 1 - s["skew"] * sr + (s["kurtosis"] - 1) / 4 * sr**2
    return float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / np.sqrt(max(denom, 1e-12))))
