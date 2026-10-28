"""The standard result report (CLAUDE.md rule 5): net metrics, cost sensitivity, the
buy-and-hold SPY benchmark, regime and crisis breakdowns, trial count and deflated Sharpe.

Build it with `validation_report`, print it with `format_report`. Phase 5 renders the
same tables as HTML.
"""

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult
from .data import MarketData
from .study import Study
from .validation import CRISES, crisis_table, deflated_sharpe, regime_table

METRIC_COLS = [
    "cagr",
    "vol",
    "sharpe",
    "max_drawdown",
    "calmar",
    "turnover_per_year",
    "cost_drag_per_year",
]


def validation_report(
    results: dict[str, BacktestResult],
    data: MarketData,
    benchmark: str = "buy_hold_SPY",
    study: Study | None = None,
    vix_threshold: float = 25.0,
    crises: dict[str, tuple[str, str]] = CRISES,
) -> dict[str, pd.DataFrame | str]:
    """All tables for one comparison. `results` must include the SPY benchmark."""
    if benchmark not in results:
        raise ValueError(f"report needs the benchmark '{benchmark}' among the results (rule 5)")
    returns = {k: r.returns for k, r in results.items()}
    metrics = pd.DataFrame({k: r.metrics() for k, r in results.items()}).T
    metrics = metrics[[c for c in METRIC_COLS if c in metrics]]
    bench = metrics.loc[benchmark]
    metrics["excess_cagr_vs_benchmark"] = metrics["cagr"] - bench["cagr"]

    trial_sharpes = list(study.trials()["sharpe_per_period"]) if study is not None else []
    selection = pd.DataFrame(
        {
            k: {
                "n_trials": len(trial_sharpes) if (study and k != benchmark) else 1,
                "deflated_sharpe": deflated_sharpe(
                    r.returns, trial_sharpes if k != benchmark else []
                ),
            }
            for k, r in results.items()
        }
    ).T

    period = next(iter(results.values())).returns.index
    holdout = (
        "no holdout defined"
        if data.holdout_start is None
        else (
            f"holdout from {pd.Timestamp(data.holdout_start).date()}: "
            + ("USED (final run)" if any(r.holdout_touched for r in results.values()) else "locked")
        )
    )
    return {
        "header": f"{period[0].date()} -> {period[-1].date()}, net of costs; {holdout}",
        "metrics": metrics,
        "cost_sharpe": pd.DataFrame(
            {k: r.cost_sensitivity()["sharpe"] for k, r in results.items()}
        ),
        "cost_cagr": pd.DataFrame({k: r.cost_sensitivity()["cagr"] for k, r in results.items()}),
        "regimes": regime_table(returns, data, vix_threshold),
        "crises": crisis_table(returns, crises),
        "selection": selection,
    }


def format_report(report: dict[str, pd.DataFrame | str]) -> str:
    titles = {
        "metrics": "Out-of-sample metrics (net of costs)",
        "cost_sharpe": "Cost sensitivity: Sharpe by bps per unit turnover",
        "cost_cagr": "Cost sensitivity: CAGR by bps per unit turnover",
        "regimes": "Regimes: VIX known at the previous close",
        "crises": "Crises (part covered by the run)",
        "selection": (
            "Selection bias: deflated Sharpe = P(true Sharpe > best of n_trials by luck); "
            "vs. zero, not vs. the benchmark; >0.95 = robust"
        ),
    }
    parts = [f"=== {report['header']} ==="]
    for key, title in titles.items():
        table = report[key]
        body = "(none in the run window)" if table.empty else table.to_string()
        parts.append(f"\n--- {title} ---\n{body}")
    return "\n".join(parts)
