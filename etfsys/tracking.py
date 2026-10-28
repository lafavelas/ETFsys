"""Minimal experiment tracking: one folder per run with config, metrics and curves.

Swap for MLflow later; the fields to log stay the same.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from .backtest import Backtester, BacktestResult
from .strategy import Strategy


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o if isinstance(o, (int, str, bool, type(None))) else str(o)


def _git_rev() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):  # no git, or not a repo
        return None


def save_run(
    result: BacktestResult,
    strategy: Strategy,
    bt: Backtester,
    root: str = "runs",
    data_version: str = "synthetic",
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(root) / f"{stamp}_{result.name}"
    path.mkdir(parents=True, exist_ok=True)
    record = {
        "strategy": result.name,
        "strategy_params": strategy.params(),
        "backtest": {
            "start": bt.config.start,
            "end": bt.config.end,
            "refit_every_months": bt.config.refit_every_months,
            "cost_bps_per_turnover": bt.config.cost.bps_per_turnover,
        },
        "portfolio": vars(bt.portfolio),
        "risk": bt.risk.describe(),
        "data_version": data_version,
        "code_version": _git_rev(),
        "metrics": result.metrics(),
        "cost_sensitivity": result.cost_sensitivity().to_dict(orient="index"),
        "fit_log": result.fit_log,
    }
    (path / "run.json").write_text(json.dumps(_clean(record), indent=2))
    result.equity().to_csv(path / "equity.csv", header=["equity"])
    result.weights.to_csv(path / "weights.csv")
    return path
