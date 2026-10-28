"""Trial counter: how many configurations a study tried before picking its result.

A high Sharpe found after 200 tries means less than the same Sharpe found on the first
try. Each study keeps an append-only log, `runs/studies/<name>.jsonl`, with one line per
run. Identical configs share a hash and count once, so re-running is not a new trial.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .backtest import Backtester, BacktestResult
from .strategy import Strategy
from .tracking import _clean, run_config
from .validation import deflated_sharpe, sharpe_stats


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Study:
    """One research question, e.g. "tune SVM gamma/C" or "momentum lookbacks"."""

    name: str
    root: str | Path = "runs/studies"

    @property
    def path(self) -> Path:
        return Path(self.root) / f"{self.name}.jsonl"

    def record(
        self, result: BacktestResult, strategy: Strategy, bt: Backtester, data_version: str
    ) -> dict:
        """Append one trial line and return it."""
        config = run_config(strategy, bt, data_version)
        m = result.metrics()
        trial = {
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "config_hash": config_hash(config),
            "strategy": result.name,
            "sharpe": m.get("sharpe"),
            **sharpe_stats(result.returns),
            "holdout_touched": result.holdout_touched,
            "config": config,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:  # append only: never rewrite history
            f.write(json.dumps(_clean(trial)) + "\n")
        return trial

    def trials(self) -> pd.DataFrame:
        """One row per distinct config (the latest line for each hash)."""
        if not self.path.exists():
            return pd.DataFrame(columns=["config_hash", "strategy", "sharpe", "sharpe_per_period"])
        lines = [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x]
        df = pd.DataFrame(lines)
        return df.drop_duplicates("config_hash", keep="last").reset_index(drop=True)

    @property
    def n_trials(self) -> int:
        return len(self.trials())

    def deflated_sharpe(self, result: BacktestResult) -> float:
        """Deflated Sharpe of `result` against every trial logged in this study."""
        return deflated_sharpe(result.returns, list(self.trials()["sharpe_per_period"]))
