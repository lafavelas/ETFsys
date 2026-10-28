"""End-to-end example on synthetic or real data.

Runs three strategies through the SAME data, portfolio, risk and cost layers and
prints a side-by-side comparison plus cost sensitivity.

    uv run python run_example.py               # synthetic, ~1 min
    uv run python run_example.py --data real   # needs `python -m etfsys.data_sources.refresh`

Real data stops 2 years before the last price: that holdout stays locked (CLAUDE.md rule 6).
"""
import argparse

import pandas as pd

from etfsys import (
    Backtester, BacktestConfig, CostModel, PortfolioConstructor, RiskOverlay,
    StopLossTakeProfit, VolTarget, make_synthetic_data,
)
from etfsys.metrics import regime_split
from etfsys.strategies import BuyAndHold, MultiHorizonMomentum, SVMFactorStrategy
from etfsys.tracking import save_run

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", "{:,.3f}".format)

parser = argparse.ArgumentParser()
parser.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
args = parser.parse_args()

if args.data == "real":
    from etfsys.data_sources import load_market_data, load_reference, tracking_gap
    from etfsys.data_sources.build import SPY_EXPENSE_RATIO

    data = load_market_data()
    holdout_start = data.close.index[-1] - pd.DateOffset(years=2)
    end = str(data.close.index[data.close.index < holdout_start][-1].date())
    # Macro is already indexed by release date, so no extra lag on top.
    macro_lag = 0
    gap = tracking_gap(data.close["SPY"], load_reference()["^SP500TR"], SPY_EXPENSE_RATIO)
    print(f"data check: SPY vs S&P 500 TR after fees {gap:+.3%}/yr; holdout locked from {holdout_start.date()}")
else:
    data = make_synthetic_data()
    end = "2025-12-31"
    macro_lag = 1  # synthetic macro is stamped at the reference month end

config = BacktestConfig(start="2008-01-01", end=end, refit_every_months=12,
                        cost=CostModel(commission_bps=5, spread_bps=2))

runs = [
    # Benchmark: no overlay
    (BuyAndHold({"SPY": 1.0}, name="buy_hold_SPY"), RiskOverlay()),
    # Plug-in 1: Baek et al. SVM factor model, long or cash on SPY
    (SVMFactorStrategy(trade_ticker="SPY", macro_lag_periods=macro_lag), RiskOverlay()),
    # Plug-in 2: momentum with the shared stop-loss and vol-target rules
    (MultiHorizonMomentum(top_n=4, max_per_category=2),
     RiskOverlay([StopLossTakeProfit(stop=0.08), VolTarget(target=0.12)])),
]

results = {}
for strategy, overlay in runs:
    bt = Backtester(data, config, portfolio=PortfolioConstructor(long_only=True), risk=overlay)
    res = bt.run(strategy)
    results[res.name] = res
    folder = save_run(res, strategy, bt, data_version=args.data)
    print(f"saved {res.name} -> {folder}")

print("\n=== Out-of-sample comparison (net of costs) ===")
cols = ["cagr", "vol", "sharpe", "max_drawdown", "calmar", "turnover_per_year", "cost_drag_per_year"]
print(pd.DataFrame({k: v.metrics() for k, v in results.items()}).T[cols])

print("\n=== Cost sensitivity: Sharpe by bps per unit turnover ===")
print(pd.DataFrame({k: v.cost_sensitivity()["sharpe"] for k, v in results.items()}))

svm = results["svm_factor_SPY"]
print("\n=== SVM refits (walk-forward, time-ordered CV) ===")
print(pd.DataFrame(svm.fit_log)[["train_end", "n_train", "share_up_labels", "best_params", "cv_balanced_accuracy", "cv_std_of_best"]].to_string(index=False))

# Regime split using a stress proxy: month-end VIX above its expanding 80th percentile
vix = data.resample("M").macro["vix"].dropna()
stress = (vix > vix.expanding(36).quantile(0.8)).map({True: "stress", False: "calm"})
print("\n=== Regime split (SVM vs buy-and-hold) ===")
for name in ("svm_factor_SPY", "buy_hold_SPY"):
    print(name)
    print(regime_split(results[name].returns, stress))
