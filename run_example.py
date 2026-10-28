"""End-to-end example on synthetic or real data.

Runs three strategies through the SAME data, portfolio, risk and cost layers and
prints a side-by-side comparison plus cost sensitivity.

    uv run python run_example.py               # synthetic, ~1 min
    uv run python run_example.py --data real   # needs `python -m etfsys.data_sources.refresh`

Real data stops before HOLDOUT_START (data_sources/config.py): the holdout stays locked.
Each candidate is logged as a trial in runs/studies/, which feeds the deflated Sharpe.
"""

import argparse

import pandas as pd

from etfsys import (
    BacktestConfig,
    Backtester,
    CostModel,
    PortfolioConstructor,
    RiskOverlay,
    StopLossTakeProfit,
    VolTarget,
    make_synthetic_data,
)
from etfsys.report import format_report, validation_report
from etfsys.strategies import BuyAndHold, MultiHorizonMomentum, SVMFactorStrategy
from etfsys.study import Study
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

    data = load_market_data()  # stops before the locked holdout
    macro_lag = 0  # macro is already indexed by release date: no extra lag on top
    vix_threshold = 25.0
    gap = tracking_gap(data.close["SPY"], load_reference()["^SP500TR"], SPY_EXPENSE_RATIO)
    print(f"data check: SPY vs S&P 500 TR after fees {gap:+.3%}/yr")
else:
    data = make_synthetic_data()
    macro_lag = 1  # synthetic macro is stamped at the reference month end
    vix_threshold = 15.0  # the synthetic VIX-like series ranges about 11-18

config = BacktestConfig(
    start="2008-01-01",
    refit_every_months=12,
    cost=CostModel(commission_bps=5, spread_bps=2),
)

benchmark = (BuyAndHold({"SPY": 1.0}, name="buy_hold_SPY"), RiskOverlay())
candidates = [
    # Plug-in 1: Baek et al. SVM factor model, long or cash on SPY
    (SVMFactorStrategy(trade_ticker="SPY", macro_lag_periods=macro_lag), RiskOverlay()),
    # Plug-in 2: momentum with the shared stop-loss and vol-target rules
    (
        MultiHorizonMomentum(top_n=4, max_per_category=2),
        RiskOverlay([StopLossTakeProfit(stop=0.08), VolTarget(target=0.12)]),
    ),
]

# Every candidate config is a trial of this study; the benchmark is not.
study = Study(f"run_example_{args.data}")
results = {}
for (strategy, overlay), is_trial in [(benchmark, False)] + [(c, True) for c in candidates]:
    bt = Backtester(data, config, portfolio=PortfolioConstructor(long_only=True), risk=overlay)
    res = bt.run(strategy)
    results[res.name] = res
    folder = save_run(res, strategy, bt, data_version=args.data, study=study if is_trial else None)
    print(f"saved {res.name} -> {folder}")

print()
print(format_report(validation_report(results, data, study=study, vix_threshold=vix_threshold)))

svm = results["svm_factor_SPY"]
print("\n--- SVM refits (walk-forward, time-ordered CV) ---")
cols = ["train_end", "n_train", "share_up_labels", "best_params", "cv_balanced_accuracy"]
print(pd.DataFrame(svm.fit_log)[cols].to_string(index=False))
