"""End-to-end example on synthetic data.

Runs three strategies through the SAME data, portfolio, risk and cost layers and
prints a side-by-side comparison plus cost sensitivity. Replace
`make_synthetic_data()` with your own MarketData to use real data.

    python run_example.py
"""
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

data = make_synthetic_data()
config = BacktestConfig(start="2008-01-01", end="2025-12-31", refit_every_months=12,
                        cost=CostModel(commission_bps=5, spread_bps=2))

runs = [
    # Benchmark: no overlay
    (BuyAndHold({"SPY": 1.0}, name="buy_hold_SPY"), RiskOverlay()),
    # Plug-in 1: Baek et al. SVM factor model, long or cash on SPY
    (SVMFactorStrategy(trade_ticker="SPY"), RiskOverlay()),
    # Plug-in 2: momentum with the shared stop-loss and vol-target rules
    (MultiHorizonMomentum(top_n=4, max_per_category=2),
     RiskOverlay([StopLossTakeProfit(stop=0.08), VolTarget(target=0.12)])),
]

results = {}
for strategy, overlay in runs:
    bt = Backtester(data, config, portfolio=PortfolioConstructor(long_only=True), risk=overlay)
    res = bt.run(strategy)
    results[res.name] = res
    folder = save_run(res, strategy, bt)
    print(f"saved {res.name} -> {folder}")

print("\n=== Out-of-sample comparison (net of costs) ===")
cols = ["cagr", "vol", "sharpe", "max_drawdown", "calmar", "turnover_per_year", "cost_drag_per_year"]
print(pd.DataFrame({k: v.metrics() for k, v in results.items()}).T[cols])

print("\n=== Cost sensitivity: Sharpe by bps per unit turnover ===")
print(pd.DataFrame({k: v.cost_sensitivity()["sharpe"] for k, v in results.items()}))

svm = results["svm_factor_SPY"]
print("\n=== SVM refits (walk-forward, time-ordered CV) ===")
print(pd.DataFrame(svm.fit_log)[["train_end", "n_train", "share_up_labels", "best_params", "cv_balanced_accuracy", "cv_std_of_best"]].to_string(index=False))

# Regime split using a stress proxy: VIX-like macro above its expanding 80th percentile
vix = data.macro["vix"]
stress = (vix > vix.expanding(36).quantile(0.8)).map({True: "stress", False: "calm"})
print("\n=== Regime split (SVM vs buy-and-hold) ===")
for name in ("svm_factor_SPY", "buy_hold_SPY"):
    print(name)
    print(regime_split(results[name].returns, stress))
