# etfsys — multi-strategy ETF research skeleton

Generalizes the 7-module robo-advisor of Baek et al. (2020), *Robo-Advisors: Machine
Learning in Trend-Following ETF Investments* (Sustainability 12:6399), so any ETF
strategy plugs into shared data, portfolio, risk, cost and backtest layers.

## Quick start

```bash
pip install numpy pandas scikit-learn pytest
python run_example.py     # ~1 min on synthetic data
pytest -q                 # 8 tests: no look-ahead, costs, stops, PCA fit window
```

`run_example.py` runs three strategies through the same stack and prints net metrics,
cost sensitivity (0–50 bps), every SVM walk-forward refit, and a calm/stress regime split.
Each run is saved under `runs/<timestamp>_<name>/` (`run.json`, `equity.csv`, `weights.csv`).

**The data is synthetic.** Results show the plumbing works; they say nothing about real
markets. Replace `make_synthetic_data()` with a `MarketData` built from your Data
Gathering / Data Management modules.

## Layout

```
etfsys/
  data.py          MarketData: point-in-time tables, truncate(t), resample(freq), validation
  features.py      FactorExtractor (PCA), zigzag_labels, breakout_signals, ema_ratio, momentum_score
  strategy.py      Strategy base class: fit(data, train_end) / generate(data, t) -> target weights
  strategies/
    svm_factor.py  Plug-in 1: Baek et al. SVM factor model (with the fixes below)
    momentum.py    Plug-in 2: multi-horizon momentum with category caps
    buy_and_hold.py  Benchmark
    _template.py   Copy to add a strategy
  context.py       State passed to portfolio and risk rules on each date
  portfolio.py     PortfolioConstructor: long-only, caps, gross limit, inverse-vol
  risk.py          RiskOverlay + VolTarget, StopLossTakeProfit, RegimeFilter, DrawdownBrake
  costs.py         CostModel (commission + half spread, per unit turnover)
  backtest.py      Backtester: daily clock, walk-forward refits, BacktestResult
  metrics.py       CAGR, vol, Sharpe, Sortino, max DD, Calmar, regime split
  tracking.py      save_run(): config + metrics + curves per run
run_example.py
tests/test_core.py
```

## What changed vs. the paper's ML Analytic Module

| Paper | Here | Why |
|---|---|---|
| PCA (likely) on full sample | `FactorExtractor` fit on each training window only | No test-period information in the factors |
| Macro at reference month | `macro_lag_periods=1` | Indicators are known only after release |
| Shuffled 10-fold CV, γ only | `TimeSeriesSplit`, γ and C | Shuffled folds train on the future |
| Raw EMA level | `close / EMA − 1` | Scale-free input for an RBF kernel |
| One 60/40 split | Walk-forward refit every 12 months | Tests stability of the model over time |
| Signal goes straight to trade | Portfolio → risk overlay → cost model | Shared by every strategy |

## Adding a strategy

Copy `strategies/_template.py`. Implement `generate` (and `fit` if it learns). Exits,
sizing limits and costs come from the shared layers, configured per run:

```python
bt = Backtester(data, BacktestConfig(start="2010-01-01", cost=CostModel(5, 2)),
                portfolio=PortfolioConstructor(long_only=True, max_weight=0.3),
                risk=RiskOverlay([StopLossTakeProfit(stop=0.08), VolTarget(0.10)]))
result = bt.run(MyStrategy())
```

## Known limits

- Trades fill at the close of the decision day; no slippage beyond the cost model.
- Intraday (overnight/daytime) strategies need open-to-close legs in the engine: planned.
- `cost_sensitivity()` re-prices from logged turnover; exact unless DrawdownBrake is on.
- Runtime is dominated by resampling inside `generate`; cache weekly/monthly views if it grows.
