# Multi-Strategy ETF Trading System — Architecture Design

Oct 29, 2026 · @Quentin Favier

## Purpose and scope

The system generalizes the 7-module robo-advisor of [Baek et al. (2020)](https://doi.org/10.3390/su12166399) so that any ETF strategy plugs into one shared data, risk, cost and backtest stack. The paper's pipeline is built for a single model: an SVM on three PCA risk factors producing a monthly ±1 signal on SPY, IVV, VTI and QQQ.

Three design rules drive the changes:

- **Strategies emit target weights, not ±1 signals.** A long/short signal is the special case of ±100% in one asset; weights also cover ranked portfolios, risk parity and RL allocators.
- **Everything shared lives outside the strategy.** Data, features, portfolio construction, risk rules, costs and the backtest engine are written once and reused.
- **One validation protocol for every strategy.** Same walk-forward windows, cost model and benchmark, so results compare fairly.

The SVM factor model from the paper becomes the first plug-in and the reference implementation.

## Architecture overview

The paper's 7 modules become 6 layers, and only the strategy layer changes from one strategy to the next. The backtest engine drives everything: on each rebalance date it calls the strategy with data up to that date, then passes the target weights through portfolio, risk and cost layers.

&#91;embedded content: system architecture · 6 layers, 6 strategy plug-ins\]

| Paper module | Becomes |
| --- | --- |
| Data Gathering, Data Management | Data layer (`MarketData`) |
| Machine Learning Analytic | Feature and label libraries + SVM plug-in |
| Forecasting | `generate()` of each plug-in |
| Validation | Backtest engine with walk-forward refits |
| ETF Trading System | Portfolio builder, risk overlay, cost model |
| Trading Performance | Metrics, reports and experiment tracking |

## Data layer

The data layer hands every downstream module one `MarketData` object with aligned, point-in-time tables. Strategies declare which tables they need and never touch raw sources.

| Table | Index × columns | Used by | Change vs. paper |
| --- | --- | --- | --- |
| `close`, `open`, `high`, `low` | date × ticker | All strategies, backtest | Daily, adjusted for splits and dividends, open kept separately (paper: monthly close). NaN only before a fund's first bar |
| `volume` | date × ticker | Universe filter, cost model | New: shares traded (optional) |
| `macro` | date × 13 indicators | SVM factor model, regime filters | Indexed by the date each value became known (first release or release lag), not the reference month |
| `meta` | ticker × attributes | Universe filter, portfolio rules | New: category, asset class, leverage, inception date |
| `nav` | date × ticker | Premium signals | New: NAV for premium/discount features (planned) |
| liquidity | date × ticker | Universe filter, cost model | Planned: bid-ask spread, AUM |

Real data comes from `etfsys.data_sources`: yfinance prices and FRED/ALFRED macro, stored as Parquet in `data/raw` and `data/processed`, validated with pandera, and loaded by `load_market_data()`. Indicator sources, substitutes for the paper's proprietary or discontinued series, and the point-in-time rules are in `docs/data_dictionary.md`.

Four rules keep backtests honest:

- **Point in time.** A value enters the data on its release date. Revised macro series are a common source of look-ahead.
- **No survivorship.** Delisted ETFs stay in the universe for the dates they traded.
- **Resampling in one place.** Daily data is the base; weekly and monthly views are derived by the data layer, not by each strategy.
- **Contracts are validated.** Missing values, misaligned dates and unknown tickers fail loudly at load time.

## Strategy interface and plug-ins

Every strategy implements three members and nothing else. The backtest engine calls `fit` on each training window and `generate` on each rebalance date, passing only data available at that date.

```python
class Strategy(ABC):
    name: str
    frequency: str  # "D", "W", "M"
    required_data: tuple  # e.g. ("close", "macro")

    def fit(self, data: MarketData, train_end: pd.Timestamp) -> None: ...
    def generate(self, data: MarketData, t: pd.Timestamp) -> pd.Series:
        """Target weights per ticker at date t (sum of |w| <= 1 unless leverage allowed)."""
```

The paper's ML Analytic Module maps onto this interface without loss: PCA and the SVM are fitted in `fit`, and `generate` returns +1 or −1 on the traded ETF. Its sub-steps (factor extraction, labeling, feature building) move to shared libraries so other strategies can reuse them.

| Strategy | Source | Frequency | Output | New shared parts it needs |
| --- | --- | --- | --- | --- |
| SVM factor model | [Baek et al. 2020](https://doi.org/10.3390/su12166399) | M | ±1 on one ETF | Factor library, zigzag labels |
| Momentum with exit rules | [Magner & Sanhueza 2026](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7379989) | W | Top-8 equal weight | Universe filter, category caps, stop/take-profit overlay |
| Overnight long, daytime reversal | [Salotra et al. 2026](https://doi.org/10.3390/risks14040084) | D (intraday legs) | Weights per session | Open and close execution, cost sensitivity |
| RL allocator | [Bendatu & Lu 2026](https://arxiv.org/abs/2602.24037) | D or W | Continuous weights | Training environment on the backtest engine |
| NAV premium signal | [Rompotis 2026](https://www.preprints.org/manuscript/202602.0167/v1) | D | Weights | NAV table, premium features |
| Leveraged ETF overlay | [van Staden et al. 2025](https://arxiv.org/abs/2506.19200) | D | Weights with leverage cap | Leverage metadata, exposure limits |

The skeleton ships the first two rows; the other four are planned plug-ins.

## Execution stack: portfolio, risk, costs, backtest

Between a strategy's target weights and a P&L sit four shared components, applied in a fixed order on every rebalance date. None of them exists in the paper, which goes straight from SVM signal to trade.

1. **Portfolio construction** turns raw weights into a valid portfolio: normalizes gross exposure, applies per-asset caps, and optionally rescales by inverse volatility.
2. **Risk overlay** is a chain of independent rules, each able to cut or zero a weight:
   - Volatility targeting: scale the book to a target annual volatility (e.g. 10%).
   - Stop-loss and take-profit per position, measured from entry price.
   - Regime filter: reduce exposure when a stress series (e.g. VIX) is above a threshold.
   - Drawdown brake: cut exposure after a portfolio drawdown limit is breached.
3. **Cost model** charges each trade a fixed rate in bps on traded notional, plus an optional spread term. Turnover is logged so results can be re-priced at any cost level.
4. **Backtest engine** steps through rebalance dates, calls the strategy with data truncated at that date, applies steps 1 to 3, and holds weights until the next rebalance. A signal at close t is traded at close t and earns returns from t+1, never at a past price. Trading at the next open is planned with the intraday legs.

| Component | Input | Output |
| --- | --- | --- |
| `PortfolioConstructor.apply` | target weights, returns history | constrained weights |
| `RiskOverlay.apply` | weights, prices, portfolio state | adjusted weights |
| `CostModel.cost` | old weights, new weights | cost as a return deduction |
| `Backtester.run` | strategy, data, config | equity curve, weights, trades, turnover |

## Validation protocol

Every strategy is judged by the same protocol, replacing the paper's single 60/40 split with shuffled 10-fold cross-validation.

- **Walk-forward.** Refit on an expanding window every 12 months; trade the following 12 months out of sample.
- **Time-ordered tuning.** Hyperparameters are tuned with `TimeSeriesSplit` inside each training window only.
- **Locked holdout.** The last 2 years are never used for tuning or design choices and are run once at the end.
- **Cost sensitivity.** Every report shows results at 0, 5, 10, 20 and 50 bps per trade.
- **Common benchmark.** Buy-and-hold SPY, and buy-and-hold of the strategy's own universe.
- **Regime split.** Metrics by VIX regime (below and above 25) and for named crises (2008, 2020, 2022).
- **Trial count.** The number of configurations tried is logged, so a high Sharpe can be weighed against how hard it was searched for.

Each run writes its config, data version, code version and metrics to a run folder (a JSON record in the skeleton; MLflow later). Metrics: CAGR, annual volatility, Sharpe, Sortino, max drawdown, Calmar, hit rate, turnover and cost drag.

## Code skeleton

The `etfsys` package implements this design end to end: 21 offline tests pass, and `run_example.py` runs three strategies through the shared stack in about 1 minute on synthetic data, or on real data with `--data real`. Synthetic numbers prove the plumbing, not any market edge.

```
etfsys/
  data.py          MarketData: point-in-time tables, truncate(t), resample(freq)
  data_sources/    yfinance + FRED/ALFRED -> Parquet -> validated MarketData
  features.py      FactorExtractor (PCA), zigzag_labels, breakout, ema_ratio, momentum
  strategy.py      Strategy base class
  strategies/      svm_factor.py, momentum.py, buy_and_hold.py, _template.py
  portfolio.py     PortfolioConstructor
  risk.py          RiskOverlay, VolTarget, StopLossTakeProfit, RegimeFilter, DrawdownBrake
  costs.py         CostModel
  backtest.py      Backtester, walk-forward refits, BacktestResult
  metrics.py       CAGR, Sharpe, Sortino, max drawdown, Calmar, regime split
  tracking.py      save_run(): config, metrics and curves per run
run_example.py
tests/test_core.py, tests/test_data_sources.py
```

The tests pin down the guarantees the design depends on: the engine never shows a strategy data after its decision date, PCA loadings come only from the training window, buy-and-hold reproduces the asset's return minus entry cost, costs scale with turnover, and a stop closes a falling position within one day of the threshold.

Adding a strategy means copying `strategies/_template.py`, filling `generate` (and `fit` if it learns), and passing it to `Backtester.run`. Exits, caps and costs are configured per run, not inside the strategy.

## Roadmap and open decisions

Steps 1 to 4 are done. Each later step adds one capability and has a clear exit test.

1. **Interface and engine** (done): `Strategy`, `Backtester`, cost model, no-look-ahead tests.
2. **SVM plug-in** (done): Baek et al. model with train-only PCA, release lag and time-ordered CV.
3. **Shared risk and portfolio layers** (done): vol target, stops, regime filter, drawdown brake, caps.
4. **Real data adapter** (done): yfinance + FRED/ALFRED to validated `MarketData`. Exit test: `validate()` passes and SPY's adjusted close tracks the S&P 500 total return index, net of SPY's fee, within 0.1% a year.
5. **Intraday legs**: split each day into overnight and daytime returns in the engine. Exit test: reproduce the sign of the Salotra et al. overnight result on SPY.
6. **Momentum at scale**: survivorship-free universe of listed ETFs, liquidity filter, take-profit rule. Exit test: Sharpe and drawdown reported at 20 bps, as in Magner & Sanhueza.
7. **RL allocator**: wrap the backtest engine as a training environment.
8. **Tracking and combiner**: MLflow runs, then a strategy blender with its own walk-forward weights.

| Decision | Options | Default in skeleton |
| --- | --- | --- |
| Execution price | Close of decision day, or next open | Close of decision day |
| SVM target asset | Labels from the S&P 500 for all ETFs, or labels per ETF | Same ticker as traded |
| Short selling | Long/cash, or long/short | Long/cash |
| Refit cadence | 12 months expanding, or rolling window | 12 months expanding |
| Holdout | Last 2 years locked, or none | Not yet enforced in code |

### Sources

- [Baek, Lee, Uctum & Oh (2020), Robo-Advisors: Machine Learning in Trend-Following ETF Investments](https://doi.org/10.3390/su12166399)
- [Magner & Sanhueza (2026), Momentum Strategies in ETFs under Simple Operational Rules](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7379989)
- [Salotra et al. (2026), Overnight vs. Daytime Strategies Across Sector ETFs](https://doi.org/10.3390/risks14040084)
- [Bendatu & Lu (2026), Portfolio Reinforcement Learning with Scenario-Context Rollout](https://arxiv.org/abs/2602.24037)
- [Rompotis (2026), Active vs Passive Management in the AI ETFs](https://www.preprints.org/manuscript/202602.0167/v1)
- [van Staden et al. (2025), Making Leveraged ETFs Work for your Portfolio](https://arxiv.org/abs/2506.19200)
