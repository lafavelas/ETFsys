# etfsys — multi-strategy ETF research platform

Research and backtest platform for ETF strategies. It generalizes the 7-module robo-advisor
of Baek et al. (2020) so any strategy plugs into shared data, portfolio, risk, cost and
backtest layers. Research only: no live trading, no broker connections.

## Reference documents (read before large changes)

- `docs/architecture.md`: target design (layers, interfaces, validation protocol, roadmap)
- `docs/papers/baek_2020_robo_advisors.pdf`: the source paper for the SVM factor plug-in
- `README.md`: package layout and quick start

## Commands

- Install deps: `uv pip install -e ".[dev]"` (or `uv pip install <pkg>` and add it to pyproject.toml)
- Tests: `uv run pytest -q` (must pass before any commit)
- Lint/format: `uv run ruff check . --fix && uv run ruff format .`
- Example run: `uv run python run_example.py`
- Refresh data: `uv run python -m etfsys.data_sources.refresh` (once phase 1 exists)

## Architecture (do not break these boundaries)

- `data.py`: `MarketData` is the only object strategies receive. Tables are indexed by the
  date a value became KNOWN (release date for macro).
- `strategy.py`: every strategy subclasses `Strategy`, implements `generate(data, t) -> pd.Series`
  of target weights per ticker, and `fit(data, train_end)` if it learns.
- Strategies never size positions, apply stops or compute costs. That is the job of
  `portfolio.py`, `risk.py` and `costs.py`, configured per run.
- `backtest.py` owns the clock. It calls strategies with `data.truncate(t)` only.
- `features.py` holds reusable features and labels. Strategy-specific logic stays in
  `strategies/<name>.py`.
- New strategy = copy `strategies/_template.py`, register it in `strategies/__init__.py`.

## Non-negotiable rules (research validity)

1. No look-ahead. Nothing a strategy uses at date t may depend on data after t. Labels
   (e.g. zigzag) are targets only, never features.
2. Fit scalers, PCA and models on the training window only, then transform later data.
3. Hyperparameter tuning uses `TimeSeriesSplit`, never shuffled k-fold.
4. Macro data is used with its release date or a release lag (`macro_lag_periods`).
5. Every reported result is net of costs and shows the cost-sensitivity table (0–50 bps)
   and the buy-and-hold SPY benchmark.
6. Do not tune on the locked holdout (last 2 years of data) unless explicitly asked.
7. When a result looks too good (Sharpe > 2, monthly win rate > 80%), assume a bug and
   look for leakage before reporting it.

## Data

- Storage: Parquet in `data/raw/` (as downloaded) and `data/processed/` (cleaned).
  `data/` is gitignored. Query with DuckDB if needed.
- Validate every loaded table with the pandera schemas in `etfsys/data_sources/schemas.py`.
- Sources: prices from yfinance (adjusted close, OHLC); macro from FRED via fredapi, using
  ALFRED vintages for point-in-time values. API keys come from `.env` (`FRED_API_KEY`);
  never hard-code or print keys.
- Known gaps vs. the paper: Goldman Sachs FCI, Bloomberg FCI and the Conference Board
  leading credit index are proprietary; TED and LIBOR–OIS are discontinued. Use the FRED
  subset and document substitutes in `docs/data_dictionary.md`.
- yfinance is not survivorship-free. Do not run broad-universe studies on it without
  flagging the bias.

## Code conventions

- Python 3.11+, type hints on public functions, short docstrings saying what and why.
- pandas: `DatetimeIndex` sorted ascending, tickers as columns, weights as `pd.Series`
  indexed by ticker. Daily data is the base; resample with `data.resample(freq)`.
- No hidden randomness: seeds are parameters.
- Every new module or strategy ships with tests in `tests/`. Leakage-sensitive code gets
  a test that proves it only uses past data.
- Keep dependencies minimal; ask before adding a heavy one.

## Workflow

- For tasks touching more than one module, propose a short plan first and wait for approval.
- Work in small steps; run tests after each; commit with a clear message after each green step.
- Do not delete or rewrite files in `data/` or `runs/` without asking.
- Update `README.md` and `docs/architecture.md` when an interface changes.

## Status and roadmap

Done: Strategy interface, backtest engine, cost model, risk overlay, portfolio builder,
SVM factor plug-in, momentum plug-in, buy-and-hold benchmark, synthetic data, 8 tests.

Next (in order, one phase at a time):
1. Real data adapter: yfinance + FRED -> Parquet -> validated `MarketData`; data dictionary.
2. Validation protocol in code: locked holdout, regime report, trial counter per study.
3. Intraday legs in the engine (open/close) + overnight/daytime plug-in (Salotra et al. 2026).
4. Momentum at scale: universe and liquidity filters, take-profit rule (Magner & Sanhueza 2026).
5. Experiment tracking with MLflow; HTML report per run.
6. RL allocator plug-in using the backtest engine as its environment.
7. Strategy combiner with walk-forward weights.

Definition of done for a phase: tests pass, ruff clean, README/architecture updated,
example or notebook shows the new capability, results reported net of costs.
