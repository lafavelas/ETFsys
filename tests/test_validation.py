"""Validation protocol: holdout lock, point-in-time regimes, trial counter, deflated Sharpe."""

import json

import numpy as np
import pandas as pd
import pytest

from etfsys import BacktestConfig, Backtester, MarketData, make_synthetic_data
from etfsys.data import HoldoutError
from etfsys.report import format_report, validation_report
from etfsys.strategies import BuyAndHold, MultiHorizonMomentum
from etfsys.study import Study
from etfsys.tracking import save_run
from etfsys.validation import crisis_table, deflated_sharpe, expected_max_sharpe, vix_regime

HOLDOUT = pd.Timestamp("2019-01-02")


@pytest.fixture(scope="module")
def data():
    d = make_synthetic_data(start="2015-01-01", end="2019-12-31", seed=1)
    return MarketData(**{**vars(d), "holdout_start": HOLDOUT}).validate()


# ------------------------------------------------------------------- holdout
def test_locked_data_contains_nothing_from_the_holdout(data):
    locked = data.lock_holdout()
    assert locked.holdout_locked and not data.holdout_locked
    for name in ("close", "open", "high", "low", "macro"):
        assert getattr(locked, name).index.max() < HOLDOUT
    assert locked.truncate("2018-06-01").holdout_start == HOLDOUT  # survives truncation


def test_backtest_refuses_to_enter_the_holdout(data):
    with pytest.raises(HoldoutError):
        Backtester(data, BacktestConfig(start="2016-01-04")).run(BuyAndHold())
    ok = Backtester(data, BacktestConfig(start="2016-01-04", end="2018-12-31")).run(BuyAndHold())
    assert not ok.holdout_touched


def test_final_holdout_run_is_allowed_and_flagged(data, tmp_path):
    bt = Backtester(data, BacktestConfig(start="2016-01-04", final_holdout_run=True))
    res = bt.run(BuyAndHold())
    assert res.holdout_touched
    record = json.loads((save_run(res, BuyAndHold(), bt, root=tmp_path) / "run.json").read_text())
    assert record["holdout_touched"] is True
    assert record["data"]["holdout_start"].startswith("2019-01-02")


# ------------------------------------------------------------------- regimes
def test_vix_regime_uses_only_the_previous_close():
    days = pd.bdate_range("2020-03-02", periods=5)
    macro = pd.DataFrame({"vix": [20.0, 20.0, 40.0, 40.0, 20.0]}, index=days)
    close = pd.DataFrame({"SPY": np.linspace(100, 104, 5)}, index=days)
    d = MarketData(close=close, macro=macro).validate()
    label = vix_regime(d, days, threshold=25)
    assert pd.isna(label.iloc[0])  # nothing known before the first day
    assert label.iloc[2] == "calm"  # VIX jumps on day 2, but day-2 returns were held from day 1
    assert label.iloc[3] == "stress"


def test_vix_regime_ignores_values_released_later():
    days = pd.bdate_range("2020-03-02", periods=4)
    # A value released on day 3 must not label days 1-3, whatever period it describes.
    macro = pd.DataFrame({"vix": [15.0, 50.0]}, index=[days[0], days[3]])
    d = MarketData(close=pd.DataFrame({"SPY": [1.0, 1.0, 1.0, 1.0]}, index=days), macro=macro)
    assert list(vix_regime(d.validate(), days).iloc[1:]) == ["calm", "calm", "calm"]


def test_crisis_table_reports_the_covered_part_only():
    r = pd.Series(0.01, index=pd.bdate_range("2008-01-02", "2008-01-31"))
    t = crisis_table(
        {"x": r}, {"gfc": ("2007-10-09", "2009-03-09"), "later": ("2020-02-19", "2020-03-23")}
    )
    assert list(t.index) == [("x", "gfc")]
    assert t.loc[("x", "gfc"), "from"] == "2008-01-02"


# ------------------------------------------------------- selection bias / trials
def test_deflated_sharpe_falls_as_trials_rise():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 2500))
    few = deflated_sharpe(r, list(rng.normal(0, 0.02, 3)))
    many = deflated_sharpe(r, list(rng.normal(0, 0.02, 300)))
    assert 0 <= many < few <= 1
    assert expected_max_sharpe([0.05]) == 0.0  # one trial: nothing to deflate


def test_study_counts_distinct_configs_and_only_appends(data, tmp_path):
    study = Study("unit", root=tmp_path)
    cfg = BacktestConfig(start="2016-01-04", end="2018-12-31")
    for top_n in (2, 3, 2):  # the second top_n=2 run is a repeat, not a new trial
        strat = MultiHorizonMomentum(top_n=top_n)
        bt = Backtester(data, cfg)
        res = bt.run(strat)
        before = study.path.read_text() if study.path.exists() else ""
        save_run(res, strat, bt, root=tmp_path / "runs", study=study)
        assert study.path.read_text().startswith(before)  # history never rewritten
    assert len(study.path.read_text().splitlines()) == 3
    assert study.n_trials == 2
    assert 0 <= study.deflated_sharpe(res) <= 1


def test_report_requires_the_benchmark_and_prints_every_section(data):
    cfg = BacktestConfig(start="2016-01-04", end="2018-12-31")
    bench = Backtester(data, cfg).run(BuyAndHold(name="buy_hold_SPY"))
    mom = Backtester(data, cfg).run(MultiHorizonMomentum())
    with pytest.raises(ValueError, match="benchmark"):
        validation_report({mom.name: mom}, data)
    rep = validation_report({bench.name: bench, mom.name: mom}, data, vix_threshold=15)
    text = format_report(rep)
    for section in ("net of costs", "Cost sensitivity", "Regimes", "Crises", "deflated Sharpe"):
        assert section in text
    assert "locked" in rep["header"]
    assert rep["metrics"].loc["buy_hold_SPY", "excess_cagr_vs_benchmark"] == 0
