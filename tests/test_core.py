"""Core guarantees of the framework. Run with: pytest -q"""

import numpy as np
import pandas as pd
import pytest

from etfsys import (
    BacktestConfig,
    Backtester,
    CostModel,
    MarketData,
    PortfolioConstructor,
    RiskOverlay,
    StopLossTakeProfit,
    Strategy,
    make_synthetic_data,
)
from etfsys.features import FactorExtractor, zigzag_labels
from etfsys.strategies import BuyAndHold


@pytest.fixture(scope="module")
def data():
    return make_synthetic_data(start="2015-01-01", end="2019-12-31", seed=1)


def test_zigzag_marks_up_and_down_segments():
    p = pd.Series([100, 102, 105, 108, 104, 100, 97, 99, 103, 107])
    lab = zigzag_labels(p, threshold=0.03)
    assert list(lab.iloc[1:4]) == [1, 1, 1]  # rise to 108
    assert list(lab.iloc[4:7]) == [-1, -1, -1]  # fall to 97
    assert list(lab.iloc[7:]) == [1, 1, 1]  # rise again


def test_truncate_hides_the_future(data):
    t = data.close.index[500]
    cut = data.truncate(t)
    assert cut.close.index.max() == t
    assert cut.macro.index.max() <= t


class _Spy(Strategy):
    """Records the latest date it was ever shown; must never exceed t."""

    frequency = "W"
    name = "spy_on_dates"

    def __init__(self):
        self.violations = 0

    def fit(self, data, train_end):
        self.violations += int(data.close.index.max() > train_end)

    def generate(self, data, t):
        self.violations += int(data.close.index.max() > t)
        return pd.Series({"SPY": 1.0})


def test_engine_never_leaks_future_data(data):
    spy = _Spy()
    Backtester(data, BacktestConfig(start="2016-01-01")).run(spy)
    assert spy.violations == 0


def test_buy_and_hold_matches_asset_return_minus_entry_cost(data):
    cfg = BacktestConfig(start="2016-01-04", cost=CostModel(commission_bps=10, spread_bps=0))
    res = Backtester(data, cfg).run(BuyAndHold({"SPY": 1.0}))
    entry = res.weights["SPY"].gt(0).idxmax()  # first monthly rebalance date
    px = data.close["SPY"]
    expected = px.iloc[-1] / px.loc[entry] * (1 - 10 / 1e4)
    # monthly rebalances back to 100% trade nothing for a single asset
    assert res.equity().iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_costs_scale_with_turnover():
    cm = CostModel(commission_bps=5, spread_bps=2)
    old = pd.Series({"A": 0.5, "B": 0.5})
    new = pd.Series({"A": 0.0, "B": 0.5, "C": 0.5})
    assert cm.turnover(old, new) == pytest.approx(1.0)
    assert cm.cost(old, new) == pytest.approx(6 / 1e4)


def test_stop_loss_closes_a_falling_position():
    idx = pd.bdate_range("2020-01-01", periods=40)
    falling = pd.Series(np.linspace(100, 80, 40), index=idx)
    close = pd.DataFrame({"X": falling})
    md = MarketData(close=close).validate()
    risk = RiskOverlay([StopLossTakeProfit(stop=0.05)])
    res = Backtester(md, BacktestConfig(start=str(idx[0].date())), risk=risk).run(
        BuyAndHold({"X": 1.0})
    )
    w = res.weights["X"]
    entry = w.gt(0).idxmax()
    stopped = w.loc[entry:].eq(0).idxmax()
    assert stopped > entry
    loss_at_stop = falling.loc[stopped] / falling.loc[entry] - 1
    assert -0.07 < loss_at_stop <= -0.05


def test_factor_loadings_depend_only_on_training_window(data):
    m = data.resample("M").macro
    train = m.loc[:"2017-12-31"]
    a = FactorExtractor(3).fit(train)
    before = a.loadings().copy()
    a.transform(m)  # transforming later data must not refit
    pd.testing.assert_frame_equal(before, a.loadings())


def test_portfolio_caps_gross_and_shorts():
    idx = pd.bdate_range("2020-01-01", periods=5)
    md = MarketData(close=pd.DataFrame({"A": 1.0, "B": 1.0}, index=idx))
    from etfsys import Context

    ctx = Context(
        t=idx[-1], data=md, weights=pd.Series(0.0, index=["A", "B"]), equity=1, peak_equity=1
    )
    w = PortfolioConstructor(max_gross=1.0, long_only=True).apply(
        pd.Series({"A": 1.5, "B": -0.5}), ctx
    )
    assert w["B"] == 0 and w.abs().sum() == pytest.approx(1.0)
