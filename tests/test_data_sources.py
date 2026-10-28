"""Real data adapter: point-in-time dating, cleaning, schemas, MarketData round trip.

All offline: small hand-built frames stand in for yfinance and FRED responses.
"""

from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from etfsys import Context, MarketData, PortfolioConstructor
from etfsys.data_sources import fred, load_market_data, tracking_gap
from etfsys.data_sources.prices import clean_prices, wide_to_long
from etfsys.data_sources.schemas import MACRO_LONG, PRICES

D = pd.Timestamp


# ------------------------------------------------------------------- FRED dating
def _monthly_vintage_case():
    # ALFRED archive starts mid-2011: obs up to 2011-04 exist only in the first vintage.
    earliest = pd.DataFrame(
        {
            "obs_date": pd.to_datetime(["2011-02-01", "2011-03-01", "2011-04-01"]),
            "value": [0.1, 0.2, 0.3],
        }
    )
    initial = pd.DataFrame(
        {
            "obs_date": pd.to_datetime(["2011-05-01", "2011-06-01"]),
            "realtime_start": pd.to_datetime(["2011-06-23", "2011-07-25"]),
            "value": [-0.37, 0.05],
        }
    )
    return fred.vintage_point_in_time(initial, earliest, "M", lag_bdays=20)


def test_vintage_rows_keep_release_dates_and_lag_older_history():
    rows = _monthly_vintage_case()
    first = rows[rows["method"] == "first_release"].set_index("obs_date")
    assert first.loc[D("2011-05-01"), "known_date"] == D("2011-06-23")
    old = rows[rows["method"] == "earliest_vintage"].set_index("obs_date")
    # March data covers the whole month: known 20 business days after 31 March.
    assert old.loc[D("2011-03-01"), "known_date"] == D("2011-03-31") + pd.offsets.BDay(20)
    MACRO_LONG.validate(rows.assign(name="cfnai"))


def test_weekly_and_daily_lags_start_from_the_observation_date():
    latest = pd.DataFrame({"obs_date": pd.to_datetime(["2024-03-08"]), "value": [14.7]})  # a Friday
    rows = fred.lagged_point_in_time(latest, lag_bdays=1)
    assert rows["known_date"].iloc[0] == D("2024-03-11")  # next business day, Monday


def test_spread_uses_common_dates_only():
    a = pd.DataFrame(
        {"obs_date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "value": [5.0, 5.1]}
    )
    b = pd.DataFrame({"obs_date": pd.to_datetime(["2024-01-03"]), "value": [4.0]})
    out = fred.combine([a, b])
    assert list(out["value"]) == pytest.approx([1.1])


def test_macro_value_is_invisible_before_its_release_date():
    rows = _monthly_vintage_case().assign(name="cfnai")
    wide = fred.macro_wide(rows)
    dates = pd.bdate_range("2011-01-03", "2011-08-31")
    close = pd.DataFrame({"SPY": np.linspace(100, 110, len(dates))}, index=dates)
    data = MarketData(close=close, macro=wide).validate()
    before, on = data.truncate("2011-06-22"), data.truncate("2011-06-23")
    assert -0.37 not in before.macro["cfnai"].values  # May value, released 23 June
    assert on.macro["cfnai"].iloc[-1] == -0.37
    monthly = data.truncate("2011-06-30").resample("M").macro["cfnai"]
    assert monthly.loc["2011-05"].iloc[0] != -0.37  # not yet known at the May close


def test_macro_wide_ignores_rows_that_only_redate_older_observations():
    rows = pd.DataFrame(
        {
            "name": "x",
            "obs_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-15"]),
            "known_date": pd.to_datetime(["2020-01-10", "2020-02-10", "2020-03-01"]),
            "value": [1.0, 2.0, 9.0],
            "method": "lag",
        }
    )
    wide = fred.macro_wide(rows)
    assert list(wide["x"]) == [1.0, 2.0]  # the stale 15 Jan row must not replace February


def test_schema_rejects_values_known_before_they_are_observed():
    bad = pd.DataFrame(
        {
            "name": ["x"],
            "obs_date": [D("2020-02-01")],
            "known_date": [D("2020-01-01")],
            "value": [1.0],
            "method": ["lag"],
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        MACRO_LONG.validate(bad)


def test_fred_errors_never_contain_the_api_key(monkeypatch):
    secret = "abc123secret"

    def boom(url, timeout):
        raise HTTPError(url, 400, "Bad Request", hdrs=None, fp=None)

    monkeypatch.setattr(fred, "urlopen", boom)
    client = fred.FredClient(api_key=secret, retries=1)
    with pytest.raises(fred.FredError) as err:
        client.get("series/observations", series_id="VIXCLS")
    assert secret not in str(err.value) and err.value.__cause__ is None


# ------------------------------------------------------------------------ prices
def _yf_frame():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    cols = pd.MultiIndex.from_product(
        [["Adj Close", "Close", "High", "Low", "Open", "Volume"], ["AAA", "BBB"]],
        names=["Price", "Ticker"],
    )
    vals = np.array(
        [
            # adj  adj   close close high  high  low  low  open open vol  vol
            [50.0, np.nan, 100.0, np.nan, 101.0, np.nan, 99.0, np.nan, 100.0, np.nan, 1e6, np.nan],
            [51.0, 20.0, 102.0, 20.0, 101.5, 21.0, 100.0, 19.0, 100.5, 19.5, 2e6, 5e5],
        ]
    )
    return pd.DataFrame(vals, index=pd.Index(dates, name="Date"), columns=cols)


def test_prices_are_adjusted_and_bad_bars_are_widened():
    raw = wide_to_long(_yf_frame())
    assert len(raw) == 3  # BBB has no bar before its launch
    clean, report = clean_prices(raw)
    PRICES.validate(clean)
    aaa = clean[clean["ticker"] == "AAA"].set_index("date")
    assert aaa.loc["2024-01-02", "close"] == 50.0
    assert aaa.loc["2024-01-02", "open"] == pytest.approx(50.0)  # 100 * 0.5 factor
    assert aaa.loc["2024-01-03", "high"] == pytest.approx(51.0)  # high 101.5 < close 102: widened
    assert report["widened_bars"] == 1


def test_price_schema_rejects_inconsistent_bars():
    bad = pd.DataFrame(
        {
            "date": [D("2024-01-02")],
            "ticker": ["AAA"],
            "open": [10.0],
            "high": [9.0],
            "low": [8.0],
            "close": [9.5],
            "volume": [100],
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        PRICES.validate(bad)


# -------------------------------------------------------------------- MarketData
def test_nan_allowed_only_before_a_funds_first_price():
    idx = pd.bdate_range("2024-01-01", periods=4)
    ok = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3], "B": [np.nan, np.nan, 5.0, 5.1]}, index=idx)
    MarketData(close=ok).validate()
    gap = ok.copy()
    gap.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN after first price"):
        MarketData(close=gap).validate()


def test_portfolio_does_not_buy_a_fund_before_launch():
    idx = pd.bdate_range("2024-01-01", periods=3)
    close = pd.DataFrame({"A": [1.0, 1.1, 1.2], "B": [np.nan, np.nan, 5.0]}, index=idx)
    data = MarketData(close=close).validate()
    ctx = Context(
        t=idx[1], data=data, weights=pd.Series(0.0, index=["A", "B"]), equity=1.0, peak_equity=1.0
    )
    w = PortfolioConstructor().apply(pd.Series({"A": 0.5, "B": 0.5}), ctx)
    assert w["B"] == 0.0 and w["A"] == 0.5


def test_load_market_data_round_trip(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    clean, _ = clean_prices(wide_to_long(_yf_frame()))
    clean["ticker"] = clean["ticker"].map({"AAA": "SPY", "BBB": "GLD"})
    clean.to_parquet(processed / "prices.parquet", index=False)
    _monthly_vintage_case().assign(name="cfnai").to_parquet(
        processed / "macro.parquet", index=False
    )

    data = load_market_data(tmp_path)
    assert set(data.tickers) == {"SPY", "GLD"}
    assert data.volume is not None and data.macro.index.is_monotonic_increasing
    assert data.meta.loc["GLD", "category"] == "Commodity"
    assert data.meta.loc["GLD", "inception"] == D("2024-01-03")
    assert data.truncate("2024-01-02").close.index.max() == D("2024-01-02")


def test_tracking_gap_is_zero_when_etf_equals_index_minus_fee():
    idx = pd.date_range("2010-01-01", "2020-01-01", freq="D")
    years = (idx - idx[0]).days / 365.25
    index = pd.Series(1.10**years, index=idx)
    etf = pd.Series((1.10 - 0.001) ** years, index=idx)
    assert tracking_gap(etf, index, 0.001) == pytest.approx(0.0, abs=1e-9)
