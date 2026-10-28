"""Data layer: one point-in-time container handed to every downstream module.

Strategies never read raw sources. They receive a `MarketData` already truncated
at the decision date by the backtest engine, so look-ahead is impossible by
construction (as long as tables are indexed by the date the value became known).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

FREQ_PERIOD = {"D": "D", "W": "W-FRI", "M": "M"}


def period_last(df: pd.DataFrame, freq: str, how: str = "last") -> pd.DataFrame:
    """Resample to `freq`, indexed by the LAST TRADING DATE of each period.

    Calendar month-ends (pandas 'ME') can fall on weekends; keeping the real
    trading date lets monthly decisions line up with the daily backtest clock.
    """
    if freq == "D":
        return df
    periods = df.index.to_period(FREQ_PERIOD[freq])
    grouped = df.groupby(periods)
    out = getattr(grouped, how)()
    last_dates = df.index.to_series().groupby(periods).max()
    out.index = pd.DatetimeIndex(last_dates.loc[out.index].values, name=df.index.name)
    return out


@dataclass
class MarketData:
    """Aligned, point-in-time market data.

    close/open/high/low : date x ticker, daily; NaN only before a fund's first bar
    volume              : date x ticker, shares traded (optional)
    macro               : date x indicator, indexed by RELEASE date
    meta                : ticker x attributes (category, asset_class, leverage, ...)
    nav                 : date x ticker (optional, for premium strategies)
    """

    close: pd.DataFrame
    open: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None
    macro: pd.DataFrame | None = None
    meta: pd.DataFrame | None = None
    nav: pd.DataFrame | None = None
    as_of: pd.Timestamp | None = field(default=None)

    # ------------------------------------------------------------------ checks
    def validate(self) -> MarketData:
        if not isinstance(self.close.index, pd.DatetimeIndex):
            raise TypeError("close must have a DatetimeIndex")
        if not self.close.index.is_monotonic_increasing:
            raise ValueError("close index must be sorted ascending")
        if self.close.index.has_duplicates:
            raise ValueError("close index has duplicate dates")
        started = self.close.notna().cummax()
        gaps = (started & self.close.isna()).sum()
        if gaps.any():
            raise ValueError(f"close has NaN after first price: {gaps[gaps > 0].to_dict()}")
        for name in ("open", "high", "low", "volume", "nav"):
            tbl = getattr(self, name)
            if tbl is not None:
                if not tbl.index.equals(self.close.index):
                    raise ValueError(f"{name} index does not match close")
                missing = set(tbl.columns) ^ set(self.close.columns)
                if missing:
                    raise ValueError(f"{name} tickers differ from close: {sorted(missing)}")
        if self.macro is not None:
            idx = self.macro.index
            if (
                not isinstance(idx, pd.DatetimeIndex)
                or not idx.is_monotonic_increasing
                or idx.has_duplicates
            ):
                raise ValueError("macro needs a sorted, unique DatetimeIndex of release dates")
        if self.meta is not None:
            unknown = set(self.close.columns) - set(self.meta.index)
            if unknown:
                raise ValueError(f"tickers missing from meta: {sorted(unknown)}")
        return self

    def has(self, table: str) -> bool:
        return getattr(self, table, None) is not None

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    # ------------------------------------------------------- point-in-time ops
    def truncate(self, t: pd.Timestamp) -> MarketData:
        """Everything known at the close of date t, and nothing after."""
        t = pd.Timestamp(t)

        def cut(df):
            return None if df is None else df.loc[:t]

        return replace(
            self,
            close=cut(self.close),
            open=cut(self.open),
            high=cut(self.high),
            low=cut(self.low),
            volume=cut(self.volume),
            macro=cut(self.macro),
            nav=cut(self.nav),
            as_of=t,
        )

    def resample(self, freq: str) -> MarketData:
        """Weekly/monthly view. Macro is forward-filled onto the price calendar first."""
        if freq == "D":
            return self
        macro = None
        if self.macro is not None:
            macro = self.macro.reindex(self.close.index.union(self.macro.index)).ffill()
            macro = period_last(macro.loc[self.close.index], freq)
        return replace(
            self,
            close=period_last(self.close, freq, "last"),
            open=None if self.open is None else period_last(self.open, freq, "first"),
            high=None if self.high is None else period_last(self.high, freq, "max"),
            low=None if self.low is None else period_last(self.low, freq, "min"),
            volume=None if self.volume is None else period_last(self.volume, freq, "sum"),
            nav=None if self.nav is None else period_last(self.nav, freq, "last"),
            macro=macro,
        )


# ---------------------------------------------------------------------------
# Synthetic data so the skeleton runs end to end without a data vendor.
# Replace with your Data Gathering / Data Management output.
# ---------------------------------------------------------------------------
def make_synthetic_data(start="2002-01-01", end="2025-12-31", seed=7) -> MarketData:
    """Daily ETF prices + 16 monthly macro indicators driven by a hidden stress regime.

    The regime makes equity drift negative and volatility high in stress, and pushes
    the macro indicators (spreads, VIX-like series) up, so the SVM factor model has a
    real but noisy signal to find. Numbers are illustrative, not calibrated.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # Hidden 2-state Markov regime: 0 = calm, 1 = stress
    p_stay = {0: 0.995, 1: 0.98}
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        s = regime[i - 1]
        regime[i] = s if rng.random() < p_stay[s] else 1 - s
    stress = regime.astype(float)

    mkt_mu = np.where(stress == 1, -0.0012, 0.0006)
    mkt_sig = np.where(stress == 1, 0.022, 0.009)
    mkt = mkt_mu + mkt_sig * rng.standard_normal(n)

    spec = {
        # ticker: (beta to market, idio vol, extra drift, category, asset_class)
        "SPY": (1.00, 0.0005, 0.0, "US Large Cap", "Equity"),
        "IVV": (1.00, 0.0005, 0.0, "US Large Cap", "Equity"),
        "VTI": (1.02, 0.0015, 0.0, "US Total Market", "Equity"),
        "QQQ": (1.20, 0.0060, 0.0001, "US Growth", "Equity"),
        "XLK": (1.15, 0.0070, 0.0001, "Sector", "Equity"),
        "XLE": (0.90, 0.0120, 0.0, "Sector", "Equity"),
        "XLF": (1.20, 0.0090, -0.00005, "Sector", "Equity"),
        "XLV": (0.75, 0.0060, 0.0001, "Sector", "Equity"),
        "XLU": (0.55, 0.0070, 0.00005, "Sector", "Equity"),
        "XLP": (0.60, 0.0050, 0.00005, "Sector", "Equity"),
        "TLT": (-0.30, 0.0080, 0.0001, "Treasury", "Bond"),
        "GLD": (0.05, 0.0100, 0.0002, "Commodity", "Commodity"),
    }
    rets, meta_rows = {}, []
    for tk, (beta, ivol, drift, cat, ac) in spec.items():
        r = beta * mkt + drift + ivol * rng.standard_normal(n)
        if tk == "TLT":  # flight to quality in stress
            r = r + 0.0008 * stress
        rets[tk] = r
        meta_rows.append({"ticker": tk, "category": cat, "asset_class": ac, "leverage": 1.0})
    rets = pd.DataFrame(rets, index=dates)
    close = 100 * (1 + rets).cumprod()

    gap = 0.4  # share of daily move that happens overnight
    prev = close.shift(1).fillna(close.iloc[0])
    open_ = prev * (1 + gap * rets)
    span = (close - open_).abs()
    noise = pd.DataFrame(
        np.abs(rng.standard_normal(close.shape)), index=dates, columns=close.columns
    )
    high = pd.concat([open_, close]).groupby(level=0).max() + span * 0.3 * noise
    low = pd.concat([open_, close]).groupby(level=0).min() - span * 0.3 * noise

    # Monthly macro indicators, released at month end and loading on smoothed stress
    month_end = period_last(pd.DataFrame({"s": stress}, index=dates), "M", "mean")["s"]
    smooth = month_end.ewm(alpha=0.6).mean().values
    m = len(month_end)
    names = [
        "lead_credit",
        "gs_fci",
        "bbg_fci",
        "cfnai",
        "nfci",
        "kc_stress",
        "ted",
        "libor_ois",
        "cp_tbill",
        "baa_tsy",
        "hy_oas",
        "muni_tsy",
        "vix",
        "vxo",
        "vxn",
        "vxd",
    ]
    sign = {"bbg_fci": -1, "cfnai": -1, "nfci": -1}
    macro = {}
    for k, name in enumerate(names):
        load = sign.get(name, 1) * (0.8 + 0.4 * rng.random())
        level = 10 + 5 * rng.random()
        macro[name] = level + 4 * load * smooth + rng.standard_normal(m) * 0.8
    macro = pd.DataFrame(macro, index=month_end.index)

    meta = pd.DataFrame(meta_rows).set_index("ticker")
    return MarketData(
        close=close, open=open_, high=high, low=low, macro=macro, meta=meta
    ).validate()
