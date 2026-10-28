"""FRED / ALFRED macro download with point-in-time dating.

Talks to the FRED REST API directly instead of through fredapi: fredapi drops the
release date of initial-release observations, and its all-vintages call times out on
weekly series that are revised every week (NFCI). Every row leaves this module with a
`known_date`, the first date the value could have been used.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from pandas.tseries.offsets import BDay, MonthEnd

from .config import MACRO, MacroSeries

BASE = "https://api.stlouisfed.org/fred/"
FAR_PAST, FAR_FUTURE = "1776-07-04", "9999-12-31"


class FredError(RuntimeError):
    pass


def _key_from_env() -> str:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise FredError("FRED_API_KEY is not set: copy .env.example to .env and fill it in")
    return key


class FredClient:
    """Minimal JSON client with retries on server errors. Never puts the key in errors."""

    def __init__(self, api_key: str | None = None, retries: int = 4, timeout: float = 120):
        self._key = api_key or _key_from_env()
        self.retries = retries
        self.timeout = timeout

    def get(self, path: str, **params) -> dict:
        query = urlencode({**params, "api_key": self._key, "file_type": "json"})
        where = f"{path} {params.get('series_id', '')}".strip()
        for attempt in range(self.retries):
            try:
                with urlopen(f"{BASE}{path}?{query}", timeout=self.timeout) as resp:
                    return json.load(resp)
            except HTTPError as e:
                if e.code < 500 or attempt == self.retries - 1:
                    raise FredError(f"{where}: HTTP {e.code}") from None
            except (URLError, TimeoutError) as e:
                if attempt == self.retries - 1:
                    raise FredError(f"{where}: {type(e).__name__}") from None
            time.sleep(2**attempt)
        raise FredError(where)  # pragma: no cover


# ----------------------------------------------------------------- raw fetches
def _observations(payload: dict, with_release: bool = False) -> pd.DataFrame:
    obs = pd.DataFrame(payload["observations"])
    cols = ["date", "realtime_start", "value"] if with_release else ["date", "value"]
    if obs.empty:
        obs = pd.DataFrame(columns=cols)
    out = obs[cols].rename(columns={"date": "obs_date"})
    out["obs_date"] = pd.to_datetime(out["obs_date"])
    if with_release:
        out["realtime_start"] = pd.to_datetime(out["realtime_start"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")  # FRED writes "." for missing
    return out.dropna(subset=["value"]).reset_index(drop=True)


def fetch_latest(client: FredClient, series_id: str) -> pd.DataFrame:
    """Current vintage: obs_date, value."""
    return _observations(client.get("series/observations", series_id=series_id))


def fetch_first_vintage_date(client: FredClient, series_id: str) -> pd.Timestamp:
    d = client.get("series/vintagedates", series_id=series_id, limit=1, sort_order="asc")
    return pd.Timestamp(d["vintage_dates"][0])


def fetch_vintage(client: FredClient, series_id: str, vintage: pd.Timestamp) -> pd.DataFrame:
    """The whole series as published on `vintage`: obs_date, value."""
    day = vintage.strftime("%Y-%m-%d")
    payload = client.get(
        "series/observations", series_id=series_id, realtime_start=day, realtime_end=day
    )
    return _observations(payload)


def fetch_initial_releases(
    client: FredClient, series_id: str, start: pd.Timestamp, chunk_years: int = 2
) -> pd.DataFrame:
    """First-release value and release date of every observation ALFRED archived.

    FRED's initial-release mode (output_type=4) skips observations already in the first
    archived vintage, so these rows all have a real release date. Queried in chunks:
    one call over a weekly, always-revised series times out.
    """
    parts = []
    lo = pd.Timestamp(start)
    today = pd.Timestamp.today().normalize()
    while lo <= today:
        hi = lo + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1)
        payload = client.get(
            "series/observations",
            series_id=series_id,
            output_type=4,
            realtime_start=FAR_PAST,
            realtime_end=FAR_FUTURE,
            observation_start=lo.strftime("%Y-%m-%d"),
            observation_end=hi.strftime("%Y-%m-%d"),
        )
        parts.append(_observations(payload, with_release=True))
        lo = hi + pd.Timedelta(days=1)
    parts = [p for p in parts if not p.empty]
    if not parts:
        return _observations({"observations": []}, with_release=True)
    out = pd.concat(parts, ignore_index=True)
    return (
        out.sort_values(["obs_date", "realtime_start"])
        .drop_duplicates("obs_date")
        .reset_index(drop=True)
    )


# ------------------------------------------------------------ point-in-time rules
def period_end(obs_date: pd.Series, freq: str) -> pd.Series:
    """FRED dates monthly observations on the 1st; the data covers the whole month.
    Weekly series are dated on the last day of their week, daily ones on the day."""
    return obs_date + MonthEnd(0) if freq == "M" else obs_date


def vintage_point_in_time(
    initial: pd.DataFrame, earliest: pd.DataFrame, freq: str, lag_bdays: int
) -> pd.DataFrame:
    """Merge first releases (real release dates) with pre-ALFRED history (fallback lag).

    Pre-ALFRED values come from the earliest archived vintage, so they include revisions
    made before archiving began. Their dates are conservative; their values are not
    first releases. `docs/data_dictionary.md` records this limit.
    """
    cutoff = initial["obs_date"].min() if not initial.empty else pd.Timestamp.max
    old = earliest[earliest["obs_date"] < cutoff].copy()
    old["known_date"] = period_end(old["obs_date"], freq) + BDay(lag_bdays)
    old["method"] = "earliest_vintage"
    new = initial.rename(columns={"realtime_start": "known_date"}).copy()
    new["method"] = "first_release"
    cols = ["obs_date", "known_date", "value", "method"]
    return (
        pd.concat([old[cols], new[cols]], ignore_index=True)
        .sort_values("obs_date")
        .reset_index(drop=True)
    )


def lagged_point_in_time(latest: pd.DataFrame, lag_bdays: int) -> pd.DataFrame:
    out = latest.copy()
    out["known_date"] = out["obs_date"] + BDay(lag_bdays)
    out["method"] = "lag"
    return out[["obs_date", "known_date", "value", "method"]]


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """One series as-is, or the spread first minus second on common dates."""
    if len(frames) == 1:
        return frames[0]
    a, b = frames
    m = a.merge(b, on="obs_date", suffixes=("_a", "_b"))
    return pd.DataFrame({"obs_date": m["obs_date"], "value": m["value_a"] - m["value_b"]})


# ------------------------------------------------------------------ orchestration
def download_series(client: FredClient, spec: MacroSeries, raw_dir: Path) -> pd.DataFrame:
    """Fetch one indicator, save raw responses under `raw_dir`, return point-in-time rows."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if spec.method == "vintage":
        (sid,) = spec.fred_ids
        first = fetch_first_vintage_date(client, sid)
        initial = fetch_initial_releases(client, sid, first - pd.DateOffset(years=1))
        earliest = fetch_vintage(client, sid, first)
        initial.to_parquet(raw_dir / f"{sid}_initial.parquet", index=False)
        earliest.assign(vintage=first).to_parquet(raw_dir / f"{sid}_vintage0.parquet", index=False)
        rows = vintage_point_in_time(initial, earliest, spec.freq, spec.lag_bdays)
    else:
        frames = []
        for sid in spec.fred_ids:
            latest = fetch_latest(client, sid)
            latest.to_parquet(raw_dir / f"{sid}.parquet", index=False)
            frames.append(latest)
        rows = lagged_point_in_time(combine(frames), spec.lag_bdays)
    rows.insert(0, "name", spec.name)
    return rows


def download_macro(
    client: FredClient,
    raw_dir: Path,
    specs: tuple[MacroSeries, ...] = MACRO,
    start: str | None = None,
) -> pd.DataFrame:
    """Long table: name, obs_date, known_date, value, method."""
    out = pd.concat([download_series(client, s, raw_dir) for s in specs], ignore_index=True)
    if start is not None:
        out = out[out["obs_date"] >= pd.Timestamp(start)]
    return out.reset_index(drop=True)


def macro_wide(long: pd.DataFrame) -> pd.DataFrame:
    """date x indicator, indexed by KNOWN date: each cell is the newest observation
    known that day. Rows that only re-date an older observation are dropped."""
    cols = {}
    for name, g in long.groupby("name", sort=False):
        g = g.sort_values(["known_date", "obs_date"])
        prev = g["obs_date"].cummax().shift(1)
        newer = prev.isna() | (g["obs_date"] > prev)
        g = g[newer].drop_duplicates("known_date", keep="last")
        cols[name] = g.set_index("known_date")["value"]
    wide = pd.concat(cols, axis=1).sort_index()
    wide.index.name = "date"
    return wide
