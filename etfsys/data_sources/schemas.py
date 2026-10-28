"""Pandera contracts for every stored table. Loading code validates before use."""

from __future__ import annotations

import pandera.pandas as pa

_price = pa.Column(float, pa.Check.gt(0))

RAW_PRICES = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]"),
        "ticker": pa.Column(str),
        "open": pa.Column(float, pa.Check.gt(0), nullable=True),
        "high": pa.Column(float, pa.Check.gt(0), nullable=True),
        "low": pa.Column(float, pa.Check.gt(0), nullable=True),
        "close": pa.Column(float, pa.Check.gt(0), nullable=True),
        "adj_close": pa.Column(float, pa.Check.gt(0), nullable=True),
        "volume": pa.Column(float, pa.Check.ge(0), nullable=True),
    },
    unique=["date", "ticker"],
    strict=True,
    coerce=True,
)

PRICES = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]"),
        "ticker": pa.Column(str),
        "open": _price,
        "high": _price,
        "low": _price,
        "close": _price,
        "volume": pa.Column("int64", pa.Check.ge(0)),
    },
    checks=[
        pa.Check(
            lambda d: d["high"] >= d[["open", "close"]].max(axis=1), error="high below open/close"
        ),
        pa.Check(
            lambda d: d["low"] <= d[["open", "close"]].min(axis=1), error="low above open/close"
        ),
    ],
    unique=["date", "ticker"],
    strict=True,
    coerce=True,
)

MACRO_LONG = pa.DataFrameSchema(
    {
        "name": pa.Column(str),
        "obs_date": pa.Column("datetime64[ns]"),
        "known_date": pa.Column("datetime64[ns]"),
        "value": pa.Column(float),
        "method": pa.Column(str, pa.Check.isin(["first_release", "earliest_vintage", "lag"])),
    },
    # The core point-in-time guarantee: nothing is known before the period it measures.
    checks=pa.Check(lambda d: d["known_date"] >= d["obs_date"], error="known before observed"),
    unique=["name", "obs_date"],
    strict=True,
    coerce=True,
)
