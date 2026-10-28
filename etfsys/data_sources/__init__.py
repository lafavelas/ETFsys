"""Real data adapter: yfinance prices + FRED/ALFRED macro -> Parquet -> MarketData.

Needs the `data` extra: `uv pip install -e ".[data]"`. Run `python -m
etfsys.data_sources.refresh` once, then `load_market_data()` anywhere.
"""

from .build import load_market_data, load_reference, tracking_gap

__all__ = ["load_market_data", "load_reference", "tracking_gap"]
