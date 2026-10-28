"""Download prices and macro, store raw and processed Parquet, validate, report.

    uv run python -m etfsys.data_sources.refresh [--root data] [--start 2000-01-01]

Re-running overwrites the files it owns in <root>/raw and <root>/processed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .build import SPY_EXPENSE_RATIO, load_market_data, load_reference, tracking_gap
from .config import MACRO, REFERENCE, START, UNIVERSE
from .fred import FredClient, download_macro
from .prices import clean_prices, download_prices
from .schemas import MACRO_LONG, PRICES, RAW_PRICES


def refresh_prices(root: Path, start: str, end: str | None) -> dict:
    raw_dir, out_dir = root / "raw", root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = RAW_PRICES.validate(download_prices([*UNIVERSE, *REFERENCE], start, end))
    raw.to_parquet(raw_dir / "prices.parquet", index=False)
    clean, report = clean_prices(raw)
    clean = PRICES.validate(clean)
    is_ref = clean["ticker"].isin(list(REFERENCE))
    clean[~is_ref].to_parquet(out_dir / "prices.parquet", index=False)
    clean[is_ref].to_parquet(out_dir / "reference.parquet", index=False)
    return report


def refresh_macro(root: Path) -> pd.DataFrame:
    long = download_macro(FredClient(), root / "raw" / "fred", MACRO)
    long = MACRO_LONG.validate(long)
    (root / "processed").mkdir(parents=True, exist_ok=True)
    long.to_parquet(root / "processed" / "macro.parquet", index=False)
    return long


def summary(root: Path) -> None:
    data = load_market_data(root, unlock_holdout=True)  # data-quality check only, no research
    print(
        f"\nprices: {len(data.tickers)} tickers, {data.close.index[0].date()} -> {data.close.index[-1].date()}"
    )
    print(f"locked holdout from {data.holdout_start.date()}: load_market_data() stops before it")
    print(data.meta[["category", "inception"]].to_string())
    long = pd.read_parquet(root / "processed" / "macro.parquet")
    macro = long.groupby("name").agg(
        first_obs=("obs_date", "min"),
        last_obs=("obs_date", "max"),
        rows=("value", "size"),
        first_release_share=("method", lambda m: round(float((m == "first_release").mean()), 2)),
    )
    print("\nmacro (indexed by known date in MarketData):")
    print(macro.to_string())
    ref = load_reference(root)
    gap = tracking_gap(data.close["SPY"], ref["^SP500TR"], SPY_EXPENSE_RATIO)
    print(
        f"\nSPY vs S&P 500 TR after {SPY_EXPENSE_RATIO:.4%} fee: {gap:+.3%} per year (target within 0.1%)"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--skip-prices", action="store_true")
    ap.add_argument("--skip-macro", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root)
    if not args.skip_prices:
        print("prices:", refresh_prices(root, args.start, args.end))
    if not args.skip_macro:
        long = refresh_macro(root)
        print(f"macro: {len(long)} rows, {long['name'].nunique()} indicators")
    summary(root)


if __name__ == "__main__":
    main()
