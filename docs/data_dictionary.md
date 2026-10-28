# Data dictionary

What `etfsys.data_sources` downloads, how each value is dated, and where it departs from
Baek et al. (2020). Built by `uv run python -m etfsys.data_sources.refresh`; loaded with
`load_market_data()`. Definitions live in `etfsys/data_sources/config.py`.

## Storage

| File | Content | Schema |
| --- | --- | --- |
| `data/raw/prices.parquet` | yfinance as downloaded: unadjusted OHLC, `adj_close`, volume, long format | `RAW_PRICES` |
| `data/raw/fred/<ID>.parquet` | Latest vintage of an unrevised daily series | — |
| `data/raw/fred/<ID>_initial.parquet` | ALFRED first releases: `obs_date`, `realtime_start`, `value` | — |
| `data/raw/fred/<ID>_vintage0.parquet` | The earliest archived vintage of a revised series | — |
| `data/processed/prices.parquet` | Adjusted, cleaned OHLCV for the universe | `PRICES` |
| `data/processed/reference.parquet` | `^SP500TR`, used only for the data check | `PRICES` |
| `data/processed/macro.parquet` | `name, obs_date, known_date, value, method` | `MACRO_LONG` |

`MarketData.macro` is the wide version of `macro.parquet`, indexed by `known_date`.
Each cell holds the newest observation known on that date.

## Prices

- **Universe:** SPY, IVV, VTI, QQQ, XLK, XLE, XLF, XLV, XLU, XLP, TLT, GLD, from 2000-01-03.
  IVV starts 2000-05-19, VTI 2001-06-15, TLT 2002-07-30 and GLD 2004-11-18. Prices are NaN
  before a fund's first bar, and `PortfolioConstructor` never buys a fund without a price.
- **Adjustment:** `close` is yfinance's adjusted close, which includes dividends and splits,
  so close-to-close returns are total returns. `open`, `high` and `low` are multiplied by the
  same daily factor (`adj_close / close`) so that intraday legs stay consistent with `close`.
- **Cleaning:** a bar whose high/low does not bracket its open and close is widened. In the
  first refresh (2026-10-28) that was 143 of 85,366 bars (0.17%); no rows were dropped.
- **Check:** SPY's adjusted close vs. the S&P 500 total return index, after SPY's 0.0945% fee,
  differed by +0.008% a year over 2000-2026 (exit test: within 0.1%).
- **Bias:** yfinance only knows funds that still trade (not survivorship-free). That is
  harmless for this fixed list of large ETFs; flag it for any broad-universe study.

## Macro indicators

| Column | FRED series | Freq | Dating | Paper indicator it covers |
| --- | --- | --- | --- | --- |
| `cfnai` | CFNAI | M | vintage, fallback 20 bd | Chicago Fed National Activity Index |
| `nfci` | NFCI | W | vintage, fallback 5 bd | Chicago Fed NFCI |
| `kc_stress` | KCFSI | M | vintage, fallback 10 bd | Kansas City Fed Financial Stress Index |
| `cp_tbill` | DCPF3M − DTB3 | D | lag 1 bd | 3M commercial paper − T-bill |
| `baa_tsy` | BAA10Y | D | lag 1 bd | Moody's Baa − 10Y Treasury |
| `vix` | VIXCLS | D | lag 1 bd | VIX |
| `vxn` | VXNCLS | D | lag 1 bd | VXN (Nasdaq-100 vol) |
| `vxd` | VXDCLS | D | lag 1 bd | VXD (DJIA vol) |
| `anfci` | ANFCI | W | vintage, fallback 5 bd | **Substitute** for the Goldman Sachs FCI (proprietary) |
| `stlfsi` | STLFSI4 | W | vintage, fallback 5 bd | **Substitute** for the Bloomberg FCI (proprietary) |
| `nfci_credit` | NFCICREDIT | W | vintage, fallback 5 bd | **Substitute** for the Conference Board leading credit index (proprietary) |
| `baa_aaa` | DBAA − DAAA | D | lag 1 bd | **Substitute** for the ICE BofA high-yield OAS |
| `term_spread` | T10Y3M | D | lag 1 bd | **Substitute** for the TED spread and LIBOR–OIS |

bd = business days. Not covered:

- **Municipal − Treasury spread:** FRED has no free daily muni yield series with history. Dropped.
- **VXO:** discontinued in 2021-09 and almost identical to VIX. Dropped.

Why each substitute was chosen:

- **ANFCI** is the NFCI with business-cycle effects removed. Like the Goldman Sachs FCI, it
  measures financial conditions in excess of what the economy explains.
- **STLFSI4** is a broad weekly stress index built from rates, spreads and volatility, close
  in scope to the Bloomberg FCI. Its sign is the reverse of Bloomberg's (higher = more stress).
  The PCA in `FactorExtractor` does not care about sign.
- **NFCICREDIT** is the credit sub-index of the NFCI and covers credit conditions, as the
  Conference Board index does.
- **Baa − Aaa** replaces HY OAS because FRED now keeps only about 3 years of ICE BofA data
  (from 2023-10 in the 2026-10 refresh). It is a credit-quality spread, but investment grade
  only, so it reacts less than high yield.
- **10Y − 3M** stands in for TED (ended 2022-01) and LIBOR–OIS (LIBOR retired). It is a
  term spread, not a funding spread; `cp_tbill` stays the closest funding-stress measure.
  Treat this as the weakest substitute.

The macro table starts in 1967-2001 depending on the series; `vxn` (2001-02) is the latest
to start, so models that need every column begin in 2001.

## Point-in-time rules

Each row gets a `known_date`, the first day the value could have been used. `MarketData`
is indexed by it, so `truncate(t)` only returns values known by t.

- **`vintage` (revised series).** FRED's ALFRED archive keeps each published version.
  - Observations first released after archiving began get their **first-release value** and
    their real release date (`method = first_release`).
  - Older observations exist only in the earliest archived vintage (`method = earliest_vintage`).
    They are dated at period end plus the fallback lag. Each fallback is longer than the 95th
    percentile of the real delays observed in ALFRED:

    | Series | Real delay (median / 95th pct / max, bd) | Fallback |
    | --- | --- | --- |
    | CFNAI | 16 / 19 / 61 | 20 |
    | KCFSI | 6 / 8 / 30 | 10 |
    | NFCI, ANFCI, NFCICREDIT | 3 / 4 / 14 | 5 |
    | STLFSI4 | 4 / 4 / 4 | 5 |

  - Archiving starts at 2010-11 (KCFSI), 2011-05 (CFNAI, NFCI, ANFCI), 2012-07 (NFCICREDIT) and
    2022-11 (STLFSI4). So 57-88% of each series' rows come from the earliest vintage.
- **`lag` (unrevised market series).** The latest values, known one business day after the
  observation date. FRED publishes daily market data the next morning.

**Limits.**

- *Revised history.* Earliest-vintage values include revisions made before archiving began,
  so pre-2011 macro data (pre-2022 for STLFSI4) is slightly cleaner than what was really
  available then. The dates are conservative; the values are not true first releases. If a
  result relies on that period, check it on post-2011 data.
- *Revisions after first release are ignored.* As of any date the table shows the
  first-release value of the newest observation. It never shows the later revised value of
  an older one.
- *SVM macro lag.* Because macro is already release-dated, `SVMFactorStrategy` uses
  `macro_lag_periods=0` on real data. The default of 1 is kept for synthetic data.
