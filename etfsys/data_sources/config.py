"""What to download: the ETF universe and the macro indicators, with point-in-time rules.

Each macro indicator maps to one or two FRED series and to the Baek et al. (2020)
indicator it stands in for. `docs/data_dictionary.md` explains every substitute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

START = "2000-01-01"

# ticker: (category, asset_class). Same universe as the synthetic data.
UNIVERSE: dict[str, tuple[str, str]] = {
    "SPY": ("US Large Cap", "Equity"),
    "IVV": ("US Large Cap", "Equity"),
    "VTI": ("US Total Market", "Equity"),
    "QQQ": ("US Growth", "Equity"),
    "XLK": ("Sector", "Equity"),
    "XLE": ("Sector", "Equity"),
    "XLF": ("Sector", "Equity"),
    "XLV": ("Sector", "Equity"),
    "XLU": ("Sector", "Equity"),
    "XLP": ("Sector", "Equity"),
    "TLT": ("Treasury", "Bond"),
    "GLD": ("Commodity", "Commodity"),
}

# Reference indices used only to check the price data, never traded.
REFERENCE = {"^SP500TR": "S&P 500 total return index"}


@dataclass(frozen=True)
class MacroSeries:
    """One column of `MarketData.macro`.

    method:
      "vintage" - revised series. Observations released after ALFRED started archiving
                  get their first-release value and real release date. Earlier ones get
                  the earliest archived vintage, known at period end + `lag_bdays`.
      "lag"     - unrevised market series: latest values, known at obs date + `lag_bdays`.
    fred_ids: one series, or two whose difference (first minus second) is the indicator.
    """

    name: str
    fred_ids: tuple[str, ...]
    method: Literal["vintage", "lag"]
    freq: Literal["D", "W", "M"]
    lag_bdays: int
    paper: str  # the Baek et al. indicator it replaces ("" = same indicator)


MACRO: tuple[MacroSeries, ...] = (
    # Lags for vintage series are fallbacks for pre-ALFRED history; each exceeds the 95th
    # percentile of release delays observed in ALFRED (CFNAI 19, KCFSI 8, weekly 4 bdays).
    MacroSeries("cfnai", ("CFNAI",), "vintage", "M", 20, ""),
    MacroSeries("nfci", ("NFCI",), "vintage", "W", 5, ""),
    MacroSeries("anfci", ("ANFCI",), "vintage", "W", 5, "Goldman Sachs FCI"),
    MacroSeries(
        "nfci_credit", ("NFCICREDIT",), "vintage", "W", 5, "Conference Board leading credit index"
    ),
    MacroSeries("stlfsi", ("STLFSI4",), "vintage", "W", 5, "Bloomberg FCI"),
    MacroSeries("kc_stress", ("KCFSI",), "vintage", "M", 10, ""),
    MacroSeries("cp_tbill", ("DCPF3M", "DTB3"), "lag", "D", 1, ""),
    MacroSeries("baa_tsy", ("BAA10Y",), "lag", "D", 1, ""),
    MacroSeries("baa_aaa", ("DBAA", "DAAA"), "lag", "D", 1, "ICE BofA high-yield OAS"),
    MacroSeries("term_spread", ("T10Y3M",), "lag", "D", 1, "TED spread / LIBOR-OIS"),
    MacroSeries("vix", ("VIXCLS",), "lag", "D", 1, ""),
    MacroSeries("vxn", ("VXNCLS",), "lag", "D", 1, ""),
    MacroSeries("vxd", ("VXDCLS",), "lag", "D", 1, ""),
)
