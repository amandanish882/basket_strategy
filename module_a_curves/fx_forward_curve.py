"""
FX forward curve via Covered Interest Parity.

For each foreign currency vs USD:
    F(t, T) = S(t) * exp((r_d - r_f) * T)                    (CIP)
    carry(t) = (F/S - 1) / T ~= r_d - r_f                    (per-year)

For the FX carry leg we hold foreign currency and short USD, so the carry
*earned* by longing the foreign ccy is  r_f - r_d.

Rate sources (all free FRED):
    USD  : DGS3MO  (3-month T-bill, daily)
    EUR  : IR3TIB01EZM156N  (euro-area 3M interbank, monthly -> ffill)
    GBP  : IR3TIB01GBM156N
    JPY  : IR3TIB01JPM156N
    AUD  : IR3TIB01AUM156N

Spot (all FRED, all USD-quoted):
    EUR  : DEXUSEU  (USD per EUR)
    GBP  : DEXUSUK  (USD per GBP)
    JPY  : DEXJPUS  (JPY per USD)   -- inverted below to USD per JPY
    AUD  : DEXUSAL  (USD per AUD)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from module_a_data.loaders import get_fred_panel, get_fred_series
from config import DEFAULT_RUN


FOREIGN_RATE_CODES: dict[str, str] = {
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "JPY": "IR3TIB01JPM156N",
    "AUD": "IR3TIB01AUM156N",
}

FX_SPOT_CODES: dict[str, tuple[str, bool]] = {
    # ccy : (fred_code, invert_so_usd_per_unit_of_foreign)
    "EUR": ("DEXUSEU", False),   # USD per EUR already
    "GBP": ("DEXUSUK", False),   # USD per GBP already
    "JPY": ("DEXJPUS", True),    # JPY per USD -> invert for USD per JPY
    "AUD": ("DEXUSAL", False),   # USD per AUD already
}


@dataclass
class FxForwardCurve:
    """Per-pair FX forward curve derived via CIP."""

    pair: str                           # e.g. 'EUR'  (meaning EURUSD)
    spot_usd_per_fcy: pd.Series         # S(t) in USD per 1 unit of foreign ccy
    usd_rate: pd.Series                 # r_USD per-annum decimal
    foreign_rate: pd.Series             # r_FCY per-annum decimal

    def aligned(self) -> pd.DataFrame:
        df = pd.concat([
            self.spot_usd_per_fcy.rename("S"),
            self.usd_rate.rename("r_d"),
            self.foreign_rate.rename("r_f"),
        ], axis=1).dropna()
        return df

    def carry(self) -> pd.Series:
        """Carry earned by longing the foreign ccy: r_f - r_d (annualized)."""
        df = self.aligned()
        return (df["r_f"] - df["r_d"]).rename(f"{self.pair}_carry")

    def implied_forward(self, tenor_years: float = 0.25) -> pd.Series:
        """CIP-implied F(t, T) = S(t) * exp((r_d - r_f) * T)."""
        df = self.aligned()
        return (df["S"] * np.exp((df["r_d"] - df["r_f"]) * tenor_years)).rename(f"{self.pair}_fwd{tenor_years:g}")

    def spot_return(self) -> pd.Series:
        """Daily return of holding 1 unit of foreign ccy (USD-denominated)."""
        return self.spot_usd_per_fcy.pct_change().rename(f"{self.pair}_ret")


def build_fx_forward_panel(
    start: date | None = None,
    end: date | None = None,
) -> dict[str, FxForwardCurve]:
    start = start or DEFAULT_RUN.start_date
    end = end or DEFAULT_RUN.end_date
    usd_rate = get_fred_series("DGS3MO", start, end) / 100.0
    foreign_rates = get_fred_panel(FOREIGN_RATE_CODES.values(), start, end) / 100.0
    spots_raw = get_fred_panel([c for c, _ in FX_SPOT_CODES.values()], start, end)
    foreign_rates = foreign_rates.reindex(usd_rate.index).ffill()
    spots_raw = spots_raw.reindex(usd_rate.index).ffill()
    curves: dict[str, FxForwardCurve] = {}
    for ccy, (code, invert) in FX_SPOT_CODES.items():
        raw = spots_raw[code]
        spot = (1.0 / raw) if invert else raw
        fr_code = FOREIGN_RATE_CODES[ccy]
        curves[ccy] = FxForwardCurve(
            pair=ccy,
            spot_usd_per_fcy=spot.dropna(),
            usd_rate=usd_rate,
            foreign_rate=foreign_rates[fr_code],
        )
    return curves


if __name__ == "__main__":
    curves = build_fx_forward_panel(start=date(2023, 1, 1), end=date(2024, 6, 30))
    print("FX carry snapshot (last obs):")
    for ccy, c in curves.items():
        carry = c.carry().dropna()
        if len(carry) == 0:
            print(f"  {ccy}  NO DATA")
            continue
        spot_last = c.spot_usd_per_fcy.dropna().iloc[-1]
        print(f"  {ccy}  S={spot_last:.4f}  carry(r_f - r_d)={carry.iloc[-1]*100:+.2f}%  n={len(carry)}")
    c = curves["EUR"]
    fwd3m = c.implied_forward(0.25).dropna()
    print(f"\nEUR 3M CIP forward (last): {fwd3m.iloc[-1]:.4f}")
