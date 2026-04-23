"""
Commodity curve proxy: realized roll yield from the front-month generic.

Free-tier data (yfinance) gives us a back-adjusted front-month continuous
series (CL=F / HO=F / RB=F / NG=F) but not the full expiry chain. We proxy
the curve shape with the *realized* roll yield of a continuously held
front-long position over a `lookback` window:

    realized_roll_yield(t) = (252 / lookback) * log(F_t / F_{t-lookback})

Positive -> the curve has been backwardated over the window (long front
profited from rolling). Negative -> contango (long front paid roll).

Signal: sign(realized_roll_yield) gives the position direction in the
commodity index. This is the Koijen-Moskowitz-Pedersen (2018) insight that
time-series realized returns are a robust proxy for carry when direct
curve observations are noisy or unavailable.

For WTI only, a more direct roll-yield proxy using FRED DCOILWTICO spot
is also provided via roll_yield_from_spot() as a cross-check.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from module_a_data.loaders import get_fred_series, get_yf_panel
from config import COMMODITY_GENERICS, DEFAULT_RUN


@dataclass
class CommodityCurve:
    """Per-commodity curve proxy built from front-month price history."""

    ticker: str
    front_prices: pd.Series
    lookback_days: int = 63   # ~3 months

    def realized_roll_yield(self) -> pd.Series:
        """Annualized realized roll yield of the front-long rolled position."""
        log_ret = np.log(self.front_prices / self.front_prices.shift(self.lookback_days))
        return (252.0 / self.lookback_days) * log_ret.rename("roll_yield")

    def curve_state(self) -> pd.Series:
        """+1 = backwardation-regime, -1 = contango, 0 = flat."""
        ry = self.realized_roll_yield()
        return np.sign(ry).fillna(0).astype(int).rename("curve_state")

    def position_weight(self, target_vol: float = 0.10) -> pd.Series:
        """Vol-scaled signed position on the front. EWMA vol over 21d."""
        rets = self.front_prices.pct_change()
        ewma_var = rets.pow(2).ewm(alpha=1 - 0.94, adjust=False).mean()
        ann_vol = (ewma_var * 252).pow(0.5)
        raw = self.curve_state().shift(1)                    # use prior-day regime
        scale = (target_vol / ann_vol.replace(0, np.nan)).clip(upper=1.5)
        return (raw * scale).fillna(0).rename(f"{self.ticker}_wt")


def build_commodity_curves(
    tickers: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    lookback_days: int = 63,
) -> dict[str, CommodityCurve]:
    tickers = tickers or COMMODITY_GENERICS
    start = start or DEFAULT_RUN.start_date
    end = end or DEFAULT_RUN.end_date
    panel = get_yf_panel(tickers, start, end)
    return {
        t: CommodityCurve(ticker=t, front_prices=panel[t].dropna(), lookback_days=lookback_days)
        for t in tickers
    }


def roll_yield_from_spot(start: date, end: date, lookback_days: int = 21) -> pd.Series:
    """
    WTI-only direct roll-yield cross-check: 12 * log(F_front / S_spot)
    where S_spot = FRED DCOILWTICO and F_front = yfinance CL=F.
    Returns a pd.Series of annualized roll yield by date.
    """
    spot = get_fred_series("DCOILWTICO", start, end)
    front = get_yf_panel(["CL=F"], start, end)["CL=F"]
    aligned = pd.concat([spot.rename("spot"), front.rename("front")], axis=1).dropna()
    return (12.0 * np.log(aligned["front"] / aligned["spot"])).rename("wti_roll_yield_annual")


if __name__ == "__main__":
    curves = build_commodity_curves(start=date(2023, 1, 1), end=date(2024, 6, 30))
    print("Commodity roll-yield snapshot (last obs):")
    for t, c in curves.items():
        ry = c.realized_roll_yield().dropna()
        st = c.curve_state().dropna()
        print(f"  {t:5s}  roll_yield={ry.iloc[-1]:+.3f}   state={int(st.iloc[-1]):+d}   n={len(ry)}")
    wti = roll_yield_from_spot(date(2023, 1, 1), date(2024, 6, 30))
    print(f"\nWTI direct roll-yield (spot vs front), last: {wti.dropna().iloc[-1]:+.3f}")
