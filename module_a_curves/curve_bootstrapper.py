"""
USD SOFR/Treasury zero curve bootstrap (pure Python).

Builds a discount curve D(t) from market quotes sequentially:
  - Short end (<= 1Y): deposits via D(T) = 1 / (1 + r * alpha)
  - Long end (> 1Y):   par swaps via
        D(T_n) = (1 - S_n * sum_{j<n} alpha_j D(T_j)) / (1 + S_n * alpha_n)
Interpolation on ln D(t) (piecewise-constant forward rates).

Convenience: build_curve_from_fred() pulls DGS1MO/3MO/2/5/10/30 for a date
and treats them as par rates at those tenors (standard approximation for a
CV-grade project; a full SOFR-swap bootstrap would use SWPM-style swap quotes).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date as _date
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from module_a_data.loaders import get_fred_panel


class DayCountConvention(Enum):
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"


class InstrumentType(Enum):
    DEPOSIT = "deposit"
    SWAP = "swap"


@dataclass
class CurveInstrument:
    instrument_type: InstrumentType
    maturity_years: float
    rate: float                                    # decimal, 0.045 = 4.5%
    day_count: DayCountConvention = DayCountConvention.ACT_360
    payment_frequency: int = 2                     # semi-annual swaps


class DiscountCurve:
    """Discount curve with log-linear interpolation on ln D(t)."""

    def __init__(self, times: np.ndarray, discount_factors: np.ndarray, valuation_date: str = ""):
        self.times = np.asarray(times, dtype=float)
        self.dfs = np.asarray(discount_factors, dtype=float)
        self._log_dfs = np.log(self.dfs)
        self.valuation_date = valuation_date

    def df(self, t: float) -> float:
        if t <= 0:
            return 1.0
        if t >= self.times[-1]:
            z_last = -self._log_dfs[-1] / self.times[-1]
            return float(np.exp(-z_last * t))
        return float(np.exp(np.interp(t, self.times, self._log_dfs)))

    def zero_rate(self, t: float) -> float:
        if t <= 0:
            return self.zero_rate(1 / 365)
        d = self.df(t)
        return -np.log(d) / t if d > 0 else 0.0

    def forward_rate(self, t1: float, t2: float) -> float:
        if abs(t2 - t1) < 1e-10:
            return self.zero_rate(t1)
        d1, d2 = self.df(t1), self.df(t2)
        return (np.log(d1) - np.log(d2)) / (t2 - t1)

    def par_rate(self, maturity: float, frequency: int = 2) -> float:
        n = max(1, int(round(maturity * frequency)))
        dt = maturity / n
        alpha = 1.0 / frequency
        times = np.array([(i + 1) * dt for i in range(n)])
        annuity = sum(alpha * self.df(t) for t in times)
        return (1.0 - self.df(maturity)) / annuity if annuity > 0 else 0.0

    def dv01(self, maturity: float, notional: float = 1e6) -> float:
        """DV01 of a par swap with the given maturity (notional in $)."""
        base = self.par_rate(maturity)
        shifted = self.shift_parallel(1.0).par_rate(maturity)
        return (shifted - base) * notional

    def swap_npv(self, fixed_rate: float, maturity: float, frequency: int = 2,
                 receiver: bool = True) -> float:
        """Per-unit-notional NPV of a fixed-for-float swap under this curve.
        receiver=True means the position receives the fixed leg (long duration).
        At-par entry has NPV 0 iff fixed_rate == self.par_rate(maturity)."""
        n = max(1, int(round(maturity * frequency)))
        dt = maturity / n
        alpha = 1.0 / frequency
        times = np.array([(i + 1) * dt for i in range(n)])
        annuity = sum(alpha * self.df(t) for t in times)
        fixed_leg = fixed_rate * annuity
        float_leg = 1.0 - self.df(maturity)
        value = fixed_leg - float_leg
        return value if receiver else -value

    def shift_parallel(self, shift_bps: float) -> "DiscountCurve":
        shift = shift_bps / 10000.0
        new_dfs = np.array([
            np.exp(-(self.zero_rate(t) + shift) * t) if t > 0 else 1.0
            for t in self.times
        ])
        return DiscountCurve(self.times, new_dfs, self.valuation_date)

    def key_rate_shift(self, bucket_years: float, shift_bps: float, width: float = 1.0) -> "DiscountCurve":
        """Triangle key-rate bump centered at bucket_years with half-width `width`."""
        shift = shift_bps / 10000.0
        weights = np.maximum(0.0, 1.0 - np.abs(self.times - bucket_years) / width)
        new_dfs = np.array([
            np.exp(-(self.zero_rate(t) + shift * w) * t) if t > 0 else 1.0
            for t, w in zip(self.times, weights)
        ])
        return DiscountCurve(self.times, new_dfs, self.valuation_date)

    def zero_curve_df(self, tenors: Sequence[float] | None = None) -> pd.Series:
        tenors = tenors or [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        return pd.Series([self.zero_rate(t) for t in tenors], index=tenors, name="zero_rate")

    def __repr__(self) -> str:
        return f"DiscountCurve(val={self.valuation_date}, nodes={len(self.times)}, max_T={self.times[-1]:.1f}Y)"


class CurveBootstrapper:
    """Sequential bootstrap: deposits first, then swaps (long end)."""

    @staticmethod
    def bootstrap(instruments: Sequence[CurveInstrument], valuation_date: str = "") -> DiscountCurve:
        if not instruments:
            raise ValueError("No instruments provided")
        insts = sorted(instruments, key=lambda x: x.maturity_years)
        times: list[float] = [0.0]
        dfs: list[float] = [1.0]
        for inst in insts:
            T = inst.maturity_years
            if inst.instrument_type == InstrumentType.DEPOSIT:
                alpha = T if inst.day_count == DayCountConvention.ACT_365 else T * 365 / 360
                D = 1.0 / (1.0 + inst.rate * alpha)
            else:
                freq = inst.payment_frequency
                n = max(1, int(round(T * freq)))
                dt = T / n
                alpha = 1.0 / freq
                pay_times = [(i + 1) * dt for i in range(n)]
                tmp = DiscountCurve(np.array(times), np.array(dfs), valuation_date)
                annuity_known = sum(alpha * tmp.df(t) for t in pay_times[:-1])
                D = (1.0 - inst.rate * annuity_known) / (1.0 + inst.rate * alpha)
            times.append(T)
            dfs.append(D)
        return DiscountCurve(np.array(times), np.array(dfs), valuation_date)

    @staticmethod
    def validate(curve: DiscountCurve, instruments: Sequence[CurveInstrument]) -> pd.DataFrame:
        rows = []
        for inst in instruments:
            if inst.instrument_type == InstrumentType.DEPOSIT:
                alpha = inst.maturity_years * 365 / 360
                implied = (1.0 / curve.df(inst.maturity_years) - 1.0) / alpha
            else:
                implied = curve.par_rate(inst.maturity_years, inst.payment_frequency)
            rows.append({
                "tenor": inst.maturity_years,
                "market_rate": inst.rate,
                "model_rate": implied,
                "error_bps": (implied - inst.rate) * 10000.0,
            })
        return pd.DataFrame(rows)


FRED_TENOR_MAP = {
    "DGS1MO": (1 / 12, InstrumentType.DEPOSIT),
    "DGS3MO": (0.25, InstrumentType.DEPOSIT),
    "DGS2":   (2.0,  InstrumentType.SWAP),
    "DGS5":   (5.0,  InstrumentType.SWAP),
    "DGS10":  (10.0, InstrumentType.SWAP),
    "DGS30":  (30.0, InstrumentType.SWAP),
}


def build_curve_from_fred(as_of: _date, lookback_days: int = 10) -> DiscountCurve:
    """Pull DGS* series around `as_of`, take the most recent available row, bootstrap."""
    start = _date.fromordinal(as_of.toordinal() - lookback_days)
    panel = get_fred_panel(FRED_TENOR_MAP.keys(), start, as_of)
    row = panel.loc[:pd.Timestamp(as_of)].dropna(how="all").iloc[-1]
    instruments = []
    for code, (tenor, kind) in FRED_TENOR_MAP.items():
        if code in row.index and not pd.isna(row[code]):
            instruments.append(CurveInstrument(kind, tenor, float(row[code]) / 100.0))
    return CurveBootstrapper.bootstrap(instruments, valuation_date=str(row.name.date()))


if __name__ == "__main__":
    demo = [
        CurveInstrument(InstrumentType.DEPOSIT, 1 / 12, 0.0533),
        CurveInstrument(InstrumentType.DEPOSIT, 0.25,   0.0538),
        CurveInstrument(InstrumentType.SWAP,    2.0,    0.0470),
        CurveInstrument(InstrumentType.SWAP,    5.0,    0.0435),
        CurveInstrument(InstrumentType.SWAP,    10.0,   0.0425),
        CurveInstrument(InstrumentType.SWAP,    30.0,   0.0425),
    ]
    curve = CurveBootstrapper.bootstrap(demo, valuation_date="2024-06-28")
    print(curve)
    print(CurveBootstrapper.validate(curve, demo).round(4))
    print("\nzero curve:")
    print(curve.zero_curve_df().round(4))
    print("\nFRED-driven curve for 2024-06-28:")
    live = build_curve_from_fred(_date(2024, 6, 28))
    print(live)
    print(live.zero_curve_df().round(4))
