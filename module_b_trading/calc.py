"""Index calculator + factsheet summary stats."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_RUN, RunConfig
from module_b_trading.indices import (
    CommodityCurveIndex,
    EqVolTargetIndex,
    FxCarryIndex,
    RatesCarryIndex,
)


@dataclass
class IndexCalculator:
    rulebook: Any
    name: str

    def run(self, start: date, end: date, cfg: RunConfig = DEFAULT_RUN) -> pd.DataFrame:
        """Run the rulebook and return [ret, level, weight] with level seeded at 100."""
        df = self.rulebook.compute(start, end, cfg).copy()
        df["level"] = 100.0 * (1.0 + df["ret"].fillna(0.0)).cumprod()
        return df[["ret", "level", "weight"]]


class FactsheetBuilder:
    """Summary stats from an IndexCalculator.run() output."""

    @staticmethod
    def build(df: pd.DataFrame, name: str) -> dict:
        r = df["ret"].astype(float).fillna(0.0)
        level = df["level"].astype(float)
        n = len(r)
        if n == 0:
            return {"name": name, "level_end": np.nan, "total_return": np.nan,
                    "annual_return": np.nan, "annual_vol": np.nan, "sharpe": np.nan,
                    "max_drawdown": np.nan, "calmar": np.nan}
        total_return = float(level.iloc[-1] / 100.0 - 1.0)
        years = n / 252.0
        annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0
        annual_vol = float(r.std(ddof=0) * np.sqrt(252))
        sharpe = annual_return / annual_vol if annual_vol > 1e-12 else float("nan")
        roll_max = level.cummax()
        drawdown = (level / roll_max - 1.0)
        max_dd = float(drawdown.min())
        calmar = annual_return / abs(max_dd) if max_dd < 0 else float("nan")
        return {
            "name": name,
            "level_end": float(level.iloc[-1]),
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
        }


if __name__ == "__main__":
    s, e = date(2022, 1, 1), date(2024, 6, 30)
    pairs = [
        ("EqVolTarget",    EqVolTargetIndex()),
        ("RatesCarry",     RatesCarryIndex()),
        ("CommodityCurve", CommodityCurveIndex()),
        ("FxCarry",        FxCarryIndex()),
    ]
    for name, rb in pairs:
        calc = IndexCalculator(rulebook=rb, name=name)
        df = calc.run(s, e)
        fs = FactsheetBuilder.build(df, name)
        print(
            f"{name:16s} level_end={fs['level_end']:7.2f}  "
            f"ann_ret={fs['annual_return']:+.3f}  ann_vol={fs['annual_vol']:.3f}  "
            f"sharpe={fs['sharpe']:+.2f}  max_dd={fs['max_drawdown']:+.3f}  "
            f"calmar={fs['calmar'] if not np.isnan(fs['calmar']) else float('nan'):+.2f}"
        )
