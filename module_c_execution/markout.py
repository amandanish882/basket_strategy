from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class MarkoutResult:
    side: int
    shares: float
    decision_price: float
    arrival_price: float
    avg_execution_price: float
    end_price: float
    total_shortfall_bps: float
    delay_bps: float
    temporary_bps: float
    permanent_bps: float
    adverse_selection_bps: float


def _bps(diff: float, side: int, decision_price: float) -> float:
    if decision_price == 0.0:
        raise ValueError("decision_price must be non-zero")
    return float(diff * side / decision_price * 10000.0)


def decompose_markout(
    decision_price: float,
    arrival_price: float,
    fills: pd.Series,
    shares_per_fill: pd.Series,
    end_price: float,
    side: int,
) -> MarkoutResult:
    if side not in (1, -1):
        raise ValueError("side must be +1 (buy) or -1 (sell)")
    if len(fills) == 0 or len(shares_per_fill) == 0:
        raise ValueError("fills and shares_per_fill must be non-empty")
    if len(fills) != len(shares_per_fill):
        raise ValueError("fills and shares_per_fill must align")

    prices = np.asarray(fills.values, dtype=float)
    qty = np.asarray(shares_per_fill.values, dtype=float)
    total_shares = float(np.sum(qty))
    if total_shares <= 0.0:
        raise ValueError("total shares must be positive")
    avg_exec = float(np.sum(prices * qty) / total_shares)

    # Sign convention: positive == adverse to trader. For a buy (side=+1),
    # paying more than decision is adverse, so we use (exec - decision) * side.
    total = _bps(avg_exec - decision_price, side, decision_price)
    delay = _bps(arrival_price - decision_price, side, decision_price)
    temp = _bps(avg_exec - arrival_price, side, decision_price)
    perm = _bps(end_price - arrival_price, side, decision_price)
    adverse = _bps(end_price - avg_exec, side, decision_price)

    return MarkoutResult(
        side=side,
        shares=total_shares,
        decision_price=float(decision_price),
        arrival_price=float(arrival_price),
        avg_execution_price=avg_exec,
        end_price=float(end_price),
        total_shortfall_bps=total,
        delay_bps=delay,
        temporary_bps=temp,
        permanent_bps=perm,
        adverse_selection_bps=adverse,
    )


if __name__ == "__main__":
    idx = pd.RangeIndex(5)
    fills = pd.Series([100.05, 100.08, 100.12, 100.14, 100.15], index=idx)
    qty = pd.Series([100, 100, 100, 100, 100], index=idx)
    res = decompose_markout(
        decision_price=100.00,
        arrival_price=100.05,
        fills=fills,
        shares_per_fill=qty,
        end_price=100.20,
        side=1,
    )
    out = asdict(res)
    print("MarkoutResult =")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    check = res.total_shortfall_bps - res.delay_bps - res.adverse_selection_bps
    print(
        f"\nSanity: total - delay - adverse = {check:.4f} bps; "
        f"temp+perm = {res.temporary_bps + res.permanent_bps:.4f} bps"
    )
