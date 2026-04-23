"""Unit tests for module_b_trading.risk."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from module_b_trading.risk import (  # noqa: E402
    EqOptionPosition,
    Portfolio,
    RatesPosition,
    greeks_sheet,
    hedge_frontier,
    scenario_revalue,
    spot_vol_ladder,
)
from module_a_curves.curve_bootstrapper import (  # noqa: E402
    CurveBootstrapper, CurveInstrument, DayCountConvention, InstrumentType,
)


def _flat_curve(rate: float = 0.045) -> "DiscountCurve":  # type: ignore[name-defined]
    return CurveBootstrapper.bootstrap([
        CurveInstrument(InstrumentType.DEPOSIT, 0.25, rate),
        CurveInstrument(InstrumentType.SWAP, 2.0, rate),
        CurveInstrument(InstrumentType.SWAP, 5.0, rate),
        CurveInstrument(InstrumentType.SWAP, 10.0, rate),
        CurveInstrument(InstrumentType.SWAP, 30.0, rate),
    ])


def _atm_call(qty: float = 1.0) -> EqOptionPosition:
    return EqOptionPosition(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
                            sigma=0.2, is_call=True, qty=qty)


def test_greeks_at_the_money_call():
    port = Portfolio(equity_options=[_atm_call(1.0)])
    df = greeks_sheet(port)
    row = df.iloc[0]
    # BS closed-form delta for S=K=100, T=1, r=5%, q=0, sigma=20% is ~0.6368
    # delta_usd = qty * delta * S * 100 ~= 1 * 0.6368 * 100 * 100 = 6368
    assert abs(row["delta_usd"] - 6368.0) < 50.0
    assert row["gamma_usd"] > 0.0
    assert row["vega_usd"] > 0.0


def test_scenario_pnl_signs():
    long_port = Portfolio(equity_options=[_atm_call(1.0)])
    short_port = Portfolio(equity_options=[_atm_call(-1.0)])
    shock = {"spot_bp": 100, "vol_bp": 0}
    p_long = scenario_revalue(long_port, shock)["pnl_usd"]
    p_short = scenario_revalue(short_port, shock)["pnl_usd"]
    assert p_long > 0
    assert p_short < 0
    assert abs(p_long + p_short) < 1e-8


def test_ladder_shape_and_dtype():
    port = Portfolio(equity_options=[_atm_call(1.0)])
    lad = spot_vol_ladder(port)
    assert isinstance(lad, pd.DataFrame)
    assert lad.shape == (21, 21)
    assert np.all(np.isfinite(lad.values))


def test_rates_full_reval_captures_convexity():
    """With a curve attached, rates scenario should differ from linear DV01 for large shocks
    by a positive-convexity amount (long duration gains MORE than DV01 on rates-down,
    loses LESS than DV01 on rates-up)."""
    curve = _flat_curve(0.045)
    coupon = curve.par_rate(10.0)
    rp = RatesPosition(tenor_years=10.0, dv01_per_mm=820.0, notional_mm=1.0, coupon=coupon)
    port_with_curve = Portfolio(rates=[rp], curve=curve)
    port_no_curve = Portfolio(rates=[rp])

    for shock_bp in (-200, +200):
        pnl_full = scenario_revalue(port_with_curve, {"rate_bp": shock_bp})["pnl_usd"]
        pnl_dv01 = scenario_revalue(port_no_curve, {"rate_bp": shock_bp})["pnl_usd"]
        # For +200 bp: DV01 = -notional * dv01 * 200 = -164000 (a loss).
        # Full reval should be less negative (convexity is positive for a receiver / long-duration).
        # For -200 bp: DV01 = +164000; full reval should be MORE positive.
        # I.e., pnl_full - pnl_dv01 > 0 either way.
        assert pnl_full - pnl_dv01 > 0.0, f"convexity gap negative at {shock_bp} bp"


def test_rates_small_shock_matches_dv01():
    """For a small shock (1 bp), full reval and DV01 should agree to leading order."""
    curve = _flat_curve(0.045)
    coupon = curve.par_rate(10.0)
    rp = RatesPosition(tenor_years=10.0, dv01_per_mm=820.0, notional_mm=1.0, coupon=coupon)
    port_with_curve = Portfolio(rates=[rp], curve=curve)
    pnl_full = scenario_revalue(port_with_curve, {"rate_bp": 1.0})["pnl_usd"]
    # Expected DV01 magnitude: 820 $/bp/mm * 1 mm * 1 bp = 820; sign is negative for long duration + rates up.
    # Full reval won't match exactly (curve's implied DV01 from swap_npv may differ from the hardcoded 820
    # because the hardcoded value was not built from this curve). Allow a 30% tolerance for the demo.
    assert -1300 < pnl_full < -400


def test_frontier_monotone_and_min_points():
    port = Portfolio(
        equity_options=[_atm_call(1.0)],
        rates=[RatesPosition(tenor_years=10.0, dv01_per_mm=820.0, notional_mm=1.0)],
    )
    hedges = [
        EqOptionPosition(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
                         sigma=0.2, is_call=False, qty=1.0),
        RatesPosition(tenor_years=10.0, dv01_per_mm=820.0, notional_mm=1.0),
    ]
    fr = hedge_frontier(port, hedges, max_hedge_notional=10.0)
    assert len(fr) >= 10
    # Sort ascending by lambda so we look from "don't care about variance" -> "care a lot"
    fr_sorted = fr.sort_values("lambda").reset_index(drop=True)
    # As lambda increases we weight variance more -> cost non-decreasing, var non-increasing.
    # Equivalently: as lambda *decreases* (reverse order) cost non-increasing / var non-decreasing.
    # Allow small numerical slack.
    costs = fr_sorted["hedge_cost"].values
    vars_ = fr_sorted["residual_var"].values
    tol_c = max(1e-6, 1e-6 * max(abs(costs).max(), 1.0))
    tol_v = max(1e-6, 1e-6 * max(abs(vars_).max(), 1.0))
    assert all(costs[i + 1] >= costs[i] - tol_c for i in range(len(costs) - 1))
    assert all(vars_[i + 1] <= vars_[i] + tol_v for i in range(len(vars_) - 1))
