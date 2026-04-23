"""Cross-asset risk: Greeks, scenarios, 441-cell ladder, SLSQP hedge frontier."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Make package imports work whether risk.py is run directly or imported.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared import (  # noqa: E402
    KERNEL_BACKEND,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_theta,
    bs_vega,
)
from module_a_curves.curve_bootstrapper import DiscountCurve  # noqa: E402

# Approximate CME contract multipliers (point-value in USD).
CONTRACT_MULTIPLIERS: dict[str, float] = {
    "CL=F": 1000.0,
    "HO=F": 42000.0,
    "RB=F": 42000.0,
    "NG=F": 10000.0,
}
_DEFAULT_CMULT = 1000.0
SHARES_PER_CONTRACT = 100.0


# ---- Portfolio dataclasses ---- #
@dataclass
class EqOptionPosition:
    S: float
    K: float
    T: float
    r: float
    q: float
    sigma: float
    is_call: bool
    qty: float


@dataclass
class RatesPosition:
    tenor_years: float
    dv01_per_mm: float      # $ per bp per $1mm notional
    notional_mm: float      # signed, + = long duration
    coupon: float = 0.0     # fixed rate of the embedded par swap. 0 -> set from curve.par_rate()


@dataclass
class CommodityPosition:
    ticker: str
    spot: float
    qty: float              # contracts, signed


@dataclass
class FxPosition:
    pair: str               # 'EUR', 'GBP', ...
    spot: float             # USD per unit foreign
    qty: float              # notional in $mm of foreign ccy, signed


@dataclass
class Portfolio:
    equity_options: list[EqOptionPosition] = field(default_factory=list)
    rates: list[RatesPosition] = field(default_factory=list)
    commodities: list[CommodityPosition] = field(default_factory=list)
    fx: list[FxPosition] = field(default_factory=list)
    curve: Optional[DiscountCurve] = None


# ---- 1) Greeks sheet ---- #
_COLS = [
    "asset_class", "instrument", "qty",
    "delta_usd", "gamma_usd", "vega_usd", "theta_usd",
    "dv01_usd", "fx_delta_usd",
]


def _eq_row(pos: EqOptionPosition) -> dict:
    cp = "C" if pos.is_call else "P"
    d = bs_delta(pos.S, pos.K, pos.T, pos.r, pos.q, pos.sigma, pos.is_call)
    g = bs_gamma(pos.S, pos.K, pos.T, pos.r, pos.q, pos.sigma)
    v = bs_vega(pos.S, pos.K, pos.T, pos.r, pos.q, pos.sigma)
    th = bs_theta(pos.S, pos.K, pos.T, pos.r, pos.q, pos.sigma, pos.is_call)
    m = SHARES_PER_CONTRACT
    return {
        "asset_class": "EQ",
        "instrument": f"{cp} K={pos.K:g} T={pos.T:.2f}",
        "qty": pos.qty,
        "delta_usd": pos.qty * d * pos.S * m,
        "gamma_usd": pos.qty * g * pos.S * pos.S * 0.01 * m,
        "vega_usd": pos.qty * v * 0.01 * m,
        "theta_usd": pos.qty * th / 365.0 * m,
        "dv01_usd": 0.0,
        "fx_delta_usd": 0.0,
    }


def _rates_row(pos: RatesPosition) -> dict:
    return {
        "asset_class": "RATES",
        "instrument": f"IRS {pos.tenor_years:g}Y",
        "qty": pos.notional_mm,
        "delta_usd": 0.0, "gamma_usd": 0.0, "vega_usd": 0.0, "theta_usd": 0.0,
        "dv01_usd": pos.notional_mm * pos.dv01_per_mm,
        "fx_delta_usd": 0.0,
    }


def _cmdty_row(pos: CommodityPosition) -> dict:
    mult = CONTRACT_MULTIPLIERS.get(pos.ticker, _DEFAULT_CMULT)
    return {
        "asset_class": "CMDTY",
        "instrument": pos.ticker,
        "qty": pos.qty,
        "delta_usd": pos.qty * pos.spot * mult,
        "gamma_usd": 0.0, "vega_usd": 0.0, "theta_usd": 0.0,
        "dv01_usd": 0.0, "fx_delta_usd": 0.0,
    }


def _fx_row(pos: FxPosition) -> dict:
    return {
        "asset_class": "FX",
        "instrument": f"{pos.pair}USD",
        "qty": pos.qty,
        "delta_usd": 0.0, "gamma_usd": 0.0, "vega_usd": 0.0, "theta_usd": 0.0,
        "dv01_usd": 0.0,
        "fx_delta_usd": pos.qty * pos.spot * 1e6,
    }


def greeks_sheet(port: Portfolio) -> pd.DataFrame:
    """Position-level Greeks in USD plus a TOTAL row."""
    rows: list[dict] = []
    for p in port.equity_options:
        rows.append(_eq_row(p))
    for p in port.rates:
        rows.append(_rates_row(p))
    for p in port.commodities:
        rows.append(_cmdty_row(p))
    for p in port.fx:
        rows.append(_fx_row(p))
    df = pd.DataFrame(rows, columns=_COLS) if rows else pd.DataFrame(columns=_COLS)
    if not df.empty:
        total = {c: df[c].sum() if c not in ("asset_class", "instrument") else "TOTAL"
                 for c in _COLS}
        total["asset_class"] = "TOTAL"
        total["instrument"] = ""
        df = pd.concat([df, pd.DataFrame([total], columns=_COLS)], ignore_index=True)
    return df


# ---- 2) Scenario engine ---- #
def _eq_scenario_pnl(opts: list[EqOptionPosition], spot_bp: float, vol_bp: float) -> float:
    if not opts:
        return 0.0
    s_mult = 1.0 + spot_bp / 10000.0
    dv = vol_bp / 10000.0
    pnl = 0.0
    for p in opts:
        p0 = bs_price(p.S, p.K, p.T, p.r, p.q, p.sigma, p.is_call)
        p1 = bs_price(p.S * s_mult, p.K, p.T, p.r, p.q, max(p.sigma + dv, 1e-6), p.is_call)
        pnl += (p1 - p0) * p.qty * SHARES_PER_CONTRACT
    return pnl


def _rates_scenario_pnl(rp: list[RatesPosition], rate_bp: float,
                        curve: Optional[DiscountCurve] = None) -> float:
    """Full curve bump-and-reval when a curve is attached; DV01 linear fallback otherwise.

    The full-reval path shifts the curve by `rate_bp` via shift_parallel and reprices each
    position as a fixed-for-float par swap via DiscountCurve.swap_npv. This captures
    convexity (second-order) that pure DV01 misses for large shocks.
    """
    if not rp:
        return 0.0
    if curve is None:
        return -sum(r.notional_mm * r.dv01_per_mm * rate_bp for r in rp)
    shifted = curve.shift_parallel(rate_bp)
    pnl = 0.0
    for p in rp:
        coupon = p.coupon or curve.par_rate(p.tenor_years)
        nv_base = curve.swap_npv(coupon, p.tenor_years, receiver=True)
        nv_new = shifted.swap_npv(coupon, p.tenor_years, receiver=True)
        pnl += p.notional_mm * 1e6 * (nv_new - nv_base)
    return pnl


def _cmdty_scenario_pnl(cp: list[CommodityPosition], pct: float) -> float:
    return sum(
        p.qty * p.spot * CONTRACT_MULTIPLIERS.get(p.ticker, _DEFAULT_CMULT) * pct
        for p in cp
    )


def _fx_scenario_pnl(fxp: list[FxPosition], pct: float) -> float:
    return sum(p.qty * p.spot * 1e6 * pct for p in fxp)


def scenario_revalue(port: Portfolio, shifts: dict) -> dict:
    """Apply shocks and return {'pnl_usd', 'by_asset_class': {...}}."""
    s = shifts.get
    eq = _eq_scenario_pnl(port.equity_options, s("spot_bp", 0.0), s("vol_bp", 0.0))
    rt = _rates_scenario_pnl(port.rates, s("rate_bp", 0.0), port.curve)
    cm = _cmdty_scenario_pnl(port.commodities, s("commodity_pct", 0.0))
    fx = _fx_scenario_pnl(port.fx, s("fx_pct", 0.0))
    by = {"EQ": eq, "RATES": rt, "CMDTY": cm, "FX": fx}
    return {"pnl_usd": eq + rt + cm + fx, "by_asset_class": by}


# ---- 3) 441-cell spot x vol ladder ---- #
def spot_vol_ladder(
    port: Portfolio,
    spot_bps: Optional[np.ndarray] = None,
    vol_bps: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """21x21 PnL grid over equity spot/vol shocks (other legs not bumped)."""
    if spot_bps is None:
        spot_bps = np.linspace(-1000, 1000, 21)
    if vol_bps is None:
        vol_bps = np.linspace(-500, 500, 21)
    grid = np.zeros((len(spot_bps), len(vol_bps)))
    for i, sbp in enumerate(spot_bps):
        for j, vbp in enumerate(vol_bps):
            grid[i, j] = _eq_scenario_pnl(port.equity_options, float(sbp), float(vbp))
    return pd.DataFrame(grid, index=np.round(spot_bps, 4), columns=np.round(vol_bps, 4))


# ---- 4) SLSQP cost-variance hedge frontier ---- #
_EQ_SPOT_SIGMA = 0.20          # annual
_RATES_DAILY_BP = 10.0         # 1-sigma daily in bp


def _candidate_cost(c, x: float) -> float:
    if isinstance(c, EqOptionPosition):
        p = bs_price(c.S, c.K, c.T, c.r, c.q, c.sigma, c.is_call)
        half = 0.02 * p * SHARES_PER_CONTRACT
        return abs(x) * half
    if isinstance(c, RatesPosition):
        half_bp = 0.5
        return abs(x) * abs(c.dv01_per_mm) * half_bp
    return 0.0


def _candidate_exposure(c, x: float) -> tuple[float, float]:
    """Return (delta_usd_exposure, dv01_usd_exposure) contribution for x units."""
    if isinstance(c, EqOptionPosition):
        d = bs_delta(c.S, c.K, c.T, c.r, c.q, c.sigma, c.is_call)
        return x * d * c.S * SHARES_PER_CONTRACT, 0.0
    if isinstance(c, RatesPosition):
        return 0.0, x * c.dv01_per_mm
    return 0.0, 0.0


def _portfolio_exposure(port: Portfolio) -> tuple[float, float]:
    d_usd = sum(
        p.qty * bs_delta(p.S, p.K, p.T, p.r, p.q, p.sigma, p.is_call) * p.S * SHARES_PER_CONTRACT
        for p in port.equity_options
    )
    dv01 = sum(r.notional_mm * r.dv01_per_mm for r in port.rates)
    return d_usd, dv01


def _residual_variance(port_d: float, port_dv: float,
                       hedges: list, x: np.ndarray) -> float:
    rd, rv = port_d, port_dv
    for c, xi in zip(hedges, x):
        dd, dv = _candidate_exposure(c, xi)
        rd += dd
        rv += dv
    # Diagonal variance: equity delta at 20% annual (daily-style scale);
    # rates DV01 at 10 bp daily shock.
    var_eq = (rd * _EQ_SPOT_SIGMA) ** 2
    var_rt = (rv * _RATES_DAILY_BP) ** 2
    return var_eq + var_rt


def _hedge_cost(hedges: list, x: np.ndarray) -> float:
    return sum(_candidate_cost(c, xi) for c, xi in zip(hedges, x))


def _min_variance_seed(port_d: float, port_dv: float, hedges: list,
                        cap: float) -> np.ndarray:
    """Closed-form min-variance hedge on the diagonal (eq-delta / dv01) system."""
    n = len(hedges)
    x = np.zeros(n)
    eq_idx = [i for i, c in enumerate(hedges) if isinstance(c, EqOptionPosition)]
    rt_idx = [i for i, c in enumerate(hedges) if isinstance(c, RatesPosition)]
    if eq_idx:
        # pick first eq candidate as the hedge instrument
        i = eq_idx[0]
        c = hedges[i]
        d_per_unit = bs_delta(c.S, c.K, c.T, c.r, c.q, c.sigma, c.is_call) * c.S * SHARES_PER_CONTRACT
        if abs(d_per_unit) > 1e-12:
            x[i] = np.clip(-port_d / d_per_unit, -cap, cap)
    if rt_idx:
        i = rt_idx[0]
        c = hedges[i]
        if abs(c.dv01_per_mm) > 1e-12:
            x[i] = np.clip(-port_dv / c.dv01_per_mm, -cap, cap)
    return x


def _smooth_abs(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.sqrt(x * x + eps * eps)


def _smooth_cost(hedges: list, x: np.ndarray) -> float:
    c = 0.0
    ax = _smooth_abs(x)
    for i, cand in enumerate(hedges):
        if isinstance(cand, EqOptionPosition):
            p = bs_price(cand.S, cand.K, cand.T, cand.r, cand.q, cand.sigma, cand.is_call)
            half = 0.02 * p * SHARES_PER_CONTRACT
            c += ax[i] * half
        elif isinstance(cand, RatesPosition):
            c += ax[i] * abs(cand.dv01_per_mm) * 0.5
    return c


def hedge_frontier(
    port: Portfolio,
    hedge_candidates: list[Union[EqOptionPosition, RatesPosition]],
    lambdas: Optional[np.ndarray] = None,
    max_hedge_notional: float = 1e6,
) -> pd.DataFrame:
    """Trace cost vs residual-variance frontier by sweeping lambda."""
    if lambdas is None:
        lambdas = np.logspace(-8, -2, 15)
    if not hedge_candidates:
        raise ValueError("Need at least one hedge candidate.")

    port_d, port_dv = _portfolio_exposure(port)
    n = len(hedge_candidates)
    bounds = [(-max_hedge_notional, max_hedge_notional)] * n

    # Seed warm start with the variance-only minimiser so SLSQP starts in
    # the hedged basin (avoids the x=0 kink of |x| cost).
    lam_sorted = np.sort(np.asarray(lambdas))[::-1]  # descending
    x_warm = _min_variance_seed(port_d, port_dv, hedge_candidates, max_hedge_notional)

    rows = []
    for lam in lam_sorted:
        def obj(x, lam=lam):
            return _smooth_cost(hedge_candidates, x) + lam * _residual_variance(
                port_d, port_dv, hedge_candidates, x
            )
        res = minimize(obj, x_warm, method="SLSQP", bounds=bounds,
                       options={"ftol": 1e-10, "maxiter": 400})
        x_opt = res.x
        x_warm = x_opt  # warm-start next (smaller) lambda
        cost = _hedge_cost(hedge_candidates, x_opt)
        var = _residual_variance(port_d, port_dv, hedge_candidates, x_opt)
        rows.append({
            "lambda": float(lam),
            "hedge_cost": float(cost),
            "residual_var": float(var),
            "residual_stdev": float(np.sqrt(max(var, 0.0))),
            "sum_abs_hedge": float(np.sum(np.abs(x_opt))),
        })
    return pd.DataFrame(rows)


# ---- Smoke test ---- #
def _demo_portfolio() -> Portfolio:
    from module_a_curves.curve_bootstrapper import (
        CurveBootstrapper, CurveInstrument, DayCountConvention, InstrumentType,
    )
    spx = EqOptionPosition(S=5000.0, K=5000.0, T=0.25, r=0.05, q=0.015,
                           sigma=0.18, is_call=True, qty=1.0)
    curve = CurveBootstrapper.bootstrap([
        CurveInstrument(InstrumentType.DEPOSIT, 0.25, 0.053),
        CurveInstrument(InstrumentType.SWAP,    2.0,  0.047),
        CurveInstrument(InstrumentType.SWAP,    5.0,  0.043),
        CurveInstrument(InstrumentType.SWAP,    10.0, 0.043),
        CurveInstrument(InstrumentType.SWAP,    30.0, 0.043),
    ])
    ust = RatesPosition(tenor_years=10.0, dv01_per_mm=820.0, notional_mm=1.0,
                        coupon=curve.par_rate(10.0))
    cl = CommodityPosition(ticker="CL=F", spot=78.0, qty=1.0)
    eur = FxPosition(pair="EUR", spot=1.08, qty=1.0)
    return Portfolio(equity_options=[spx], rates=[ust],
                     commodities=[cl], fx=[eur], curve=curve)


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print(f"KERNEL_BACKEND={KERNEL_BACKEND}")

    port = _demo_portfolio()

    print("\n--- Greeks sheet ---")
    print(greeks_sheet(port).round(2).to_string(index=False))

    print("\n--- Scenario: spot -100 bp, vol +500 bp ---")
    sc = scenario_revalue(port, {"spot_bp": -100, "vol_bp": 500,
                                  "rate_bp": 0, "commodity_pct": 0, "fx_pct": 0})
    print(f"  total PnL  = ${sc['pnl_usd']:,.2f}")
    for k, v in sc["by_asset_class"].items():
        print(f"  {k:<6}     = ${v:,.2f}")

    print("\n--- 21x21 spot/vol ladder ---")
    lad = spot_vol_ladder(port)
    corners = lad.iloc[0, 0] + lad.iloc[0, -1] + lad.iloc[-1, 0] + lad.iloc[-1, -1]
    print(f"  shape={lad.shape}  sum_of_corners=${corners:,.2f}  cells={lad.size}")

    print("\n--- Hedge frontier (head) ---")
    hedges = [
        EqOptionPosition(S=5000.0, K=5000.0, T=0.25, r=0.05, q=0.015,
                         sigma=0.18, is_call=False, qty=1.0),
        RatesPosition(tenor_years=10.0, dv01_per_mm=820.0, notional_mm=1.0),
    ]
    fr = hedge_frontier(port, hedges, max_hedge_notional=10.0)
    print(fr.head().to_string(index=False))
