"""End-to-end demo orchestrator for the Cross-Asset Strategic Indices QIS Platform.

One command runs:
  1. Four QIS rulebooks + cross-asset vol-target overlay
  2. Factsheet summary stats
  3. 441-cell spot x vol risk ladder on a demo cross-asset portfolio
  4. SLSQP cost-variance hedge frontier
  5. Almgren-Chriss optimal trading trajectory + markout decomposition

Artefacts saved under outputs/.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from config import DEFAULT_RUN, OUTPUT_DIR, PROJECT_NAME, ensure_dirs  # noqa: E402
from module_a_curves.curve_bootstrapper import build_curve_from_fred  # noqa: E402
from module_a_data.loaders import get_fred_series, get_yf_close  # noqa: E402
from module_b_trading.calc import FactsheetBuilder, IndexCalculator  # noqa: E402
from module_b_trading.indices import (  # noqa: E402
    CommodityCurveIndex, EqVolTargetIndex, FxCarryIndex, RatesCarryIndex, XAssetOverlay,
)
from module_b_trading.risk import (  # noqa: E402
    CommodityPosition, EqOptionPosition, FxPosition, Portfolio, RatesPosition,
    hedge_frontier, spot_vol_ladder,
)
from module_c_execution.almgren_chriss import (  # noqa: E402
    AlmgrenChrissParams, optimal_schedule, twap_schedule,
)
from module_c_execution.markout import decompose_markout  # noqa: E402
from shared import plot_index_levels, KERNEL_BACKEND, set_theme, suptitle  # noqa: E402


RULEBOOKS = [
    ("EqVolTarget", EqVolTargetIndex()),
    ("RatesCarry", RatesCarryIndex()),
    ("CommodityCurve", CommodityCurveIndex()),
    ("FxCarry", FxCarryIndex()),
]
_LEG_KEY = {"EqVolTarget": "eq", "RatesCarry": "rates",
            "CommodityCurve": "commodities", "FxCarry": "fx"}


class _OverlayRulebook:
    """Adapter so XAssetOverlay fits the IndexCalculator.compute(start,end,cfg) API."""
    def __init__(self, leg_returns: pd.DataFrame) -> None:
        self._overlay = XAssetOverlay(leg_returns)

    def compute(self, start: date, end: date, cfg) -> pd.DataFrame:  # noqa: ARG002
        return self._overlay.compute(cfg)


def _compute_legs(start: date, end: date, cfg) -> tuple[dict, pd.DataFrame]:
    legs: dict[str, pd.DataFrame] = {
        n: IndexCalculator(rulebook=rb, name=n).run(start, end, cfg) for n, rb in RULEBOOKS
    }
    leg_returns = pd.concat({_LEG_KEY[n]: legs[n]["ret"] for n in legs}, axis=1).dropna()
    legs["XAssetOverlay"] = IndexCalculator(
        _OverlayRulebook(leg_returns), "XAssetOverlay"
    ).run(start, end, cfg)
    return legs, leg_returns


def _save_levels(legs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    levels = pd.concat({n: df["level"] for n, df in legs.items()}, axis=1).ffill()
    levels.to_csv(OUTPUT_DIR / "index_levels.csv")
    fig, ax = plt.subplots()
    plot_index_levels(ax, levels)
    suptitle(fig, "Index Levels - 4 Rulebooks + Cross-Asset Overlay", PROJECT_NAME)
    fig.savefig(OUTPUT_DIR / "index_levels.png"); plt.close(fig)
    return levels


def _save_factsheet(legs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fs = pd.DataFrame(
        [FactsheetBuilder.build(df, n) for n, df in legs.items()]
    ).set_index("name")
    fs.to_csv(OUTPUT_DIR / "factsheet.csv")
    return fs


def _build_demo_portfolio(as_of: date) -> Portfolio:
    try:
        spy = float(get_yf_close("SPY", date(as_of.year - 1, as_of.month, 1), as_of).iloc[-1])
    except Exception:
        spy = 500.0
    try:
        r10 = float(get_fred_series("DGS10", date(as_of.year - 1, 1, 1), as_of).iloc[-1]) / 100.0
    except Exception:
        r10 = 0.042
    try:
        curve = build_curve_from_fred(as_of)
    except Exception:
        curve = None
    spx_call = EqOptionPosition(S=spy, K=round(spy), T=0.25, r=r10, q=0.015,
                                sigma=0.18, is_call=True, qty=1000.0)
    ust10 = RatesPosition(tenor_years=10.0, dv01_per_mm=820.0, notional_mm=60.97)
    cl = CommodityPosition(ticker="CL=F", spot=78.0, qty=100.0)
    eur = FxPosition(pair="EUR", spot=1.08, qty=10.0)
    return Portfolio(equity_options=[spx_call], rates=[ust10],
                     commodities=[cl], fx=[eur], curve=curve)


def _save_ladder(port: Portfolio) -> pd.DataFrame:
    spot_bps = np.linspace(-1000, 1000, 21)
    vol_bps = np.linspace(-500, 500, 21)
    lad = spot_vol_ladder(port, spot_bps=spot_bps, vol_bps=vol_bps)
    fig, ax = plt.subplots()
    im = ax.imshow(lad.values, origin="lower", aspect="auto",
                   extent=(vol_bps[0], vol_bps[-1], spot_bps[0], spot_bps[-1]),
                   cmap="RdYlGn")
    ax.set_xlabel("Vol shock (bps)"); ax.set_ylabel("Spot shock (bps)")
    fig.colorbar(im, ax=ax, label="PnL (USD)")
    suptitle(fig, "Risk Ladder - 21x21 Spot x Vol Revaluation", PROJECT_NAME)
    fig.savefig(OUTPUT_DIR / "risk_ladder.png"); plt.close(fig)
    return lad


def _save_frontier(port: Portfolio) -> pd.DataFrame:
    S = port.equity_options[0].S
    r = port.equity_options[0].r
    hedges = [
        EqOptionPosition(S=S, K=round(S), T=0.25, r=r, q=0.015, sigma=0.20,
                         is_call=False, qty=10.0),
        RatesPosition(tenor_years=2.0,  dv01_per_mm=190.0,  notional_mm=1.0),
        RatesPosition(tenor_years=5.0,  dv01_per_mm=460.0,  notional_mm=1.0),
        RatesPosition(tenor_years=10.0, dv01_per_mm=820.0,  notional_mm=1.0),
        RatesPosition(tenor_years=30.0, dv01_per_mm=1900.0, notional_mm=1.0),
    ]
    fr = hedge_frontier(port, hedges, lambdas=np.logspace(-10, -2, 12),
                        max_hedge_notional=100.0)
    baseline_var = float(fr["residual_var"].max())
    fr["variance_reduction_pct"] = 100.0 * (1.0 - fr["residual_var"] / baseline_var)
    fig, ax = plt.subplots()
    sc = ax.scatter(fr["variance_reduction_pct"], fr["hedge_cost"],
                    c=np.log10(fr["lambda"]), cmap="viridis", s=40)
    label_offsets = {0: (4, 4), 3: (4, -12), 6: (-42, 6), 9: (-42, -12), len(fr) - 1: (4, 4)}
    for i, (_, row) in enumerate(fr.iterrows()):
        if i not in label_offsets:
            continue
        ax.annotate(f"l={row['lambda']:.1e}",
                    (row["variance_reduction_pct"], row["hedge_cost"]),
                    fontsize=6, alpha=0.8, xytext=label_offsets[i],
                    textcoords="offset points",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0})
    fig.colorbar(sc, ax=ax, label="log10(lambda)")
    ax.set_xlabel("Variance reduction vs unhedged proxy (%)"); ax.set_ylabel("Hedge cost (USD)")
    suptitle(fig, "SLSQP Cost-Variance Hedge Frontier", PROJECT_NAME)
    fig.savefig(OUTPUT_DIR / "slsqp_frontier.png"); plt.close(fig)
    return fr


def _save_execution() -> tuple[object, object]:
    X, T, N = 1_000_000.0, 1.0, 20
    params = AlmgrenChrissParams(gamma=2.5e-7, eta=2.5e-6, eps=1e-4,
                                 sigma=0.20, lam=2e-4)
    opt = optimal_schedule(X, T, N, params)
    twap = twap_schedule(X, T, N, params)
    fig, ax = plt.subplots()
    ax.plot(opt.times, opt.holdings,
            label=f"Almgren-Chriss (kappa={opt.kappa:.3f})", linewidth=2.0)
    ax.plot(twap.times, twap.holdings, label="TWAP baseline",
            linestyle="--", linewidth=1.5)
    ax.set_xlabel("t / T"); ax.set_ylabel("Holdings (shares)")
    ax.legend(loc="best")
    suptitle(fig, "Almgren-Chriss Optimal Trajectory vs TWAP", PROJECT_NAME)
    fig.savefig(OUTPUT_DIR / "execution_schedule.png"); plt.close(fig)
    return opt, twap


def _save_markout() -> pd.DataFrame:
    idx = pd.RangeIndex(5)
    fills = pd.Series([100.05, 100.08, 100.12, 100.14, 100.15], index=idx)
    qty = pd.Series([100, 100, 100, 100, 100], index=idx)
    res = decompose_markout(decision_price=100.00, arrival_price=100.05,
                            fills=fills, shares_per_fill=qty,
                            end_price=100.20, side=1)
    df = pd.DataFrame([{
        "side": res.side, "shares": res.shares,
        "decision_price": res.decision_price, "arrival_price": res.arrival_price,
        "avg_exec_price": res.avg_execution_price, "end_price": res.end_price,
        "total_shortfall_bps": res.total_shortfall_bps,
        "delay_bps": res.delay_bps, "temporary_bps": res.temporary_bps,
        "permanent_bps": res.permanent_bps,
        "adverse_selection_bps": res.adverse_selection_bps,
    }])
    df.to_csv(OUTPUT_DIR / "markout_table.csv", index=False)
    return df


def _print_summary(start, end, leg_returns, levels, fs, lad, fr,
                   opt, twap, mk, timings, elapsed) -> None:
    p = print
    p("=" * 72); p(PROJECT_NAME); p("=" * 72)
    p(f"KERNEL_BACKEND           = {KERNEL_BACKEND}")
    p(f"window                   = {start} -> {end}")
    p(f"leg_returns rows         = {len(leg_returns)}")
    p(f"levels rows x cols       = {levels.shape}\n\nFactsheet rows:")
    for name, row in fs.iterrows():
        p(f"  {name:16s} level_end={row['level_end']:8.2f}  "
          f"ann_ret={row['annual_return']:+.3f}  "
          f"ann_vol={row['annual_vol']:.3f}  "
          f"sharpe={row['sharpe']:+.2f}  max_dd={row['max_drawdown']:+.3f}")
    p(f"\nRisk ladder shape        = {lad.shape} (= {lad.size} cells)")
    p(f"Ladder min/max PnL (USD) = {lad.values.min():,.0f} / {lad.values.max():,.0f}")
    p(f"Hedge frontier points    = {len(fr)}")
    p(f"Frontier cost range      = {fr['hedge_cost'].min():,.0f} -> "
      f"{fr['hedge_cost'].max():,.0f}")
    p(f"\nAlmgren-Chriss kappa     = {opt.kappa:.4f}")
    p(f"  permanent_impact       = {opt.permanent_impact:,.2f}")
    p(f"  temporary_impact       = {opt.temporary_impact:,.2f}")
    p(f"  expected_cost          = {opt.expected_cost:,.2f}")
    p(f"  timing_risk            = {opt.timing_risk:,.2f}")
    p(f"TWAP expected_cost       = {twap.expected_cost:,.2f}")
    p(f"\nMarkout total_shortfall  = {mk['total_shortfall_bps'].iloc[0]:+.2f} bps")
    p("\nTimings (s): " + "  ".join(f"{k}={v:.2f}" for k, v in timings.items()))
    p(f"elapsed                  = {elapsed:.2f} s")
    p("=" * 72)


def main(start: Optional[date] = None, end: Optional[date] = None) -> None:
    start = start or date(2022, 1, 1)
    end = end or date(2024, 6, 30)
    cfg = replace(DEFAULT_RUN, start_date=start, end_date=end)
    ensure_dirs(); set_theme()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    t = time.perf_counter(); legs, leg_returns = _compute_legs(start, end, cfg)
    timings["legs"] = time.perf_counter() - t
    t = time.perf_counter(); levels = _save_levels(legs); fs = _save_factsheet(legs)
    timings["io"] = time.perf_counter() - t
    t = time.perf_counter(); port = _build_demo_portfolio(end); lad = _save_ladder(port)
    timings["ladder"] = time.perf_counter() - t
    t = time.perf_counter(); fr = _save_frontier(port)
    timings["frontier"] = time.perf_counter() - t
    t = time.perf_counter(); opt, twap = _save_execution(); mk = _save_markout()
    timings["exec"] = time.perf_counter() - t
    elapsed = time.perf_counter() - t0
    _print_summary(start, end, leg_returns, levels, fs, lad, fr, opt, twap, mk,
                   timings, elapsed)


if __name__ == "__main__":
    main()
