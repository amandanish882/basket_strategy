"""
Publication-grade figure generator for the Cross-Asset Strategic Indices platform.

Produces 13 PNGs under outputs/ covering the full pipeline:
    hero_3d_ladder        - 3D spot/vol PnL surface (hero image)
    risk_ladder_2d        - 2D heatmap of the same grid
    index_levels          - 5-series index levels
    drawdown_curves       - 5 drawdown series
    leg_correlation       - 5x5 correlation heatmap
    rolling_sharpe        - 63-day rolling Sharpe on the overlay
    zero_curve            - SOFR/UST bootstrapped zero + DF (2024-06-28)
    commodity_roll_ts     - realized roll yield for CL/HO/RB/NG
    fx_carry_ts           - CIP carry (r_f - r_d) for EUR/GBP/JPY/AUD
    slsqp_frontier        - SLSQP cost vs residual stdev
    execution_schedule    - Almgren-Chriss vs TWAP trajectory
    markout_waterfall     - horizontal bar chart of markout decomposition
    summary_dashboard     - 2x3 panel combining several views

Run from the project root:
    python3 make_figures.py
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

from config import DEFAULT_RUN, OUTPUT_DIR, PROJECT_NAME, ensure_dirs  # noqa: E402
from module_a_curves.commodity_curve import build_commodity_curves  # noqa: E402
from module_a_curves.curve_bootstrapper import build_curve_from_fred  # noqa: E402
from module_a_curves.fx_forward_curve import build_fx_forward_panel  # noqa: E402
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
from shared import LEG_COLOURS, KERNEL_BACKEND, plot_index_levels, set_theme, suptitle  # noqa: E402


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


START = date(2022, 1, 1)
END = date(2024, 6, 30)
CURVE_ASOF = date(2024, 6, 28)

RULEBOOKS = [
    ("EqVolTarget", EqVolTargetIndex()),
    ("RatesCarry", RatesCarryIndex()),
    ("CommodityCurve", CommodityCurveIndex()),
    ("FxCarry", FxCarryIndex()),
]
_LEG_KEY = {"EqVolTarget": "eq", "RatesCarry": "rates",
            "CommodityCurve": "commodities", "FxCarry": "fx"}


# ---------- helpers ---------- #
def _savefig(fig, name: str) -> Path:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024.0
    print(f"  [{size_kb:7.1f} KB] {path.name}")
    return path


def _compute_legs(cfg) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    legs: dict[str, pd.DataFrame] = {}
    for n, rb in RULEBOOKS:
        legs[n] = IndexCalculator(rulebook=rb, name=n).run(START, END, cfg)
    leg_returns = pd.concat({_LEG_KEY[n]: legs[n]["ret"] for n in legs}, axis=1).dropna()
    overlay_df = XAssetOverlay(leg_returns).compute(cfg)
    overlay_df["level"] = 100.0 * (1.0 + overlay_df["ret"].fillna(0.0)).cumprod()
    legs["XAssetOverlay"] = overlay_df[["ret", "level", "weight"]]
    return legs, leg_returns


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


# ---------- figures ---------- #
def fig_hero_3d_ladder(ladder: pd.DataFrame) -> None:
    spot_bps = np.asarray(ladder.index, dtype=float)
    vol_bps = np.asarray(ladder.columns, dtype=float)
    V, S = np.meshgrid(vol_bps, spot_bps)
    Z = ladder.values / 1e6  # USD millions for readable colourbar

    fig = plt.figure(figsize=(10.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        S, V, Z,
        cmap=cm.RdYlGn,
        linewidth=0.0,
        antialiased=True,
        rstride=1,
        cstride=1,
        alpha=0.92,
        edgecolor="#555555",
    )
    ax.contour(S, V, Z, zdir="z", offset=Z.min(), cmap=cm.RdYlGn, alpha=0.55, levels=10)
    ax.set_xlabel("Spot shock (bps)", labelpad=8)
    ax.set_ylabel("Vol shock (bps)", labelpad=8)
    ax.set_zlabel("PnL ($M)", labelpad=6)
    ax.view_init(elev=30, azim=-60)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08)
    cbar.set_label("PnL ($M)")
    suptitle(
        fig,
        "Cross-Asset 441-Cell Spot x Vol PnL Surface",
        "21 x 21 full-revaluation grid - Black-Scholes C++ kernel",
    )
    _savefig(fig, "hero_3d_ladder.png")


def fig_risk_ladder_2d(ladder: pd.DataFrame) -> None:
    spot_bps = np.asarray(ladder.index, dtype=float)
    vol_bps = np.asarray(ladder.columns, dtype=float)
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    im = ax.imshow(
        ladder.values,
        origin="lower",
        aspect="auto",
        extent=(vol_bps[0], vol_bps[-1], spot_bps[0], spot_bps[-1]),
        cmap="RdYlGn",
    )
    ax.set_xlabel("Vol shock (bps)")
    ax.set_ylabel("Spot shock (bps)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PnL (USD)")
    suptitle(fig, "21 x 21 Spot x Vol Risk Ladder", "Full-revaluation PnL (USD)")
    _savefig(fig, "risk_ladder_2d.png")
    # Keep the legacy filename in sync.
    fig2, ax2 = plt.subplots(figsize=(9.0, 6.0))
    im2 = ax2.imshow(
        ladder.values, origin="lower", aspect="auto",
        extent=(vol_bps[0], vol_bps[-1], spot_bps[0], spot_bps[-1]),
        cmap="RdYlGn",
    )
    ax2.set_xlabel("Vol shock (bps)")
    ax2.set_ylabel("Spot shock (bps)")
    fig2.colorbar(im2, ax=ax2, label="PnL (USD)")
    suptitle(fig2, "Risk Ladder - 21x21 Spot x Vol Revaluation", PROJECT_NAME)
    _savefig(fig2, "risk_ladder.png")


def fig_index_levels(legs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    levels = pd.concat({n: df["level"] for n, df in legs.items()}, axis=1).ffill()
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    plot_index_levels(ax, levels)
    suptitle(fig, "Index Levels - 4 Rulebooks + Cross-Asset Overlay",
             f"{START.isoformat()} -> {END.isoformat()}  -  base = 100")
    _savefig(fig, "index_levels.png")
    return levels


def fig_drawdown_curves(legs: Dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    for name, df in legs.items():
        level = df["level"].ffill()
        dd = level / level.cummax() - 1.0
        lw = 2.0 if name == "XAssetOverlay" else 1.2
        ax.plot(dd.index, dd.values * 100.0, label=name,
                color=LEG_COLOURS.get(name), linewidth=lw)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.axhline(0, color="#888888", linewidth=0.7, alpha=0.4)
    ax.legend(loc="lower left", ncol=2)
    suptitle(fig, "Drawdown Curves - 4 Rulebooks + Cross-Asset Overlay",
             "Peak-to-trough drawdown, %")
    _savefig(fig, "drawdown_curves.png")


def fig_leg_correlation(legs: Dict[str, pd.DataFrame]) -> None:
    rets = pd.concat({n: df["ret"] for n, df in legs.items()}, axis=1).dropna()
    order = ["EqVolTarget", "RatesCarry", "CommodityCurve", "FxCarry", "XAssetOverlay"]
    rets = rets[[c for c in order if c in rets.columns]]
    corr = rets.corr()
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iat[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    color="white" if abs(val) > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    suptitle(fig, "Leg Return Correlation Matrix",
             "Daily returns, 4 legs + overlay")
    _savefig(fig, "leg_correlation.png")


def fig_rolling_sharpe(overlay: pd.DataFrame, window: int = 63) -> None:
    r = overlay["ret"].astype(float)
    mu = r.rolling(window).mean() * 252.0
    sigma = r.rolling(window).std(ddof=0) * np.sqrt(252.0)
    rs = (mu / sigma.replace(0, np.nan)).dropna()
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.plot(rs.index, rs.values, color="#1f4e79", linewidth=1.5)
    ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.5)
    ax.fill_between(rs.index, rs.values, 0.0,
                     where=rs.values >= 0, color="#2ca02c", alpha=0.18)
    ax.fill_between(rs.index, rs.values, 0.0,
                     where=rs.values < 0, color="#d62728", alpha=0.18)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{window}-day rolling Sharpe")
    suptitle(fig, "Cross-Asset Overlay - 63-Day Rolling Sharpe",
             "Annualised, window = 63 business days")
    _savefig(fig, "rolling_sharpe.png")


def fig_zero_curve(as_of: date = CURVE_ASOF) -> None:
    curve = build_curve_from_fred(as_of)
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30])
    zeros = np.array([curve.zero_rate(t) * 100.0 for t in tenors])
    dfs = np.array([curve.df(t) for t in tenors])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    ax1, ax2 = axes
    ax1.plot(tenors, zeros, marker="o", linewidth=1.6, color="#1f4e79")
    ax1.scatter(tenors, zeros, color="#1f4e79", s=30, zorder=5)
    ax1.set_xlabel("Tenor (years)")
    ax1.set_ylabel("Zero rate (%)")
    ax1.set_title("Zero rates")
    ax2.plot(tenors, dfs, marker="o", linewidth=1.6, color="#d62728")
    ax2.scatter(tenors, dfs, color="#d62728", s=30, zorder=5)
    ax2.set_xlabel("Tenor (years)")
    ax2.set_ylabel("Discount factor")
    ax2.set_title("Discount factors")
    suptitle(fig, f"SOFR/UST Bootstrapped Zero Curve - {as_of.isoformat()}",
             "Sequential bootstrap, log-linear interpolation on ln D(t)")
    _savefig(fig, "zero_curve.png")


def fig_commodity_roll_ts() -> None:
    curves = build_commodity_curves(["CL=F", "HO=F", "RB=F", "NG=F"], START, END)
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    for t, c in curves.items():
        ry = c.realized_roll_yield().dropna() * 100.0
        ax.plot(ry.index, ry.values, label=t, linewidth=1.2)
    ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Realised roll yield (%, annualised)")
    ax.legend(loc="best", ncol=4)
    suptitle(fig, "Commodity Realised Roll Yield - 63-Day Window",
             "(252/63) * log(F_t / F_{t-63}), CL/HO/RB/NG front generics")
    _savefig(fig, "commodity_roll_ts.png")


def fig_fx_carry_ts() -> None:
    curves = build_fx_forward_panel(START, END)
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    colour_map = {"EUR": "#1f4e79", "GBP": "#9c0000",
                  "JPY": "#2ca02c", "AUD": "#ff7f0e"}
    for ccy, c in curves.items():
        carry = c.carry().dropna() * 100.0
        ax.plot(carry.index, carry.values, label=ccy,
                color=colour_map.get(ccy), linewidth=1.25)
    ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Carry r_f - r_d (%, annualised)")
    ax.legend(loc="best", ncol=4)
    suptitle(fig, "G10-Subset FX CIP Carry - r_f - r_d",
             "Foreign-minus-domestic 3-month interbank rate, FRED sources")
    _savefig(fig, "fx_carry_ts.png")


def fig_slsqp_frontier(port: Portfolio) -> pd.DataFrame:
    S = port.equity_options[0].S
    r = port.equity_options[0].r
    hedges = [
        EqOptionPosition(S=S, K=round(S), T=0.25, r=r, q=0.015, sigma=0.20,
                         is_call=False, qty=10.0),
        RatesPosition(tenor_years=2.0, dv01_per_mm=190.0, notional_mm=1.0),
        RatesPosition(tenor_years=5.0, dv01_per_mm=460.0, notional_mm=1.0),
        RatesPosition(tenor_years=10.0, dv01_per_mm=820.0, notional_mm=1.0),
        RatesPosition(tenor_years=30.0, dv01_per_mm=1900.0, notional_mm=1.0),
    ]
    fr = hedge_frontier(port, hedges, lambdas=np.logspace(-10, -2, 12),
                        max_hedge_notional=100.0)
    baseline_var = float(fr["residual_var"].max())
    fr["variance_reduction_pct"] = 100.0 * (1.0 - fr["residual_var"] / baseline_var)
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    sc = ax.scatter(fr["variance_reduction_pct"], fr["hedge_cost"],
                    c=np.log10(fr["lambda"]), cmap="viridis", s=55, edgecolor="black")
    label_offsets = {0: (6, 6), 3: (6, -12), 6: (-44, 8),
                     9: (-44, -12), len(fr) - 1: (6, 6)}
    for i, (_, row) in enumerate(fr.iterrows()):
        if i not in label_offsets:
            continue
        ax.annotate(f"l={row['lambda']:.1e}",
                    (row["variance_reduction_pct"], row["hedge_cost"]),
                    fontsize=6, alpha=0.8, xytext=label_offsets[i],
                    textcoords="offset points",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0})
    ax.plot(fr["variance_reduction_pct"], fr["hedge_cost"], color="#666666",
            alpha=0.35, linewidth=0.8, zorder=1)
    fig.colorbar(sc, ax=ax, label="log10(lambda)")
    ax.set_xlabel("Variance reduction vs unhedged proxy (%)")
    ax.set_ylabel("Hedge cost (USD)")
    suptitle(fig, "SLSQP Cost-Variance Hedge Frontier",
             "12-point lambda sweep, x-axis normalised to make the trade-off visible")
    _savefig(fig, "slsqp_frontier.png")
    return fr


def fig_execution_schedule() -> tuple[object, object]:
    X, T, N = 1_000_000.0, 1.0, 20
    params = AlmgrenChrissParams(gamma=2.5e-7, eta=2.5e-6, eps=1e-4,
                                 sigma=0.20, lam=2e-4)
    opt = optimal_schedule(X, T, N, params)
    twap = twap_schedule(X, T, N, params)
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.plot(opt.times, opt.holdings / 1e3,
            label=f"Almgren-Chriss (kappa={opt.kappa:.3f})",
            color="#1f4e79", linewidth=2.2)
    ax.plot(twap.times, twap.holdings / 1e3, label="TWAP baseline",
            color="#d62728", linestyle="--", linewidth=1.6)
    ax.fill_between(opt.times, opt.holdings / 1e3, twap.holdings / 1e3,
                    alpha=0.12, color="#1f4e79")
    ax.set_xlabel("t / T")
    ax.set_ylabel("Holdings (thousand shares)")
    ax.legend(loc="best")
    suptitle(fig, "Almgren-Chriss Optimal Trajectory vs TWAP",
             f"X = 1,000,000 shares, N = 20, lambda = 2e-4, sigma = 0.20")
    _savefig(fig, "execution_schedule.png")
    return opt, twap


def fig_markout_waterfall() -> pd.DataFrame:
    idx = pd.RangeIndex(5)
    fills = pd.Series([100.05, 100.08, 100.12, 100.14, 100.15], index=idx)
    qty = pd.Series([100, 100, 100, 100, 100], index=idx)
    res = decompose_markout(decision_price=100.00, arrival_price=100.05,
                            fills=fills, shares_per_fill=qty,
                            end_price=100.20, side=1)
    components = [
        ("Delay", res.delay_bps),
        ("Temporary", res.temporary_bps),
        ("Permanent", res.permanent_bps),
        ("Adverse Selection", res.adverse_selection_bps),
        ("Total Shortfall", res.total_shortfall_bps),
    ]
    labels = [c[0] for c in components]
    values = [c[1] for c in components]
    colours = ["#1f4e79", "#ff7f0e", "#d62728", "#9467bd", "#333333"]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colours, edgecolor="black", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    for i, v in enumerate(values):
        ax.text(v + (0.2 if v >= 0 else -0.2), i, f"{v:+.2f} bps",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlabel("Impact on trader (bps, positive = adverse)")
    suptitle(fig, "Implementation Shortfall Decomposition",
             "decision=100.00, arrival=100.05, avg exec=100.108, end=100.20")
    _savefig(fig, "markout_waterfall.png")
    return pd.DataFrame([{
        "delay_bps": res.delay_bps,
        "temporary_bps": res.temporary_bps,
        "permanent_bps": res.permanent_bps,
        "adverse_selection_bps": res.adverse_selection_bps,
        "total_shortfall_bps": res.total_shortfall_bps,
    }])


def fig_summary_dashboard(
    legs: Dict[str, pd.DataFrame],
    ladder: pd.DataFrame,
    frontier: pd.DataFrame,
    opt, twap,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))

    # 1 - Index levels
    ax = axes[0, 0]
    for name, df in legs.items():
        lw = 2.0 if name == "XAssetOverlay" else 1.1
        ax.plot(df.index, df["level"], label=name, color=LEG_COLOURS.get(name), linewidth=lw)
    ax.set_title("Index levels (base = 100)")
    ax.legend(loc="best", ncol=2, fontsize=7)
    ax.set_xlabel("Date")

    # 2 - Drawdowns
    ax = axes[0, 1]
    for name, df in legs.items():
        level = df["level"].ffill()
        dd = (level / level.cummax() - 1.0) * 100.0
        lw = 2.0 if name == "XAssetOverlay" else 1.0
        ax.plot(dd.index, dd.values, label=name,
                color=LEG_COLOURS.get(name), linewidth=lw)
    ax.axhline(0, color="#888888", linewidth=0.7, alpha=0.4)
    ax.set_title("Drawdowns (%)")
    ax.set_xlabel("Date")

    # 3 - Leg correlation
    ax = axes[0, 2]
    rets = pd.concat({n: df["ret"] for n, df in legs.items()}, axis=1).dropna()
    order = ["EqVolTarget", "RatesCarry", "CommodityCurve", "FxCarry", "XAssetOverlay"]
    rets = rets[[c for c in order if c in rets.columns]]
    corr = rets.corr()
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels([c[:4] for c in corr.columns], rotation=0, fontsize=8)
    ax.set_yticklabels([c[:4] for c in corr.index], fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iat[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr.iat[i, j]) > 0.5 else "black")
    ax.set_title("Correlation matrix")

    # 4 - 2D risk ladder
    ax = axes[1, 0]
    spot_bps = np.asarray(ladder.index, dtype=float)
    vol_bps = np.asarray(ladder.columns, dtype=float)
    im2 = ax.imshow(ladder.values, origin="lower", aspect="auto",
                    extent=(vol_bps[0], vol_bps[-1], spot_bps[0], spot_bps[-1]),
                    cmap="RdYlGn")
    ax.set_xlabel("Vol shock (bps)")
    ax.set_ylabel("Spot shock (bps)")
    ax.set_title("21 x 21 spot x vol ladder")
    fig.colorbar(im2, ax=ax, shrink=0.85, label="PnL (USD)")

    # 5 - Frontier
    ax = axes[1, 1]
    if "variance_reduction_pct" not in frontier:
        baseline_var = float(frontier["residual_var"].max())
        frontier = frontier.assign(
            variance_reduction_pct=100.0 * (1.0 - frontier["residual_var"] / baseline_var)
        )
    sc = ax.scatter(frontier["variance_reduction_pct"], frontier["hedge_cost"],
                    c=np.log10(frontier["lambda"]), cmap="viridis",
                    s=40, edgecolor="black")
    ax.plot(frontier["variance_reduction_pct"], frontier["hedge_cost"],
            color="#666666", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Variance reduction (%)")
    ax.set_ylabel("Hedge cost (USD)")
    ax.set_title("SLSQP frontier")
    fig.colorbar(sc, ax=ax, shrink=0.85, label="log10(lambda)")

    # 6 - Execution schedule
    ax = axes[1, 2]
    ax.plot(opt.times, opt.holdings / 1e3,
            label=f"AC (k={opt.kappa:.2f})",
            color="#1f4e79", linewidth=2.0)
    ax.plot(twap.times, twap.holdings / 1e3, label="TWAP",
            color="#d62728", linestyle="--", linewidth=1.4)
    ax.legend(loc="best")
    ax.set_xlabel("t / T")
    ax.set_ylabel("Holdings (k shares)")
    ax.set_title("AC optimal vs TWAP")

    suptitle(fig, "Cross-Asset Indices Platform - Summary Dashboard",
             f"{START.isoformat()} -> {END.isoformat()}   |   kernel = {KERNEL_BACKEND}")
    _savefig(fig, "summary_dashboard.png")


def main() -> None:
    ensure_dirs()
    set_theme()
    print(f"KERNEL_BACKEND = {KERNEL_BACKEND}")
    print(f"Window: {START} -> {END}   |   Output dir: {OUTPUT_DIR}")
    print("-" * 72)

    cfg = replace(DEFAULT_RUN, start_date=START, end_date=END)

    print("Computing 4 rulebooks + overlay ...")
    legs, leg_returns = _compute_legs(cfg)

    # Save a factsheet artefact too, so README tables stay in sync.
    fs = pd.DataFrame(
        [FactsheetBuilder.build(df, n) for n, df in legs.items()]
    ).set_index("name")
    fs.to_csv(OUTPUT_DIR / "factsheet.csv")
    levels = pd.concat({n: df["level"] for n, df in legs.items()}, axis=1).ffill()
    levels.to_csv(OUTPUT_DIR / "index_levels.csv")

    print("Building demo portfolio ...")
    port = _build_demo_portfolio(END)

    print("Rendering figures ...")
    ladder = spot_vol_ladder(port,
                             spot_bps=np.linspace(-1000, 1000, 21),
                             vol_bps=np.linspace(-500, 500, 21))

    fig_hero_3d_ladder(ladder)
    fig_risk_ladder_2d(ladder)
    fig_index_levels(legs)
    fig_drawdown_curves(legs)
    fig_leg_correlation(legs)
    fig_rolling_sharpe(legs["XAssetOverlay"])
    fig_zero_curve(CURVE_ASOF)
    fig_commodity_roll_ts()
    fig_fx_carry_ts()
    frontier = fig_slsqp_frontier(port)
    opt, twap = fig_execution_schedule()
    fig_markout_waterfall()
    fig_summary_dashboard(legs, ladder, frontier, opt, twap)

    print("-" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
