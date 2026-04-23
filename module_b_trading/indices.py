"""Four rules-based QIS indices + a cross-asset vol-target overlay.

Each rulebook returns a DataFrame indexed by date with columns ['ret', 'weight'].
All math is deliberately transparent -- no hidden look-ahead.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_RUN, RunConfig
from module_a_curves.commodity_curve import build_commodity_curves
from module_a_curves.fx_forward_curve import build_fx_forward_panel
from module_a_data.loaders import get_fred_panel, get_yf_close


def _ewma_var(rets: pd.Series, lam: float, seed_window: int = 21) -> pd.Series:
    """RiskMetrics EWMA variance, seeded from first seed_window realised var."""
    r = rets.fillna(0.0).astype(float)
    out = np.full(len(r), np.nan)
    if len(r) < seed_window:
        return pd.Series(out, index=r.index)
    seed = float(r.iloc[:seed_window].var(ddof=0))
    v = seed
    out[seed_window - 1] = v
    vals = r.values
    for i in range(seed_window, len(r)):
        v = lam * v + (1 - lam) * (vals[i - 1] ** 2)
        out[i] = v
    return pd.Series(out, index=r.index).ffill()


def _month_end_mask(idx: pd.DatetimeIndex) -> pd.Series:
    """True on the last observation of each calendar month."""
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).transform("max") == s


@dataclass
class EqVolTargetIndex:
    name: str = "EqVolTarget"

    def compute(self, start: date, end: date, cfg: RunConfig = DEFAULT_RUN) -> pd.DataFrame:
        px = get_yf_close("SPY", start, end).dropna()
        r = np.log(px / px.shift(1)).rename("r_spx")
        var_t = _ewma_var(r, cfg.ewma_lambda)
        sigma_hat = (252.0 * var_t).pow(0.5)
        raw_L = (cfg.vol_target_equity / sigma_hat.replace(0, np.nan)).clip(0, cfg.leverage_cap)
        me = _month_end_mask(r.index)
        L = raw_L.where(me).ffill().fillna(0.0)
        L_prev = L.shift(1).fillna(0.0)
        turnover = (L - L_prev).abs().where(me, 0.0)
        ret = L_prev * r - cfg.tcost_bps * 1e-4 * turnover
        ret = ret.fillna(0.0).rename("ret")
        return pd.DataFrame({"ret": ret, "weight": L_prev.rename("weight")}).dropna()


@dataclass
class RatesCarryIndex:
    name: str = "RatesCarry"

    def compute(self, start: date, end: date, cfg: RunConfig = DEFAULT_RUN) -> pd.DataFrame:
        panel = get_fred_panel(["DGS2", "DGS10"], start, end).dropna()
        slope = (panel["DGS10"] - panel["DGS2"]).rename("slope")
        signal_me = np.sign(slope).where(_month_end_mask(slope.index)).ffill().fillna(0.0)
        # FRED yields are in percent. Convert to decimal, then apply DV01-weighted
        # 2s10s: D_2~2, D_10~9, scaled for a DV01-neutral curve trade.
        d2 = panel["DGS2"].diff() / 100.0     # decimal daily yield change
        d10 = panel["DGS10"].diff() / 100.0
        raw = (-signal_me.shift(1).fillna(0.0) * (2.0 * d2 - 9.0 * d10)).fillna(0.0)
        # Ex-ante vol scale to 5% via EWMA
        var_r = _ewma_var(raw, cfg.ewma_lambda)
        ann = (252.0 * var_r).pow(0.5)
        scale = (0.05 / ann.replace(0, np.nan)).clip(0, cfg.leverage_cap).shift(1).fillna(0.0)
        ret = (scale * raw).fillna(0.0).rename("ret")
        weight = (signal_me * scale).rename("weight")
        return pd.DataFrame({"ret": ret, "weight": weight}).dropna()


@dataclass
class CommodityCurveIndex:
    name: str = "CommodityCurve"

    def compute(self, start: date, end: date, cfg: RunConfig = DEFAULT_RUN) -> pd.DataFrame:
        curves = build_commodity_curves(["CL=F", "HO=F", "RB=F", "NG=F"], start, end)
        leg_rets = {}
        leg_wts = {}
        for t, c in curves.items():
            px = c.front_prices.dropna()
            r = px.pct_change().rename(t)
            ry = c.realized_roll_yield()
            sign_raw = np.sign(ry.shift(1)).reindex(r.index).fillna(0.0)
            me = _month_end_mask(r.index)
            sign_me = sign_raw.where(me).ffill().fillna(0.0)
            var_r = _ewma_var(r, cfg.ewma_lambda)
            ann = (252.0 * var_r).pow(0.5)
            scale = (0.05 / ann.replace(0, np.nan)).clip(0, cfg.leverage_cap)
            w = (sign_me * scale).shift(1).fillna(0.0)
            leg_rets[t] = r
            leg_wts[t] = w
        R = pd.DataFrame(leg_rets).fillna(0.0)
        W = pd.DataFrame(leg_wts).reindex(R.index).fillna(0.0)
        ret = (W * R).mean(axis=1).rename("ret")
        weight = W.mean(axis=1).rename("weight")
        return pd.DataFrame({"ret": ret, "weight": weight}).dropna()


@dataclass
class FxCarryIndex:
    name: str = "FxCarry"

    def compute(self, start: date, end: date, cfg: RunConfig = DEFAULT_RUN) -> pd.DataFrame:
        curves = build_fx_forward_panel(start, end)
        carry_df = pd.concat({c: curves[c].carry() for c in curves}, axis=1).ffill().dropna()
        ret_df = pd.concat({c: curves[c].spot_return() for c in curves}, axis=1).reindex(carry_df.index).fillna(0.0)
        idx = carry_df.index
        me = _month_end_mask(idx)
        # Ranking: +0.5 to top-2, -0.5 to bottom-2
        def rank_row(row: pd.Series) -> pd.Series:
            order = row.rank(method="first")
            w = pd.Series(0.0, index=row.index)
            w[order >= 3] = 0.5
            w[order <= 2] = -0.5
            return w
        raw_wts = carry_df.apply(rank_row, axis=1)
        W = raw_wts.where(me.values[:, None] if isinstance(me, np.ndarray) else me, other=np.nan).ffill().fillna(0.0)
        W_prev = W.shift(1).fillna(0.0)
        carry_prev = carry_df.shift(1).fillna(0.0)
        leg_ret = ret_df + carry_prev / 252.0
        port_ret = (W_prev * leg_ret).sum(axis=1) / 2.0
        # Vol-target to 8%
        var_p = _ewma_var(port_ret, cfg.ewma_lambda)
        ann = (252.0 * var_p).pow(0.5)
        L = (cfg.vol_target_fx / ann.replace(0, np.nan)).clip(0, cfg.leverage_cap).shift(1).fillna(0.0)
        ret = (L * port_ret).fillna(0.0).rename("ret")
        weight = (L * W_prev.abs().sum(axis=1)).rename("weight")
        return pd.DataFrame({"ret": ret, "weight": weight}).dropna()


class XAssetOverlay:
    """Cross-asset 10% vol overlay on equally weighted leg returns."""

    def __init__(self, leg_returns: pd.DataFrame) -> None:
        expected = ["eq", "rates", "commodities", "fx"]
        missing = [c for c in expected if c not in leg_returns.columns]
        if missing:
            raise ValueError(f"Missing leg columns: {missing}")
        self.leg_returns = leg_returns[expected].astype(float).fillna(0.0)

    def compute(self, cfg: RunConfig = DEFAULT_RUN) -> pd.DataFrame:
        port = self.leg_returns.sum(axis=1).rename("port_ret")
        var_p = _ewma_var(port, cfg.ewma_lambda)
        ann = (252.0 * var_p).pow(0.5)
        L = (cfg.vol_target_overlay / ann.replace(0, np.nan)).clip(0, cfg.leverage_cap)
        L_prev = L.shift(1).fillna(0.0)
        ret = (L_prev * port).fillna(0.0).rename("ret")
        return pd.DataFrame({"ret": ret, "weight": L_prev.rename("weight")}).dropna()


Rulebook = Protocol  # structural typing placeholder for API documentation


def _summary(name: str, ret: pd.Series, level: pd.Series) -> str:
    ann_vol = float(ret.std(ddof=0) * np.sqrt(252))
    ann_ret = float(ret.mean() * 252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else float("nan")
    return f"{name:16s} level_end={level.iloc[-1]:7.2f}  realized_vol={ann_vol:.3f}  sharpe={sharpe:+.2f}"


if __name__ == "__main__":
    s, e = date(2022, 1, 1), date(2024, 6, 30)
    rulebooks = [EqVolTargetIndex(), RatesCarryIndex(), CommodityCurveIndex(), FxCarryIndex()]
    leg_frames: dict[str, pd.Series] = {}
    for rb in rulebooks:
        df = rb.compute(s, e)
        lvl = 100.0 * (1.0 + df["ret"]).cumprod()
        print(_summary(rb.name, df["ret"], lvl))
        leg_frames[rb.name] = df["ret"]
    legs = pd.DataFrame({
        "eq": leg_frames["EqVolTarget"],
        "rates": leg_frames["RatesCarry"],
        "commodities": leg_frames["CommodityCurve"],
        "fx": leg_frames["FxCarry"],
    }).dropna()
    ov = XAssetOverlay(legs).compute()
    ov_lvl = 100.0 * (1.0 + ov["ret"]).cumprod()
    print(_summary("XAssetOverlay", ov["ret"], ov_lvl))
