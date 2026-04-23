"""
Project identity and runtime config.

Cross-Asset Strategic Indices (QIS) Calc, Risk & Replication Platform

Independent calculation, risk-management, and algorithmic replication of four
rules-based reference indices (equity vol-target, rates carry, commodity
curve/roll, FX carry) plus a cross-asset vol-target overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


PROJECT_NAME = "Cross-Asset Strategic Indices (QIS) Calc, Risk & Replication Platform"
PROJECT_SLUG = "strategic_indices_qis_platform"
ASSET_CLASS = "Multi-asset (Equities / Rates / Commodities / FX)"
DESK = "QTR Strategic Indices (QIS) - Vice President"

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class RunConfig:
    start_date: date = date(2019, 1, 1)
    end_date: date = date(2024, 12, 31)
    vol_target_equity: float = 0.10        # 10% ex-ante vol for EQ leg
    vol_target_fx: float = 0.08            # 8% ex-ante vol for FX leg
    vol_target_overlay: float = 0.10       # 10% ex-ante vol for overlay
    leverage_cap: float = 1.5
    ewma_lambda: float = 0.94              # RiskMetrics decay
    rebalance_freq: str = "ME"             # month-end
    tcost_bps: float = 2.0                 # per-rebalance transaction cost


DEFAULT_RUN = RunConfig()

FRED_SERIES: dict[str, str] = {
    "DGS1MO": "1M Treasury",
    "DGS3MO": "3M Treasury",
    "DGS2": "2Y Treasury",
    "DGS5": "5Y Treasury",
    "DGS10": "10Y Treasury",
    "DGS30": "30Y Treasury",
    "SOFR": "SOFR",
    "VIXCLS": "VIX",
    "DEXUSEU": "USD/EUR",
    "DEXJPUS": "JPY/USD",
    "DEXUSUK": "USD/GBP",
    "DEXUSAL": "USD/AUD",
    "DCOILWTICO": "WTI Crude",
}

EQUITY_TICKERS: list[str] = ["SPY", "^GSPC"]
COMMODITY_GENERICS: list[str] = ["CL=F", "HO=F", "RB=F", "NG=F"]
FX_FUTURES: list[str] = ["6E=F", "6B=F", "6J=F", "6A=F"]
RATES_FUTURES: list[str] = ["ZT=F", "ZF=F", "ZN=F", "ZB=F"]


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"{PROJECT_NAME}")
    print(f"  slug:  {PROJECT_SLUG}")
    print(f"  desk:  {DESK}")
    print(f"  asset: {ASSET_CLASS}")
    print(f"  run:   {DEFAULT_RUN.start_date} -> {DEFAULT_RUN.end_date}")
    print(f"  cache: {CACHE_DIR}")
