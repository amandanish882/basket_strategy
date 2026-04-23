"""
Public data loaders for FRED and yfinance with a CSV cache.

Every loader returns a tz-naive pandas Series/DataFrame indexed by calendar date.
Results are cached to .cache/ as CSV keyed on (source, ticker, start, end) so
subsequent runs are offline-instant.

FRED is accessed via the public graph CSV endpoint:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>
"""

from __future__ import annotations

import hashlib
import sys
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CACHE_DIR,
    COMMODITY_GENERICS,
    DEFAULT_RUN,
    EQUITY_TICKERS,
    FRED_SERIES,
    FX_FUTURES,
    RATES_FUTURES,
    ensure_dirs,
)


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"


def _cache_path(source: str, ticker: str, start: date, end: date) -> Path:
    key = f"{source}_{ticker}_{start.isoformat()}_{end.isoformat()}"
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    safe_ticker = ticker.replace("/", "-").replace("=", "").replace("^", "")
    return CACHE_DIR / f"{source}_{safe_ticker}_{digest}.csv"


def _load_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    ensure_dirs()
    df.to_csv(path)


def get_fred_series(code: str, start: date, end: date) -> pd.Series:
    """Daily FRED series via the public fredgraph.csv endpoint. Cached."""
    path = _cache_path("fred", code, start, end)
    cached = _load_cache(path)
    if cached is not None:
        return cached.iloc[:, 0].rename(code)
    resp = requests.get(FRED_CSV_URL.format(code=code), timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    date_col, value_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.loc[str(start):str(end)].dropna()
    s = df[value_col].rename(code).astype(float)
    _save_cache(s.to_frame(), path)
    return s


def get_fred_panel(codes: Iterable[str], start: date, end: date) -> pd.DataFrame:
    cols = [get_fred_series(c, start, end) for c in codes]
    return pd.concat(cols, axis=1).sort_index()


def get_yf_close(ticker: str, start: date, end: date) -> pd.Series:
    """Daily adjusted close via yfinance. Cached."""
    path = _cache_path("yf", ticker, start, end)
    cached = _load_cache(path)
    if cached is not None:
        return cached.iloc[:, 0].rename(ticker)
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned empty frame for {ticker}")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s.astype(float).rename(ticker)
    _save_cache(s.to_frame(), path)
    return s


def get_yf_panel(tickers: Iterable[str], start: date, end: date) -> pd.DataFrame:
    cols = [get_yf_close(t, start, end) for t in tickers]
    return pd.concat(cols, axis=1).sort_index()


def get_futures_chain_front(ticker: str, start: date, end: date) -> pd.Series:
    """
    Generic front-month continuous price. yfinance returns a back-adjusted
    continuous series for tickers like CL=F, so we treat the close as the
    front-generic level directly. The 'back' series is modelled in
    commodity_curve.py via a deterministic roll assumption.
    """
    return get_yf_close(ticker, start, end)


def load_all_inputs(start: date | None = None, end: date | None = None) -> dict[str, pd.DataFrame]:
    """One-shot loader used by run_full_demo.py. Returns a dict of panels."""
    start = start or DEFAULT_RUN.start_date
    end = end or DEFAULT_RUN.end_date
    return {
        "fred": get_fred_panel(FRED_SERIES.keys(), start, end),
        "equity": get_yf_panel(EQUITY_TICKERS, start, end),
        "commodities": get_yf_panel(COMMODITY_GENERICS, start, end),
        "fx_futures": get_yf_panel(FX_FUTURES, start, end),
        "rates_futures": get_yf_panel(RATES_FUTURES, start, end),
    }


if __name__ == "__main__":
    ensure_dirs()
    start = date(2024, 1, 1)
    end = date(2024, 6, 30)
    print("Smoke test -- loading a few series...")
    try:
        dgs10 = get_fred_series("DGS10", start, end)
        print(f"  FRED DGS10  {len(dgs10):4d} rows, last={dgs10.dropna().iloc[-1]:.3f}")
    except Exception as e:
        print(f"  FRED DGS10  FAILED: {type(e).__name__}: {e}")
    try:
        spy = get_yf_close("SPY", start, end)
        print(f"  yf   SPY    {len(spy):4d} rows, last={spy.dropna().iloc[-1]:.2f}")
    except Exception as e:
        print(f"  yf   SPY    FAILED: {type(e).__name__}: {e}")
