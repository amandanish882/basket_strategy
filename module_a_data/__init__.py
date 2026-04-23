"""Data ingestion: free, public, API-first loaders (FRED + yfinance)."""
from .loaders import (
    get_fred_series,
    get_fred_panel,
    get_yf_close,
    get_yf_panel,
    get_futures_chain_front,
    load_all_inputs,
)

__all__ = [
    "get_fred_series",
    "get_fred_panel",
    "get_yf_close",
    "get_yf_panel",
    "get_futures_chain_front",
    "load_all_inputs",
]
