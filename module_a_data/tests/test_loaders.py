"""Offline-safe tests for the cache layer."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from module_a_data import loaders


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "CACHE_DIR", tmp_path)
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), name="DGS10")
    path = loaders._cache_path("fred", "DGS10", date(2024, 1, 1), date(2024, 1, 31))
    loaders._save_cache(s.to_frame(), path)
    back = loaders._load_cache(path)
    assert back is not None
    assert back.shape == (3, 1)
    assert back.iloc[-1, 0] == pytest.approx(3.0)


def test_cache_key_unique_per_ticker():
    p1 = loaders._cache_path("fred", "DGS10", date(2024, 1, 1), date(2024, 6, 30))
    p2 = loaders._cache_path("fred", "DGS2", date(2024, 1, 1), date(2024, 6, 30))
    assert p1 != p2
