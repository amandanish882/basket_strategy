from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module_c_execution.almgren_chriss import (
    AlmgrenChrissParams,
    optimal_schedule,
)
from module_c_execution.markout import decompose_markout


def _params() -> AlmgrenChrissParams:
    return AlmgrenChrissParams(
        gamma=2.5e-7, eta=2.5e-6, eps=1e-4, sigma=0.20, lam=2e-6
    )


def test_ac_schedule_declines() -> None:
    X, T, N = 1_000_000.0, 1.0, 20
    sched = optimal_schedule(X, T, N, _params())
    assert sched.holdings[0] == pytest.approx(X)
    assert sched.holdings[-1] == pytest.approx(0.0, abs=1e-6)
    diffs = np.diff(sched.holdings)
    assert np.all(diffs <= 1e-6)


def test_ac_trades_sum_to_X() -> None:
    X, T, N = 1_000_000.0, 1.0, 20
    sched = optimal_schedule(X, T, N, _params())
    assert float(np.sum(sched.trades)) == pytest.approx(X, rel=1e-9, abs=1e-6)


def test_markout_signs_buy() -> None:
    idx = pd.RangeIndex(3)
    qty = pd.Series([100, 100, 100], index=idx)
    adverse = decompose_markout(
        decision_price=100.00,
        arrival_price=100.02,
        fills=pd.Series([100.05, 100.08, 100.10], index=idx),
        shares_per_fill=qty,
        end_price=100.15,
        side=1,
    )
    assert adverse.total_shortfall_bps > 0.0

    favorable = decompose_markout(
        decision_price=100.00,
        arrival_price=99.98,
        fills=pd.Series([99.95, 99.92, 99.90], index=idx),
        shares_per_fill=qty,
        end_price=99.85,
        side=1,
    )
    assert favorable.total_shortfall_bps < 0.0
