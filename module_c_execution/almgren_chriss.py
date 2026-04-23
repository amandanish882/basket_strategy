from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AlmgrenChrissParams:
    gamma: float
    eta: float
    eps: float
    sigma: float
    lam: float


@dataclass
class ExecutionSchedule:
    times: np.ndarray
    holdings: np.ndarray
    trades: np.ndarray
    expected_cost: float
    cost_variance: float
    permanent_impact: float
    temporary_impact: float
    timing_risk: float
    kappa: float


def _validate(X: float, T: float, N: int) -> None:
    if T <= 0:
        raise ValueError("T must be > 0")
    if N < 1:
        raise ValueError("N must be >= 1")


def _kappa(p: AlmgrenChrissParams) -> float:
    if p.lam <= 0.0 or p.eta <= 0.0:
        return 0.0
    return float(np.sqrt(p.lam * p.sigma * p.sigma / p.eta))


def _cost_components(
    X: float, trades: np.ndarray, holdings: np.ndarray, tau: float, p: AlmgrenChrissParams
) -> Tuple[float, float, float, float, float]:
    # eta_tilde absorbs the discrete-time correction from permanent impact
    eta_tilde = p.eta - 0.5 * p.gamma * tau
    perm = 0.5 * p.gamma * X * X
    temp = p.eps * abs(X) + eta_tilde * float(np.sum(trades * trades))
    # timing risk uses interior holdings only (x_N = 0 contributes nothing)
    var = p.sigma * p.sigma * tau * float(np.sum(holdings[1:] * holdings[1:]))
    timing = float(np.sqrt(max(var, 0.0)))
    return perm, temp, perm + temp, var, timing


def optimal_schedule(
    X: float, T: float, N: int, p: AlmgrenChrissParams
) -> ExecutionSchedule:
    _validate(X, T, N)
    tau = T / N
    kappa = _kappa(p)
    k_idx = np.arange(N + 1, dtype=float)
    times = k_idx * tau

    if p.lam <= 0.0 or kappa * T < 1e-6:
        holdings = X * (1.0 - k_idx / N)
    else:
        holdings = X * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)

    holdings[0] = X
    holdings[-1] = 0.0
    trades = np.abs(holdings[:-1] - holdings[1:])
    perm, temp, exp_cost, var, timing = _cost_components(X, trades, holdings, tau, p)
    return ExecutionSchedule(
        times=times / T,
        holdings=holdings,
        trades=trades,
        expected_cost=exp_cost,
        cost_variance=var,
        permanent_impact=perm,
        temporary_impact=temp,
        timing_risk=timing,
        kappa=kappa,
    )


def twap_schedule(
    X: float, T: float, N: int, p: AlmgrenChrissParams
) -> ExecutionSchedule:
    _validate(X, T, N)
    tau = T / N
    k_idx = np.arange(N + 1, dtype=float)
    holdings = X * (1.0 - k_idx / N)
    holdings[0] = X
    holdings[-1] = 0.0
    trades = np.abs(holdings[:-1] - holdings[1:])
    perm, temp, exp_cost, var, timing = _cost_components(X, trades, holdings, tau, p)
    return ExecutionSchedule(
        times=(k_idx * tau) / T,
        holdings=holdings,
        trades=trades,
        expected_cost=exp_cost,
        cost_variance=var,
        permanent_impact=perm,
        temporary_impact=temp,
        timing_risk=timing,
        kappa=0.0,
    )


def _print_schedule(name: str, sched: ExecutionSchedule) -> None:
    print(f"\n=== {name} (kappa={sched.kappa:.4f}) ===")
    print(f"{'t/T':>8} {'holdings':>16} {'trade':>14}")
    for i in range(len(sched.holdings)):
        tr = sched.trades[i] if i < len(sched.trades) else 0.0
        print(f"{sched.times[i]:>8.3f} {sched.holdings[i]:>16.2f} {tr:>14.2f}")
    print(f"permanent_impact = {sched.permanent_impact:.4f}")
    print(f"temporary_impact = {sched.temporary_impact:.4f}")
    print(f"expected_cost    = {sched.expected_cost:.4f}")
    print(f"timing_risk      = {sched.timing_risk:.4f}")


if __name__ == "__main__":
    params = AlmgrenChrissParams(
        gamma=2.5e-7, eta=2.5e-6, eps=1e-4, sigma=0.20, lam=2e-6
    )
    X, T, N = 1_000_000.0, 1.0, 20
    opt = optimal_schedule(X, T, N, params)
    twap = twap_schedule(X, T, N, params)
    _print_schedule("Almgren-Chriss optimal", opt)
    _print_schedule("TWAP baseline", twap)
    print(
        f"\nOptimal vs TWAP: expected_cost {opt.expected_cost:.2f} vs "
        f"{twap.expected_cost:.2f}; timing_risk {opt.timing_risk:.2f} vs "
        f"{twap.timing_risk:.2f}"
    )
