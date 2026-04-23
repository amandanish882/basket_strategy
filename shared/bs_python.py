"""Pure-Python Black-Scholes fallback with scalar API matching C++ kernel."""
from __future__ import annotations

import math
from scipy.stats import norm


def _d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool = True) -> float:
    """Black-Scholes European option price with continuous dividend yield q."""
    if T <= 0.0 or sigma <= 0.0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    if is_call:
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool = True) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    df_q = math.exp(-q * T)
    return df_q * norm.cdf(d1) if is_call else df_q * (norm.cdf(d1) - 1.0)


def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    df_q = math.exp(-q * T)
    return df_q * norm.pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Vega per 1.00 of vol (divide by 100 for per-1%-point)."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    df_q = math.exp(-q * T)
    return S * df_q * norm.pdf(d1) * math.sqrt(T)


def bs_theta(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool = True) -> float:
    """Theta per year."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    term1 = -(S * df_q * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if is_call:
        return term1 - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
    return term1 + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)


def bs_implied_vol(price: float, S: float, K: float, T: float, r: float, q: float, is_call: bool = True) -> float:
    """Newton-Raphson implied vol, seed 0.2, 50 iter max, 1e-8 tol."""
    sigma = 0.2
    tol = 1e-8
    for _ in range(50):
        p = bs_price(S, K, T, r, q, sigma, is_call)
        v = bs_vega(S, K, T, r, q, sigma)
        diff = p - price
        if abs(diff) < tol:
            return sigma
        if v < 1e-12:
            sigma = sigma * 1.5 + 1e-4
            continue
        nxt = sigma - diff / v
        if nxt <= 0.0:
            nxt = sigma / 2.0
        elif nxt > 5.0:
            nxt = 5.0
        sigma = nxt
    return sigma
