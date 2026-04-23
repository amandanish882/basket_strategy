"""Unified Black-Scholes API - C++ kernel preferred, Python fallback."""
from __future__ import annotations

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_kernel_dir = os.path.join(_here, "cpp_kernel")
if _kernel_dir not in sys.path:
    sys.path.insert(0, _kernel_dir)

try:
    import pricing_kernel as _k
    bs_price = _k.bs_price
    bs_delta = _k.bs_delta
    bs_gamma = _k.bs_gamma
    bs_vega = _k.bs_vega
    bs_theta = _k.bs_theta
    bs_implied_vol = _k.bs_implied_vol
    KERNEL_BACKEND = "cpp"
except ImportError:
    from .bs_python import (
        bs_price,
        bs_delta,
        bs_gamma,
        bs_vega,
        bs_theta,
        bs_implied_vol,
    )
    KERNEL_BACKEND = "python"

from .plot_style import LEG_COLOURS, plot_index_levels, set_theme, suptitle  # noqa: E402

__all__ = [
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "bs_implied_vol",
    "KERNEL_BACKEND",
    "LEG_COLOURS",
    "plot_index_levels",
    "set_theme",
    "suptitle",
]
