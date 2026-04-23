"""Matplotlib theme and helpers for consistent look across demo artefacts."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt


LEG_COLOURS = {
    "EqVolTarget": "#1f77b4",
    "RatesCarry": "#2ca02c",
    "CommodityCurve": "#d62728",
    "FxCarry": "#9467bd",
    "XAssetOverlay": "#111111",
}


def set_theme() -> None:
    """Apply a consistent whitegrid theme via matplotlib rcParams."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": (8.0, 5.0),
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "savefig.bbox": "tight",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.4,
        "font.family": "sans-serif",
    })


def suptitle(fig, title: str, subtitle: Optional[str] = None) -> None:
    """Put `title` large at the top and an optional small subtitle below."""
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0, 1, 0.94))


def plot_index_levels(ax, levels) -> None:
    """Plot rulebook levels with the shared publication colour scheme."""
    for col in levels.columns:
        lw = 2.0 if col == "XAssetOverlay" else 1.3
        ax.plot(levels.index, levels[col], label=col,
                color=LEG_COLOURS.get(col), linewidth=lw)
    ax.set_xlabel("Date")
    ax.set_ylabel("Index Level (base = 100)")
    ax.axhline(100, color="#888888", linewidth=0.7, alpha=0.4)
    ax.legend(loc="best", ncol=2)
