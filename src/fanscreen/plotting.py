"""Shared figure style."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd6"
NAVY = "#1a2a6c"

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}


def use_style() -> None:
    plt.rcParams.update(STYLE)


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.02, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right", va="bottom")


def save(fig, folder: Path, name: str) -> None:
    """Write the figure as PNG (300 dpi) and PDF, then close it."""
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{ext}")
    plt.close(fig)
