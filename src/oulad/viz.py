"""Shared plotting style, so every figure in the project looks like the others.

Two reasons this is a module rather than a snippet copied into each notebook.
Consistency: a reader learns "orange means at-risk" once and it holds everywhere.
And correctness: the colours below were checked with a colour-vision-deficiency
validator, so the two series stay distinguishable for readers with the common
forms of colour blindness. Picking colours by eye is how that gets broken.

The one rule worth remembering when you add a chart: **colour means identity,
not size.** Do not shade bars darker-where-bigger — the bar length already says
that, and it burns the only channel you have left.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots. Assigned in fixed order and never cycled; if you ever need
# a ninth series, group the tail into "Other" instead of inventing a colour.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
VIOLET = "#4a3aa7"

CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]

# The project's fixed meaning for the two classes. Used everywhere.
NOT_AT_RISK = BLUE
AT_RISK = ORANGE

OUTCOME_COLORS = {
    "Pass": BLUE,
    "Distinction": AQUA,
    "Fail": ORANGE,
    "Withdrawn": MAGENTA,
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e2e1dd"


def use_project_style() -> None:
    """Apply the project's matplotlib defaults. Call once per notebook."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "figure.figsize": (8.5, 4.6),
            # Recessive axes: the data should be the darkest thing on screen.
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "axes.labelcolor": INK_SOFT,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "font.size": 10,
        }
    )


def label_bars(ax, fmt: str = "{:,.0f}", padding: float = 3) -> None:
    """Write the value on the end of each bar.

    Direct labels beat a y-axis a reader has to trace back to. Used selectively:
    on bar charts with few categories, not on anything dense.
    """
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=padding, fontsize=9, color=INK_SOFT)


def note(ax, text: str) -> None:
    """Put a short interpretive note under a chart."""
    ax.annotate(
        text,
        xy=(0, -0.20),
        xycoords="axes fraction",
        fontsize=8.5,
        color=INK_SOFT,
        va="top",
        wrap=True,
    )


def show(fig=None) -> None:
    """Tidy layout and render."""
    (fig or plt.gcf()).tight_layout()
    plt.show()
