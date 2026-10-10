"""One validated palette, shared by the static figures and the dashboard.

Colour here is assigned by the *job* it does, not by taste:

* **Categorical** — identity. Hues are taken in fixed slot order and never
  cycled; a series keeps its colour when a filter changes the series count.
* **Sequential** — magnitude. A single hue, light to dark. Never a rainbow,
  because a rainbow implies ordered categories that do not exist.
* **Diverging** — polarity. Two poles with a neutral grey midpoint; used for
  lift, where 1.0 means "independent" and is a genuine zero point.
* **Status** — state. Reserved; never reused as "series 5".

Both light and dark steps are deliberately chosen for their own surface
rather than flipped, and the slot ordering is what keeps adjacent pairs
distinguishable under colour-vision deficiency.

A note on what is *not* here: single-series bar charts get one colour, not a
gradient. Colouring bars by their own height encodes the value twice and
implies a category that does not exist.
"""

from __future__ import annotations

from typing import Sequence

#: Categorical slots, in fixed assignment order.
CATEGORICAL_LIGHT = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#6250d6",  # 7 violet
    "#e34948",  # 8 red
]
CATEGORICAL_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

#: Single-hue blue ramp for continuous magnitude.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

#: Diverging pair with a neutral midpoint (blue <-> red).
DIVERGING = {
    "low": "#1c5cab",
    "mid_light": "#f0efec",
    "mid_dark": "#383835",
    "high": "#d03b3b",
}

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

LIGHT = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "categorical": CATEGORICAL_LIGHT,
}

DARK = {
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "text_primary": "#f0efec",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "categorical": CATEGORICAL_DARK,
}

PALETTE = {"light": LIGHT, "dark": DARK, "sequential": SEQUENTIAL_BLUE, "status": STATUS}

FONT_STACK = ["system-ui", "DejaVu Sans", "Segoe UI", "Helvetica", "sans-serif"]


def colors_for(n: int, mode: str = "light") -> list[str]:
    """Return *n* categorical colours in fixed slot order.

    Past eight series the palette is exhausted. Rather than inventing a ninth
    hue (which would be indistinguishable from an existing one), callers are
    expected to fold the tail into "Other" or facet the chart; this raises so
    that mistake is loud rather than silent.
    """
    slots = PALETTE[mode]["categorical"]
    if n > len(slots):
        raise ValueError(
            f"{n} categorical series requested but only {len(slots)} validated slots exist. "
            "Fold the tail into 'Other', or use small multiples."
        )
    return slots[:n]


def sequential_colors(n: int, low: int = 3, high: int = 11) -> list[str]:
    """Sample *n* evenly spaced steps from the blue ramp, light to dark."""
    if n <= 0:
        return []
    if n == 1:
        return [SEQUENTIAL_BLUE[high]]
    span = high - low
    return [SEQUENTIAL_BLUE[low + round(i * span / (n - 1))] for i in range(n)]


def apply_theme(mode: str = "light", dpi: int = 130) -> dict:
    """Configure matplotlib's rcParams for *mode* and return the palette dict.

    Applies the recessive-chrome rules in one place: hairline gridlines on the
    value axis only, no top/right spines, muted tick labels, and the surface
    colour behind both the figure and the axes.
    """
    import matplotlib as mpl
    from cycler import cycler

    theme = PALETTE[mode]
    mpl.rcParams.update(
        {
            "figure.facecolor": theme["surface"],
            "figure.edgecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "savefig.edgecolor": theme["surface"],
            "savefig.bbox": "tight",
            "savefig.dpi": dpi,
            "figure.dpi": dpi,
            "axes.facecolor": theme["surface"],
            "axes.edgecolor": theme["axis"],
            "axes.labelcolor": theme["text_secondary"],
            "axes.titlecolor": theme["text_primary"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.prop_cycle": cycler(color=theme["categorical"]),
            "grid.color": theme["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "text.color": theme["text_primary"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.bottom": True,
            "ytick.left": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": theme["text_secondary"],
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "patch.linewidth": 0,
        }
    )
    return theme


def style_axes(ax, theme: dict, xgrid: bool = False) -> None:
    """Apply the per-axes chrome rules that rcParams cannot express."""
    ax.set_axisbelow(True)
    ax.grid(axis="y" if not xgrid else "x", color=theme["grid"], linewidth=0.8)
    ax.grid(axis="x" if not xgrid else "y", visible=False)
    for side in ("top", "right", "left" if not xgrid else "bottom"):
        if side in ax.spines:
            ax.spines[side].set_visible(False)
    bottom = "bottom" if not xgrid else "left"
    if bottom in ax.spines:
        ax.spines[bottom].set_color(theme["axis"])
        ax.spines[bottom].set_linewidth(1.0)


def truncate(label: str, width: int = 34) -> str:
    """Shorten a long product name for an axis tick without losing the head."""
    label = str(label)
    return label if len(label) <= width else label[: width - 1].rstrip() + "…"
