"""
Plot Utilities for CM-MTD Paper Figure Reproduction.

Provides consistent publication-quality styling matching IEEE JSAC format:
  - Font sizes, line widths, and tick sizes matching published figures
  - Color palette consistent across all figures
  - Export to PNG (300 DPI), PDF (vector), and SVG
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

logger = logging.getLogger("cm_mtd.plot")

# ─── Global Style Configuration ──────────────────────────────────────────────

# Publication color palette (matching paper colors from description)
COLORS: Dict[str, str] = {
    "CM_MTD":      "#d62728",   # Red (proposed — most prominent)
    "RRT_FRVM":    "#1f77b4",   # Blue
    "DQN_RM_FRVM": "#2ca02c",   # Green
    "STATIC":      "#7f7f7f",   # Gray
    "HAM_ONLY":    "#ff7f0e",   # Orange
    "RM_ONLY":     "#9467bd",   # Purple
}

LINE_STYLES: Dict[str, str] = {
    "CM_MTD":      "-",
    "RRT_FRVM":    "--",
    "DQN_RM_FRVM": "-.",
    "STATIC":      ":",
    "HAM_ONLY":    "--",
    "RM_ONLY":     "-.",
}

MARKERS: Dict[str, str] = {
    "CM_MTD":      "o",
    "RRT_FRVM":    "s",
    "DQN_RM_FRVM": "^",
    "STATIC":      "D",
    "HAM_ONLY":    "v",
    "RM_ONLY":     "P",
}

METHOD_LABELS: Dict[str, str] = {
    "CM_MTD":      "CM-MTD (Proposed)",
    "RRT_FRVM":    "RRT+FRVM",
    "DQN_RM_FRVM": "DQ-RM+FRVM",
    "STATIC":      "Static",
    "HAM_ONLY":    "HAM Only",
    "RM_ONLY":     "RM Only",
}


def setup_style(font_size: int = 12) -> None:
    """
    Configure matplotlib for publication-quality output.
    Mimics IEEE JSAC paper style.
    """
    # Apply seaborn base style — handle API changes across versions
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "seaborn"):
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue

    plt.rcParams.update({
        # Font
        "font.family":       "DejaVu Serif",
        "font.size":         font_size,
        "axes.titlesize":    font_size + 1,
        "axes.labelsize":    font_size,
        "xtick.labelsize":   font_size - 1,
        "ytick.labelsize":   font_size - 1,
        "legend.fontsize":   font_size - 1,
        # Lines
        "lines.linewidth":   2.0,
        "lines.markersize":  5,
        # Grid
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        # Figure
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
        # Spines
        "axes.spines.top":   False,
        "axes.spines.right": False,
        # LaTeX-style math
        "mathtext.fontset":  "cm",
    })


def save_figure(
    fig: plt.Figure,
    name: str,
    out_dir: str = "results/figures",
    formats: Tuple[str, ...] = ("png", "pdf", "svg"),
    dpi: int = 300,
) -> Dict[str, str]:
    """
    Save a figure in multiple formats.

    Args:
        fig:     Matplotlib figure.
        name:    Base filename (no extension).
        out_dir: Output directory root.
        formats: Tuple of export formats.
        dpi:     Resolution for raster formats.

    Returns:
        Dict mapping format → full file path.
    """
    paths = {}
    for fmt in formats:
        fmt_dir = Path(out_dir) / fmt
        fmt_dir.mkdir(parents=True, exist_ok=True)
        fpath = fmt_dir / f"{name}.{fmt}"
        fig.savefig(str(fpath), dpi=dpi, bbox_inches="tight")
        paths[fmt] = str(fpath)
        logger.info(f"  Saved: {fpath}")
    return paths


def moving_average(data: np.ndarray, window: int = 100) -> np.ndarray:
    """Smooth a time series with a moving average."""
    if len(data) < window:
        return np.array(data, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(data, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def add_std_shadow(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    alpha: float = 0.15,
) -> None:
    """Draw a shaded ±1 std error band around a mean curve."""
    ax.fill_between(x, mean - std, mean + std, alpha=alpha, color=color)


def plot_method_curves(
    ax: plt.Axes,
    x: np.ndarray,
    data: Dict[str, np.ndarray],      # method → [n_seeds, T] or [T]
    std_data: Optional[Dict[str, np.ndarray]] = None,
    smooth_window: int = 100,
    methods_order: Optional[List[str]] = None,
    xlabel: str = "Episode",
    ylabel: str = "Value",
    title: str = "",
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot multiple method curves with error shadows on a given Axes.

    Args:
        ax:            Target axes.
        x:             X-axis values (episode numbers).
        data:          Dict: method_name → mean values [T].
        std_data:      Dict: method_name → std values [T] (optional).
        smooth_window: Moving average window.
        methods_order: Plot order (default: all keys).
        xlabel:        X-axis label.
        ylabel:        Y-axis label.
        title:         Axes title.
        ylim:          Y-axis limits.
    """
    methods = methods_order or list(data.keys())

    for method in methods:
        if method not in data:
            continue
        mean = moving_average(np.asarray(data[method], dtype=float), smooth_window)
        n = min(len(x), len(mean))
        xp = x[:n]
        mp = mean[:n]

        color = COLORS.get(method, "#333333")
        ls    = LINE_STYLES.get(method, "-")
        label = METHOD_LABELS.get(method, method)

        ax.plot(xp, mp, color=color, linestyle=ls, linewidth=2.0, label=label)

        if std_data and method in std_data:
            std = moving_average(np.asarray(std_data[method], dtype=float), smooth_window)
            sp = std[:n]
            add_std_shadow(ax, xp, mp, sp, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc="lower right", framealpha=0.7)


def make_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, Any]:
    """Create a figure with consistent sizing."""
    if figsize is None:
        w = 5.0 * ncols
        h = 4.0 * nrows
        figsize = (w, h)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, tight_layout=True)
    return fig, axes
