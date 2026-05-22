"""Visualization helpers for the dart solver.

Display convention matches the reference code: vertical axis = board x
(positive toward the 20, i.e. up); horizontal axis = -board y (positive y is
to the left). With this convention the dart cluster you'd visually expect
appears where you'd expect on a real dartboard.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from . import board
from .ev import EvResult


def _board_angle_to_display_xy(theta_b: float, r: float) -> tuple[float, float]:
    """Map a polar (theta_b, r) in board coords to display (horizontal, vertical)."""
    return -math.sin(theta_b) * r, math.cos(theta_b) * r


def draw_board(ax: Axes, *, with_numbers: bool = True) -> None:
    """Draw the dartboard outline (rings + wedge boundaries + numbers) on ``ax``.

    Uses the display convention documented at the module top.
    """
    # Diameters expressed in normalized board units (board fits in [-0.5, 0.5]).
    r_bull = board.BULL_EYE_DIAM / (2 * board.TOTAL_DIAM)
    r_green = board.BULL_GREEN_DIAM / (2 * board.TOTAL_DIAM)
    r_triple_outer = board.TRIPLE_EXT_DIAM / (2 * board.TOTAL_DIAM)
    r_triple_inner = (board.TRIPLE_EXT_DIAM / 2 - board.BORDER) / board.TOTAL_DIAM
    r_double_outer = 0.5
    r_double_inner = (board.TOTAL_DIAM / 2 - board.BORDER) / board.TOTAL_DIAM

    for r in (r_bull, r_green, r_triple_outer, r_triple_inner, r_double_outer, r_double_inner):
        ax.add_patch(
            plt.Circle(
                (0.0, 0.0), r, fill=False, linestyle="--", linewidth=1.3, color="red"
            )
        )

    # Wedge boundaries: 20 lines at angles angle_step/2 + i*angle_step (board coords).
    segments = []
    for i in range(20):
        theta_b = board.ANGLE_STEP / 2 + i * board.ANGLE_STEP
        inner = _board_angle_to_display_xy(theta_b, r_green)
        outer = _board_angle_to_display_xy(theta_b, r_double_outer)
        segments.append([inner, outer])
    ax.add_collection(LineCollection(segments, linewidths=1.0, colors="#1f77b4"))

    if with_numbers:
        r_label = (r_double_outer + 0.025)
        for i, n in enumerate(board.NUMBERS.tolist()):
            theta_b = i * board.ANGLE_STEP
            x_disp, y_disp = _board_angle_to_display_xy(theta_b, r_label)
            ax.text(
                x_disp, y_disp, str(n),
                ha="center", va="center", fontsize=10, color="black",
                fontweight="bold",
            )


def plot_ev_heatmap(
    result: EvResult,
    *,
    title: str | None = None,
    mark_optimum: bool = True,
    show_numbers: bool = True,
    figsize: tuple[float, float] = (8.0, 8.0),
    cmap: str = "inferno",
) -> plt.Figure:
    """Render the EV map as a heatmap using the standard board orientation."""
    ev = result.ev
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    # Flip the y (column) axis so that positive board y ends up on the left.
    im = ax.imshow(
        ev[:, ::-1],
        origin="lower",
        extent=(-0.5, 0.5, -0.5, 0.5),
        cmap=cmap,
        interpolation="nearest",
    )

    draw_board(ax, with_numbers=show_numbers)

    if mark_optimum:
        (best_x, best_y), best_ev = result.argmax_board()
        disp_h, disp_v = -best_y, best_x
        ax.plot(disp_h, disp_v, "o", color="white", markersize=12, markeredgecolor="black")
        ax.annotate(
            f"  best EV={best_ev:.2f}\n  aim=({best_x:.3f}, {best_y:.3f})",
            xy=(disp_h, disp_v),
            xytext=(disp_h + 0.05, disp_v + 0.05),
            fontsize=10,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2),
        )

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Expected score per dart", rotation=270, labelpad=15)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
