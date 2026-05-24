"""Three-panel comparison: optimal aim trajectory vs sigma, under three anisotropy regimes.

Each panel sweeps the geometric-mean precision σ_geom = sqrt(σ_x σ_y) from
3 mm to 50 mm. The three panels differ only in *shape*:

  - Panel A — isotropic:           σ_x = σ_y = σ_geom
  - Panel B — vertical-dominant:   σ_y = 2 σ_x  (so σ_x = σ_geom/√2, σ_y = √2 σ_geom)
  - Panel C — horizontal-dominant: σ_x = 2 σ_y  (so σ_x = √2 σ_geom, σ_y = σ_geom/√2)

For each (mode, σ_geom) pair we run the FFT EV solver, locate the global
optimum, and plot that point on a dartboard background. The colour of each
dot encodes σ_geom (small = bright, large = dark).

Output: blog/figures/fig_anisotropy_trails.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from darts import board  # noqa: E402
from darts.ev import expected_value_map  # noqa: E402
from darts.viz import draw_board  # noqa: E402


HERE = Path(__file__).resolve().parent.parent
OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sweep(sigma_geom_mm: np.ndarray, mode: str, resolution: int = 384):
    """Return arrays (best_x, best_y, best_ev) for each σ_geom under `mode`."""
    xs, ys, evs = [], [], []
    for sg_mm in sigma_geom_mm:
        sg = sg_mm / board.TOTAL_DIAM  # normalised
        if mode == "isotropic":
            sx, sy = sg, sg
        elif mode == "vertical":
            sx, sy = sg / np.sqrt(2), sg * np.sqrt(2)
        elif mode == "horizontal":
            sx, sy = sg * np.sqrt(2), sg / np.sqrt(2)
        else:
            raise ValueError(mode)
        result = expected_value_map(sx, sy, resolution=resolution)
        (bx, by), bev = result.argmax_board()
        xs.append(bx)
        ys.append(by)
        evs.append(bev)
    return np.asarray(xs), np.asarray(ys), np.asarray(evs)


def main():
    # Log-spaced σ from 3 to 50 mm — covers excellent to average player range.
    sigma_geom_mm = np.geomspace(3.0, 50.0, num=28)

    modes = [
        ("isotropic",   "Isotropic\nσ_x = σ_y",                 "viridis"),
        ("vertical",    "Vertical-dominant\nσ_y = 2 σ_x  (tall ellipse)",  "viridis"),
        ("horizontal",  "Horizontal-dominant\nσ_x = 2 σ_y  (wide ellipse)", "viridis"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5))
    for ax, (mode, title, cmap) in zip(axes, modes):
        xs, ys, evs = sweep(sigma_geom_mm, mode)
        ax.set_aspect("equal")
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.55, 0.55)
        ax.set_xticks([])
        ax.set_yticks([])
        draw_board(ax, with_numbers=True)
        # Plot the optimum trail.
        sc = ax.scatter(xs, ys, c=sigma_geom_mm, cmap=cmap, s=80,
                        edgecolors="white", linewidths=0.6, zorder=10)
        # Draw a small inset showing the error-ellipse shape (at σ_geom = 25 mm).
        sg_demo = 25.0 / board.TOTAL_DIAM
        if mode == "isotropic":
            sx, sy = sg_demo, sg_demo
        elif mode == "vertical":
            sx, sy = sg_demo / np.sqrt(2), sg_demo * np.sqrt(2)
        else:
            sx, sy = sg_demo * np.sqrt(2), sg_demo / np.sqrt(2)
        from matplotlib.patches import Ellipse
        # Position the inset ellipse in the upper-right corner.
        ell = Ellipse(
            (0.39, 0.39), width=2 * sx, height=2 * sy, fill=False,
            edgecolor="black", linewidth=2.0,
        )
        ax.add_patch(ell)
        ax.text(0.39, 0.49, "1σ ellipse\n(σ_geom = 25 mm)",
                ha="center", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=12)
        # Per-panel colourbar for σ_geom.
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("σ_geom (mm)", fontsize=10)

    fig.suptitle(
        "Same overall precision, three error shapes → very different optimal aim points\n"
        "(EV-maximising aim swept as σ_geom shrinks from 50 mm down to 3 mm)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_anisotropy_trails.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'fig_anisotropy_trails.png'}")


if __name__ == "__main__":
    main()
