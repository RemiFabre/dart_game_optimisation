"""End-game visualizations: V(s) curves and outcome distributions.

Reads solver outputs from ``data/`` and writes plots to ``results/endgame/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from darts import board  # noqa: E402
from darts.io import load_npz  # noqa: E402
from darts.outcomes import IS_DOUBLE, OUTCOMES, VALUES  # noqa: E402
from darts.proba import probability_cube  # noqa: E402
from darts.viz import draw_board, plot_ev_heatmap  # noqa: E402
from darts.ev import EvResult  # noqa: E402


PLAYERS = [
    ("perfect", 0.0, 0.0, "tab:purple"),
    ("world_champion", 0.015, 0.015, "tab:cyan"),
    ("excellent", 0.02, 0.02, "tab:blue"),
    ("good", 0.07, 0.07, "tab:green"),
    ("average", 0.15, 0.09, "tab:orange"),
    ("bad", 0.20, 0.20, "tab:red"),
]


def plot_v_curves(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, sx, sy, color in PLAYERS:
        npz_path = data_dir / f"endgame_simple_{label}.npz"
        data, meta = load_npz(npz_path)
        v = data["v"]
        s = np.arange(len(v))
        ax.plot(s, v, label=f"{label} (σ={sx}, {sy})", color=color, linewidth=2)

    ax.set_xlabel("Remaining score")
    ax.set_ylabel("Expected throws to finish")
    ax.set_title("Single-dart end game (no double-out, 301-style)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "v_curves_simple.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {out_dir / 'v_curves_simple.png'}")


def plot_v_curve_official(data_dir: Path, out_dir: Path) -> None:
    """V(s) for the official 501 game (currently solved for 'good' only)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / "endgame_official_good.npz"
    if not npz_path.exists():
        return
    data, meta = load_npz(npz_path)
    v = data["v"]
    s = np.arange(len(v))
    finite = np.isfinite(v)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(s[finite], v[finite], color="tab:green", linewidth=2,
            label=f"good player σ=({meta['sigma_x']}, {meta['sigma_y']})")
    ax.set_xlabel("Remaining score")
    ax.set_ylabel("Expected throws to finish")
    ax.set_title("Official 501 end game (3-dart turns, double-out)")
    ax.axvline(1, color="black", linestyle=":", alpha=0.5, label="s=1: can't double-out")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "v_curve_official_good.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {out_dir / 'v_curve_official_good.png'}")


def plot_outcome_distribution(out_dir: Path) -> None:
    """For each player, plot the probability distribution of outcomes when
    aiming at their personal optimum (highest EV)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (label, sx, sy, color) in zip(axes.flatten(), PLAYERS):
        pc = probability_cube(sx, sy, resolution=512)
        # find the optimal-EV aim
        ev_map = np.einsum("kij,k->ij", pc.cube, VALUES.astype(pc.cube.dtype))
        idx = int(ev_map.argmax())
        i, j = divmod(idx, ev_map.shape[1])
        probs = pc.cube[:, i, j]
        # Sort by outcome value for nicer plot
        order = np.argsort([o.value for o in OUTCOMES])
        names = [OUTCOMES[k].name for k in order]
        vals = probs[order]
        colors = [
            "tab:purple" if IS_DOUBLE[k] else
            "tab:red" if OUTCOMES[k].name.startswith("T") else
            "tab:gray" if OUTCOMES[k].name == "MISS" else
            "tab:blue"
            for k in order
        ]
        ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_ylabel("P(outcome)")
        ax.set_title(f"{label} player (σ=({sx}, {sy}))  aiming at optimum  EV={ev_map[i,j]:.2f}")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Outcome distribution at optimal aim (blue=single, red=triple, purple=double, gray=miss)")
    fig.tight_layout()
    fig.savefig(out_dir / "outcome_histograms.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {out_dir / 'outcome_histograms.png'}")


def plot_optimum_trail(out_dir: Path) -> None:
    """Plot how the optimal aim point migrates as sigma changes (symmetric)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sigmas = np.linspace(0.01, 0.25, 30)
    aims = []
    evs = []
    for s in sigmas:
        from darts.ev import expected_value_map
        ev = expected_value_map(s, s, resolution=256)
        (bx, by), bev = ev.argmax_board()
        aims.append((bx, by))
        evs.append(bev)
    aims = np.array(aims)
    evs = np.array(evs)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.imshow(
        np.zeros((10, 10)),
        extent=(-0.5, 0.5, -0.5, 0.5),
        cmap="gray",
        vmin=0,
        vmax=1,
        alpha=0,
    )
    draw_board(ax, with_numbers=True)
    # Plot the trail.
    disp_h = -aims[:, 1]
    disp_v = aims[:, 0]
    sc = ax.scatter(disp_h, disp_v, c=sigmas, cmap="viridis", s=60, edgecolors="white", linewidths=0.5, zorder=10)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("σ (player precision; smaller = better)")
    ax.set_title("Migration of the optimal aim point as the player's σ varies")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    fig.tight_layout()
    fig.savefig(out_dir / "optimum_trail.png", dpi=120)
    plt.close(fig)
    print(f"  wrote {out_dir / 'optimum_trail.png'}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    out_dir = repo_root / "results" / "endgame"

    plot_v_curves(data_dir, out_dir)
    plot_v_curve_official(data_dir, out_dir)
    plot_outcome_distribution(out_dir)
    plot_optimum_trail(out_dir)


if __name__ == "__main__":
    main()
