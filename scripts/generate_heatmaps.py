"""Generate EV heatmaps for the four reference sigma configurations.

Outputs go under results/heatmaps/. These exist for visual + numerical
comparison against the reference Python implementation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402

from darts.ev import expected_value_map  # noqa: E402
from darts.viz import plot_ev_heatmap  # noqa: E402


SIGMAS = [
    ("perfect", 0.0, 0.0),
    ("world_champion", 0.015, 0.015),
    ("excellent", 0.02, 0.02),
    ("good", 0.07, 0.07),
    ("average", 0.15, 0.09),
    ("bad", 0.20, 0.20),
]

RESOLUTION = 1024


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "results" / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, sx, sy in SIGMAS:
        print(f"Computing EV map for {label} (sigma_x={sx}, sigma_y={sy})...")
        result = expected_value_map(sx, sy, resolution=RESOLUTION)
        (best_x, best_y), best_ev = result.argmax_board()
        ev_center = result.ev[result.resolution // 2, result.resolution // 2]
        print(
            f"  best aim=({best_x:+.3f}, {best_y:+.3f})  EV={best_ev:.3f}  "
            f"EV(center)={ev_center:.3f}"
        )

        fig = plot_ev_heatmap(
            result,
            title=f"{label} player  sigma=({sx}, {sy})",
        )
        fname = out_dir / f"ev_heatmap_{label}_sx{sx}_sy{sy}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")


if __name__ == "__main__":
    main()
