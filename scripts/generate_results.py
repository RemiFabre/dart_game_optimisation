"""Generate the full result bundle for the four reference sigma configurations.

Outputs are written under ``data/`` as ``.npy``/``.npz`` arrays paired with
``.json`` sidecars describing their contents. These are intended as the
canonical solver outputs that the visualization layer consumes.
"""

from __future__ import annotations

import time
from pathlib import Path

from darts.endgame import solve_official, solve_simple
from darts.ev import expected_value_map
from darts.io import (
    save_ev_map,
    save_official_solution,
    save_probability_cube,
    save_simple_solution,
)
from darts.proba import probability_cube


# Player tier definitions. sigma values are in normalised board units; multiply
# by board.TOTAL_DIAM = 340 mm to get millimetres. The labels and values are
# anchored on published measurements (see ANALYSIS.md §10.7):
#   - perfect:        sigma = 0   (mathematical ideal, the upper bound)
#   - world_champion: ~5 mm        (Tibshirani's "perfect" reference; peak PDC pros at T20)
#   - excellent:      ~6.8 mm      (strong pro / club champion)
#   - good:           ~23.8 mm     (Tibshirani's skilled amateur)
#   - average:        30 x 51 mm   (anisotropic; the typical biomechanical
#                                   pattern has vertical scatter > horizontal)
#   - bad:            ~68 mm       (Tibshirani's beginner)
SIGMAS = [
    ("perfect",        0.0,    0.0),
    ("world_champion", 0.015,  0.015),
    ("excellent",      0.02,   0.02),
    ("good",           0.07,   0.07),
    ("average",        0.09,   0.15),
    ("bad",            0.20,   0.20),
]

EV_RESOLUTION = 1024     # sub-mm aim grid for the heatmap stuff
DP_RESOLUTION = 128       # used for end-game DP — keeps the matmul cheap
SOLVE_OFFICIAL_FOR = {"good", "perfect", "world_champion"}  # interesting comparison set


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "data"
    out_dir.mkdir(exist_ok=True)

    for label, sx, sy in SIGMAS:
        print(f"\n=== {label} (sigma_x={sx}, sigma_y={sy}) ===")

        t = time.time()
        ev = expected_value_map(sx, sy, resolution=EV_RESOLUTION)
        save_ev_map(ev, out_dir, name=f"ev_{label}")
        (bx, by), best_ev = ev.argmax_board()
        print(
            f"  EV map ({EV_RESOLUTION}x{EV_RESOLUTION}) in {time.time()-t:.1f}s; "
            f"best aim=({bx:+.3f}, {by:+.3f})  EV={best_ev:.3f}"
        )

        t = time.time()
        pc = probability_cube(sx, sy, resolution=DP_RESOLUTION)
        save_probability_cube(pc, out_dir, name=f"cube_{label}")
        print(f"  Probability cube in {time.time()-t:.1f}s")

        t = time.time()
        simple_sol = solve_simple(pc, goal=301)
        save_simple_solution(simple_sol, out_dir, name=f"endgame_simple_{label}")
        print(
            f"  Simple end-game (301) in {time.time()-t:.1f}s; "
            f"V[301]={simple_sol.v[301]:.3f}"
        )

        if label in SOLVE_OFFICIAL_FOR:
            t = time.time()
            official_sol = solve_official(pc, goal=501, fixed_point_max_iter=30)
            save_official_solution(official_sol, out_dir, name=f"endgame_official_{label}")
            print(
                f"  Official end-game (501) in {time.time()-t:.1f}s; "
                f"V[501]={official_sol.v[501]:.3f}"
            )

    print("\nAll results written to", out_dir)


if __name__ == "__main__":
    main()
