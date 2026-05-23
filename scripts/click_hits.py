"""Interactive: load a dartboard photo, click landmarks + hits, fit a Gaussian.

Usage:

    python scripts/click_hits.py path/to/photo.jpg

Workflow:
    1. Two calibration clicks: first the board centre, then any point on the
       outer-bull ring (the red/green ring around the bullseye, radius
       16 mm from centre). This gives the pixel-to-mm scale.
    2. One click to mark the aim point you were using.
    3. Click each dart hit. Press 'u' to undo the last click. Press 'f' to
       finish — the script computes a Gaussian fit and prints sigma_x,
       sigma_y, rho, plus a copy-paste-ready snippet for the solver.

Coordinates are reported in normalized board units (x in [-0.5, 0.5],
with +x toward the 6 wedge — i.e. horizontal-right — and +y toward the
20 wedge — i.e. vertical-up). This matches the convention used in the
literature (Tibshirani 2011; Haugh & Wang 2022, 2024).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.patches import Circle, Ellipse

from darts import board
from darts.covariance import fit_gaussian


# Outer bull radius in mm (calibration ring).
CAL_RING_MM = board.BULL_GREEN_DIAM / 2  # = 16 mm


@dataclass
class State:
    stage: str = "calibrate_centre"
    centre_px: tuple[float, float] | None = None
    cal_ring_px: tuple[float, float] | None = None
    px_per_mm: float | None = None
    aim_px: tuple[float, float] | None = None
    hits_px: list[tuple[float, float]] = None  # type: ignore[assignment]


def _px_to_board(px: tuple[float, float], state: State) -> tuple[float, float]:
    """Convert pixel coordinates to normalized board coords.

    Pixel x increases to the right (same as board x). Pixel y increases
    downward but board y increases upward, so the y axis is flipped.
    """
    assert state.centre_px is not None and state.px_per_mm is not None
    dx_px = px[0] - state.centre_px[0]
    dy_px = px[1] - state.centre_px[1]
    # board_x = +dx_px (right in image = +x = toward 6).
    # board_y = -dy_px (up in image = +y = toward 20).
    x_mm = dx_px / state.px_per_mm
    y_mm = -dy_px / state.px_per_mm
    return x_mm / board.TOTAL_DIAM, y_mm / board.TOTAL_DIAM


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    img_path = Path(sys.argv[1])
    img = plt.imread(img_path)

    state = State(hits_px=[])
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.set_axis_off()
    msg = ax.set_title("Click the BOARD CENTRE")

    artists: list = []

    def redraw_title() -> None:
        if state.stage == "calibrate_centre":
            msg.set_text("Click the BOARD CENTRE")
        elif state.stage == "calibrate_ring":
            msg.set_text("Click any point on the OUTER BULL ring (radius 16 mm)")
        elif state.stage == "aim":
            msg.set_text("Click where you were AIMING")
        elif state.stage == "hits":
            msg.set_text(
                f"Click each HIT (collected: {len(state.hits_px)}). "
                "'u' undo, 'f' finish, 'q' quit."
            )
        elif state.stage == "done":
            msg.set_text("Done — see terminal output.")
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        px = (float(event.xdata), float(event.ydata))
        if state.stage == "calibrate_centre":
            state.centre_px = px
            artists.append(ax.scatter(*px, marker="+", c="yellow", s=200))
            state.stage = "calibrate_ring"
        elif state.stage == "calibrate_ring":
            state.cal_ring_px = px
            artists.append(ax.scatter(*px, marker="+", c="orange", s=200))
            assert state.centre_px is not None
            dist_px = float(np.hypot(px[0] - state.centre_px[0], px[1] - state.centre_px[1]))
            state.px_per_mm = dist_px / CAL_RING_MM
            artists.append(
                ax.add_patch(
                    Circle(state.centre_px, dist_px, fill=False, edgecolor="orange",
                           linestyle="--", linewidth=1.5)
                )
            )
            print(f"Calibration: {state.px_per_mm:.2f} px / mm")
            state.stage = "aim"
        elif state.stage == "aim":
            state.aim_px = px
            artists.append(ax.scatter(*px, marker="x", c="cyan", s=200))
            state.stage = "hits"
        elif state.stage == "hits":
            state.hits_px.append(px)
            artists.append(ax.scatter(*px, marker="o", c="red", s=40, edgecolors="white"))
        redraw_title()

    def on_key(event) -> None:
        if event.key == "u" and state.stage == "hits" and state.hits_px:
            state.hits_px.pop()
            # Remove the most recent hit marker.
            for a in reversed(artists):
                if hasattr(a, "get_facecolors"):
                    a.remove()
                    artists.remove(a)
                    break
            redraw_title()
        elif event.key == "f" and state.stage == "hits":
            finish()
        elif event.key == "q":
            plt.close(fig)

    def finish() -> None:
        if len(state.hits_px) < 3:
            print("Need at least 3 hits to fit a covariance.")
            return
        # Convert all to board coords.
        hits_board = np.array(
            [_px_to_board(p, state) for p in state.hits_px], dtype=np.float64
        )
        aim_board = np.array(_px_to_board(state.aim_px, state), dtype=np.float64)
        fit = fit_gaussian(hits_board, aim=aim_board)
        print("\n=== Gaussian fit ===")
        print(f"  N hits         = {len(state.hits_px)}")
        print(f"  aim (board)    = ({aim_board[0]:+.4f}, {aim_board[1]:+.4f})")
        print(f"  mean (board)   = ({fit.mean[0]:+.4f}, {fit.mean[1]:+.4f})")
        print(f"  sigma_x        = {fit.sigma_x:.4f}")
        print(f"  sigma_y        = {fit.sigma_y:.4f}")
        print(f"  correlation    = {fit.correlation:+.3f}")
        print(f"  cov            = {fit.cov.tolist()}")
        print("\nFor the solver:")
        print(f"  probability_cube({fit.sigma_x:.4f}, {fit.sigma_y:.4f}, resolution=256)")

        # Overlay ellipse on the image.
        eigvals, eigvecs = np.linalg.eigh(fit.cov)
        # Convert (board) covariance back to pixels for display.
        # 1 board unit = TOTAL_DIAM mm = TOTAL_DIAM * px_per_mm px.
        scale = board.TOTAL_DIAM * (state.px_per_mm or 1.0)
        # Board (x, y) -> image (px_x, px_y) = (x*scale, -y*scale) + centre_px.
        for n_sigma in (1.0, 2.0):
            w = 2 * n_sigma * float(np.sqrt(eigvals[0])) * scale
            h = 2 * n_sigma * float(np.sqrt(eigvals[1])) * scale
            vx, vy = eigvecs[:, 0]
            # Board direction (vx, vy) -> image direction (vx, -vy).
            angle_image = float(np.degrees(np.arctan2(-vy, vx)))
            centre_board = fit.mean
            cx_img = state.centre_px[0] + centre_board[0] * scale
            cy_img = state.centre_px[1] - centre_board[1] * scale
            artists.append(
                ax.add_patch(
                    Ellipse(
                        (cx_img, cy_img), w, h, angle=angle_image,
                        fill=False, edgecolor="lime", linewidth=2, alpha=0.8,
                    )
                )
            )
        state.stage = "done"
        redraw_title()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw_title()
    plt.show()


if __name__ == "__main__":
    main()
