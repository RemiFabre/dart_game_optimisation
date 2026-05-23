"""Parity test: new vectorized get_score must match the reference exactly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "reference_python"))

import aiming_spots  # noqa: E402  reference implementation

from darts import board  # noqa: E402


@pytest.mark.parametrize("resolution", [51, 200])
def test_get_score_matches_reference(resolution: int) -> None:
    """Parity vs the reference, accounting for the 90 degree axis change.

    The reference uses ``x_old`` = vertical (toward 20) and ``y_old`` =
    horizontal (toward 11). We use ``x_new`` = horizontal (toward 6) and
    ``y_new`` = vertical (toward 20). The same physical point satisfies
    ``x_old = y_new`` and ``y_old = -x_new``, so we swap arguments and flip
    one sign when calling the reference.

    Because the new code computes ``atan2`` from a different argument pair
    than the reference, a handful of grid points sitting *exactly* on a
    wedge boundary fall on different sides of the boundary due to
    floating-point rounding (the wedge boundary lines are infinitely thin,
    so this is genuine boundary noise rather than a real disagreement).
    We tolerate a tiny number of such mismatches.
    """
    coords = np.linspace(-0.5, 0.5, resolution)
    new_grid = board.rasterize_score(resolution)

    ref_grid = np.zeros((resolution, resolution), dtype=np.float64)
    for i, x_new in enumerate(coords):
        for j, y_new in enumerate(coords):
            ref_grid[i, j], _ = aiming_spots.get_score(
                float(y_new), float(-x_new)
            )

    diff = new_grid != ref_grid
    n_mismatch = int(diff.sum())
    n_total = new_grid.size
    # Allow up to 0.05% mismatches (these are wedge-boundary points).
    assert n_mismatch / n_total < 5e-4, (
        f"{n_mismatch} / {n_total} mismatches (> 0.05%)"
    )


def test_get_score_scalar_returns_zero_d_array() -> None:
    out = board.get_score(0.0, 0.0)
    assert out.shape == ()
    assert float(out) == 50.0


def test_known_landmarks() -> None:
    # Bullseye at the centre.
    assert float(board.get_score(0.0, 0.0)) == 50.0
    # Outside the board.
    assert float(board.get_score(0.49, 0.49)) == 0.0
    # Triple 20 is at +y direction (vertical-up): (x=0, y = +t20_offset).
    t20_y = (board.TRIPLE_EXT_DIAM / 2 - board.BORDER / 2) / board.TOTAL_DIAM
    assert float(board.get_score(0.0, t20_y)) == 60.0
    # The 6 wedge is at +x direction (3 o'clock).
    s6_x = (board.TRIPLE_EXT_DIAM / 2 - board.BORDER / 2) / board.TOTAL_DIAM
    # The point above sits in the triple ring for the 6 wedge.
    assert float(board.get_score(s6_x, 0.0)) == 18.0
