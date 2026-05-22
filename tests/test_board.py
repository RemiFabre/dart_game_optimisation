"""Parity test: new vectorized get_score must match the reference exactly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import aiming_spots  # noqa: E402  reference implementation

from darts import board  # noqa: E402


@pytest.mark.parametrize("resolution", [51, 200])
def test_get_score_matches_reference(resolution: int) -> None:
    coords = np.linspace(-0.5, 0.5, resolution)
    new_grid = board.rasterize_score(resolution)

    ref_grid = np.zeros((resolution, resolution), dtype=np.float64)
    for i, xi in enumerate(coords):
        for j, yj in enumerate(coords):
            ref_grid[i, j], _ = aiming_spots.get_score(float(xi), float(yj))

    np.testing.assert_array_equal(new_grid, ref_grid)


def test_get_score_scalar_returns_zero_d_array() -> None:
    out = board.get_score(0.0, 0.0)
    assert out.shape == ()
    assert float(out) == 50.0


def test_known_landmarks() -> None:
    # Bullseye at the centre.
    assert float(board.get_score(0.0, 0.0)) == 50.0
    # Outside the board.
    assert float(board.get_score(0.49, 0.49)) == 0.0
    # Triple 20 should be near (x = (TRIPLE_EXT_DIAM/2 - BORDER/2)/TOTAL_DIAM, y=0).
    t20_x = (board.TRIPLE_EXT_DIAM / 2 - board.BORDER / 2) / board.TOTAL_DIAM
    assert float(board.get_score(t20_x, 0.0)) == 60.0
