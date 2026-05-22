"""Dartboard geometry and scoring.

Coordinates are normalized to [-0.5, 0.5] on both axes; the board fits exactly
in that square. x+ points toward the 20 (top of the board), y+ points to the
left. This matches the convention used by the reference implementation.
"""

from __future__ import annotations

import math

import numpy as np

NUMBERS = np.array(
    [20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 6, 13, 4, 18, 1],
    dtype=np.int64,
)
ANGLE_STEP = 2 * math.pi / 20.0

# Official dimensions in mm (https://www.dimensions.com/element/dartboard).
BULL_EYE_DIAM = 12.7
BULL_GREEN_DIAM = 32.0
BORDER = 8.0
TRIPLE_EXT_DIAM = 214.0
TOTAL_DIAM = 340.0


def get_score(x, y):
    """Return the dart score at point(s) ``(x, y)`` in normalized board coords.

    ``x`` and ``y`` may be scalars or NumPy arrays of any matching shape.
    Returns a NumPy array (or 0-d array for scalar inputs) of float scores.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    x_mm = x_arr * TOTAL_DIAM
    y_mm = y_arr * TOTAL_DIAM
    distance = np.hypot(x_mm, y_mm)

    score = np.zeros_like(distance, dtype=np.float64)

    bull = distance <= BULL_EYE_DIAM / 2
    green = (distance <= BULL_GREEN_DIAM / 2) & ~bull
    interior = (distance > BULL_GREEN_DIAM / 2) & (distance <= TOTAL_DIAM / 2)

    score = np.where(bull, 50.0, score)
    score = np.where(green, 25.0, score)

    if interior.any():
        theta = np.arctan2(y_mm[interior], x_mm[interior])
        angle_index = (
            ((theta + ANGLE_STEP / 2) % (2 * np.pi)) / (2 * np.pi) * 20
        ).astype(np.int64)
        number = NUMBERS[angle_index]

        dist_i = distance[interior]
        is_double = dist_i >= TOTAL_DIAM / 2 - BORDER
        is_triple = (dist_i < TRIPLE_EXT_DIAM / 2) & (
            dist_i >= TRIPLE_EXT_DIAM / 2 - BORDER
        )
        multiplier = np.where(is_double, 2, np.where(is_triple, 3, 1))
        score[interior] = number * multiplier

    return score


def rasterize_score(resolution: int) -> np.ndarray:
    """Rasterize the score field on a square grid of side ``resolution``.

    The grid spans the normalized board square [-0.5, 0.5] x [-0.5, 0.5].
    Returns an array of shape ``(resolution, resolution)``; ``out[i, j]`` is
    the score at the point ``(x_i, y_j)`` (i indexes x, j indexes y).
    """
    coords = np.linspace(-0.5, 0.5, resolution)
    xs, ys = np.meshgrid(coords, coords, indexing="ij")
    return get_score(xs, ys)
