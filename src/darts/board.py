"""Dartboard geometry and scoring.

Coordinates are normalized to [-0.5, 0.5] on both axes; the board fits exactly
in that square. We use the standard convention from the literature
(Tibshirani 2011; Haugh & Wang 2022, 2024):

- ``x`` is the **horizontal** axis. ``+x`` points to the right (toward the 6
  wedge at the 3 o'clock position).
- ``y`` is the **vertical** axis. ``+y`` points upward (toward the 20 wedge at
  the top of the board).
- The origin ``(0, 0)`` is the centre of the bullseye.
"""

from __future__ import annotations

import math

import numpy as np

# NUMBERS_CCW is the canonical wedge order, counterclockwise starting from
# the 20 at the top (this is the order published in every paper).
NUMBERS_CCW = np.array(
    [20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 6, 13, 4, 18, 1],
    dtype=np.int64,
)
# In the new (literature) convention, theta=0 is along +x — the 6 wedge —
# so we rotate the array so its first entry sits where ``angle_index == 0``.
# 6 is at index 15 of NUMBERS_CCW, so we roll by -15.
NUMBERS = np.roll(NUMBERS_CCW, -15)
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
        # theta=0 lies on +x (3 o'clock) which is the 6 wedge; the 20 wedge
        # is at theta = pi/2. NUMBERS has been pre-rolled so that index 0
        # corresponds to the wedge containing +x.
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
    the score at the point ``(x_i, y_j)`` (axis 0 = x = horizontal,
    axis 1 = y = vertical).
    """
    coords = np.linspace(-0.5, 0.5, resolution)
    xs, ys = np.meshgrid(coords, coords, indexing="ij")
    return get_score(xs, ys)


def get_outcome_index(x, y):
    """Return the outcome index (into ``darts.outcomes.OUTCOMES``) at each point.

    Same vectorization rules as :func:`get_score`. The mapping is::

        0                 = MISS (off board)
        1..20             = S1..S20  (single 1..20)
        21..40            = D1..D20  (double 1..20)
        41..60            = T1..T20  (triple 1..20)
        61                = BULL_25  (outer bull)
        62                = BULL_50  (bullseye, double-bull)
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    x_mm = x_arr * TOTAL_DIAM
    y_mm = y_arr * TOTAL_DIAM
    distance = np.hypot(x_mm, y_mm)

    out = np.zeros_like(distance, dtype=np.int64)  # default 0 = MISS

    bull = distance <= BULL_EYE_DIAM / 2
    green = (distance <= BULL_GREEN_DIAM / 2) & ~bull
    interior = (distance > BULL_GREEN_DIAM / 2) & (distance <= TOTAL_DIAM / 2)

    out = np.where(bull, 62, out)  # BULL_50
    out = np.where(green, 61, out)  # BULL_25

    if interior.any():
        # theta=0 lies on +x (3 o'clock) which is the 6 wedge; the 20 wedge
        # is at theta = pi/2. NUMBERS has been pre-rolled so that index 0
        # corresponds to the wedge containing +x.
        theta = np.arctan2(y_mm[interior], x_mm[interior])
        angle_index = (
            ((theta + ANGLE_STEP / 2) % (2 * np.pi)) / (2 * np.pi) * 20
        ).astype(np.int64)
        number = NUMBERS[angle_index]  # value 1..20 at this wedge

        dist_i = distance[interior]
        is_double = dist_i >= TOTAL_DIAM / 2 - BORDER
        is_triple = (dist_i < TRIPLE_EXT_DIAM / 2) & (
            dist_i >= TRIPLE_EXT_DIAM / 2 - BORDER
        )

        # Single: index = number (1..20). Double: 20 + number. Triple: 40 + number.
        idx = np.where(is_double, 20 + number, np.where(is_triple, 40 + number, number))
        out[interior] = idx

    return out


def rasterize_outcome_index(resolution: int) -> np.ndarray:
    """Rasterize the outcome-index field on a square grid of side ``resolution``."""
    coords = np.linspace(-0.5, 0.5, resolution)
    xs, ys = np.meshgrid(coords, coords, indexing="ij")
    return get_outcome_index(xs, ys)
