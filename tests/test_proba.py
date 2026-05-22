"""Sanity tests for the probability cube."""

from __future__ import annotations

import numpy as np
import pytest

from darts import board
from darts.ev import expected_value_map
from darts.outcomes import IS_DOUBLE, N_OUTCOMES, OUTCOMES, VALUES
from darts.proba import (
    double_finishing_prob,
    ev_from_cube,
    outcome_masks,
    probability_cube,
)


def test_outcome_table_is_consistent() -> None:
    # Exactly 1 + 20 + 20 + 20 + 2 = 63 outcomes.
    assert N_OUTCOMES == 63
    # The 21st through 40th outcomes are the doubles.
    assert all(o.is_double for o in OUTCOMES[21:41])
    # Bullseye is the last entry and counts as a double.
    assert OUTCOMES[-1].name == "BULL_50" and OUTCOMES[-1].is_double
    # Every value is reproducible from the value table.
    assert all(VALUES[i] == o.value for i, o in enumerate(OUTCOMES))


def test_outcome_index_matches_score() -> None:
    """The score of get_outcome_index(p) must match get_score(p)."""
    coords = np.linspace(-0.49, 0.49, 50)
    xs, ys = np.meshgrid(coords, coords, indexing="ij")
    idx = board.get_outcome_index(xs, ys)
    score_direct = board.get_score(xs, ys)
    score_via_idx = VALUES[idx]
    np.testing.assert_array_equal(score_via_idx, score_direct)


def test_outcome_masks_are_a_partition() -> None:
    """Masks should be 0/1, disjoint, and cover every pixel exactly once."""
    masks = outcome_masks(resolution=64)
    # Binary.
    assert np.all((masks == 0) | (masks == 1))
    # Exactly one mask is "on" per pixel.
    total = masks.sum(axis=0)
    assert np.all(total == 1)


@pytest.mark.parametrize("sigma_x,sigma_y", [(0.02, 0.02), (0.07, 0.07), (0.15, 0.09)])
def test_probability_cube_sums_to_one(sigma_x: float, sigma_y: float) -> None:
    pc = probability_cube(sigma_x, sigma_y, resolution=128)
    totals = pc.cube.sum(axis=0)
    # Convolution preserves the total mass (the masks summed to 1 everywhere,
    # so after convolution with a normalized kernel the per-pixel sum is 1).
    np.testing.assert_allclose(totals, 1.0, atol=1e-4)


@pytest.mark.parametrize("sigma_x,sigma_y", [(0.02, 0.02), (0.07, 0.07), (0.15, 0.09)])
def test_cube_reproduces_ev_map(sigma_x: float, sigma_y: float) -> None:
    """Σ_k value[k] * P[k, i, j] must equal EV[i, j] from the direct map."""
    pc = probability_cube(sigma_x, sigma_y, resolution=256)
    ev_cube = ev_from_cube(pc)
    ev_direct = expected_value_map(sigma_x, sigma_y, resolution=256).ev
    np.testing.assert_allclose(ev_cube, ev_direct, atol=2e-3)


def test_double_finishing_prob_contains_all_doubles() -> None:
    pc = probability_cube(0.07, 0.07, resolution=64)
    finishing = double_finishing_prob(pc)
    # 20 doubles (2..40 even) + bullseye (50) = 21 finishing values.
    assert len(finishing) == 21
    for v in [2, 4, 6, 38, 40, 50]:
        assert v in finishing
    # Total double-mass at any aim point ≤ 1 (since they're a subset of outcomes).
    total_double = sum(finishing.values())
    assert total_double.max() <= 1.0 + 1e-5
