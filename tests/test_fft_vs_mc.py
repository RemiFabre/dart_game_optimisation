"""Verification gate: FFT-based EV must agree with Monte Carlo simulation.

This is the test that lets us trust the FFT solver. For a handful of aim
points and sigma values, we draw a large Monte Carlo sample, compute the
empirical mean score and its standard error, and check that the FFT EV
falls within a few MC standard errors of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from darts import board
from darts.ev import expected_value_at, expected_value_map


def _monte_carlo_ev(
    aim_x: float,
    aim_y: float,
    sigma_x: float,
    sigma_y: float,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return (mean_score, std_error_of_mean) from independent samples."""
    xs = rng.normal(aim_x, sigma_x, n_samples)
    ys = rng.normal(aim_y, sigma_y, n_samples)
    scores = board.get_score(xs, ys)
    mean = float(scores.mean())
    se = float(scores.std(ddof=1) / np.sqrt(n_samples))
    return mean, se


T20_X = (board.TRIPLE_EXT_DIAM / 2 - board.BORDER / 2) / board.TOTAL_DIAM


@pytest.mark.parametrize(
    "aim_x,aim_y,sigma_x,sigma_y",
    [
        # Center, several player levels.
        (0.0, 0.0, 0.02, 0.02),
        (0.0, 0.0, 0.07, 0.07),
        (0.0, 0.0, 0.20, 0.20),
        # Triple-20 aim for an excellent player.
        (T20_X, 0.0, 0.02, 0.02),
        # The "good player" optimum from the reference README.
        (-0.3, 0.12, 0.07, 0.07),
        # Asymmetric sigma.
        (0.0, 0.0, 0.15, 0.09),
    ],
)
def test_fft_ev_agrees_with_monte_carlo(
    aim_x: float, aim_y: float, sigma_x: float, sigma_y: float
) -> None:
    rng = np.random.default_rng(seed=42)
    n_samples = 1_000_000
    mc_mean, mc_se = _monte_carlo_ev(aim_x, aim_y, sigma_x, sigma_y, n_samples, rng)
    fft_ev = expected_value_at(aim_x, aim_y, sigma_x, sigma_y, resolution=1024)

    # 5 MC standard errors + small allowance for FFT discretization.
    tolerance = 5 * mc_se + 0.15
    assert abs(fft_ev - mc_mean) < tolerance, (
        f"FFT={fft_ev:.4f} vs MC={mc_mean:.4f} +/- {mc_se:.4f}, "
        f"diff={fft_ev - mc_mean:.4f}, tol={tolerance:.4f}"
    )


def test_fft_optimum_for_good_player_matches_reference() -> None:
    """The reference README reports best aim (-0.3, 0.12) for sigma=0.07.

    The FFT optimum should land within a couple of pixels of that, and
    its EV should be within a fraction of a point of the published 16.4.
    """
    result = expected_value_map(0.07, 0.07, resolution=512)
    (best_x, best_y), best_ev = result.argmax_board()
    assert abs(best_x - (-0.3)) < 0.05
    assert abs(best_y - 0.12) < 0.05
    assert abs(best_ev - 16.4) < 0.5
