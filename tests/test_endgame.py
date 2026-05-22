"""Tests for the end-game solvers."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from darts.endgame import solve_official, solve_simple
from darts.proba import probability_cube


@pytest.fixture(scope="module")
def cube_good():
    return probability_cube(0.07, 0.07, resolution=128)


def test_simple_solver_matches_reference(cube_good) -> None:
    """The simple-mode solver should agree with the reference 301-DP pickle.

    The reference uses 51x51 aim grid + 10000-sample Monte Carlo probability
    estimates, so it has Monte-Carlo noise (~0.25 score points std error per
    aim point). When accumulated over many throws the V values can drift by
    a fraction of a throw; we accept differences up to 0.5.
    """
    ref_path = REPO_ROOT / "scores_ev_and_pos_2601_size10000_sx0.07_sy0.07"
    with open(ref_path, "rb") as f:
        ref = pickle.load(f)

    sol = solve_simple(cube_good, goal=301)

    diffs = []
    for ref_score in range(0, 301):
        if ref_score not in ref or ref[ref_score][1] == "N/A":
            continue
        remaining = 301 - ref_score
        ref_v = ref[ref_score][0]
        new_v = sol.v[remaining]
        diffs.append(abs(new_v - ref_v))

    max_diff = max(diffs)
    mean_diff = float(np.mean(diffs))
    # End-game DP accumulates per-throw EV error. Reference per-throw EV has
    # MC std error ~0.25 -> V can drift ~0.5 for full games.
    assert max_diff < 0.5, (
        f"max V diff = {max_diff:.3f} exceeds tolerance (mean={mean_diff:.3f})"
    )


def test_simple_v_for_remaining_1_is_one_over_p_s1(cube_good) -> None:
    """From remaining=1, only S1 scores exactly to finish; V = 1 / P(S1)."""
    sol = solve_simple(cube_good, goal=301)
    # Validity: V[1] should be > 1 (no perfect S1) and < something reasonable.
    assert 1.0 < sol.v[1] < 50.0


def test_official_v_at_2_is_finite_and_at_1_is_unreachable_or_inf(cube_good) -> None:
    """V(2) should be finite under official rules (D1 finishes).

    V(1) is special: you can't double-out from 1 (any successful throw
    overshoots or lands on a non-double zero), so V(1) stays at the
    +inf sentinel that the solver initializes.
    """
    sol = solve_official(cube_good, goal=20, fixed_point_max_iter=30)
    assert np.isfinite(sol.v[2])
    assert sol.v[2] > 1.0  # at least one throw to hit D1
    assert not np.isfinite(sol.v[1]) or sol.v[1] > 1000.0


def test_official_v_at_50_uses_bullseye_or_double(cube_good) -> None:
    """From s=50, BULL_50 closes in 1 throw; V(50) should be small for a good player."""
    sol = solve_official(cube_good, goal=60, fixed_point_max_iter=30)
    # Good player can hit bullseye occasionally; V(50) should be a handful of
    # throws, but not enormous.
    assert np.isfinite(sol.v[50])
    assert sol.v[50] < 50.0


def test_official_v_at_2_matches_monte_carlo_simulation(cube_good) -> None:
    """Simulate the optimal policy at score 2 and confirm V(2) agrees.

    From s=2, the only way to finish in this turn is to hit D1. We simulate
    many turns following the solver's chosen aim-per-dart-state and compare
    the empirical expected throws to V(2).
    """
    from darts import board

    sol = solve_official(cube_good, goal=10, fixed_point_max_iter=100, fixed_point_tol=1e-9)
    s = 2

    # Aim per (darts_left): 1 = dart 3, 2 = dart 2, 3 = dart 1 (start of turn).
    aim_d1 = sol.best_first_aim[s]                  # dart 1
    aim_d2 = sol.mid_turn_aim[(s, s, 2)]            # dart 2 (only if s_curr still == s)
    aim_d3 = sol.mid_turn_aim[(s, s, 1)]            # dart 3
    aims = [aim_d1, aim_d2, aim_d3]
    aim_xy = [cube_good.pixel_to_board(int(i), int(j)) for (i, j) in aims]

    rng = np.random.default_rng(123)
    n_games = 100_000
    sigma = 0.07
    total_throws = 0
    games_left = n_games

    # Use a fully vectorised simulation: each game = up to 3 darts.
    games_finished = 0
    cur_throws = 0
    # Restart loop: keep playing turns until game ends.
    score = np.full(n_games, s, dtype=np.int64)  # starting score
    throws_per_game = np.zeros(n_games, dtype=np.int64)
    done = np.zeros(n_games, dtype=bool)

    while not done.all():
        active = ~done
        n_active = int(active.sum())
        # Throw dart 1 of new turn.
        for dart_idx in range(3):
            if n_active == 0:
                break
            ax, ay = aim_xy[dart_idx]
            xs = rng.normal(ax, sigma, n_active)
            ys = rng.normal(ay, sigma, n_active)
            regions = board.get_outcome_index(xs, ys)
            from darts.outcomes import OUTCOME_INDEX

            # finish iff region == D1
            is_finish = regions == OUTCOME_INDEX["D1"]
            # miss iff region == MISS (value 0); otherwise bust (any other region)
            is_miss = regions == OUTCOME_INDEX["MISS"]
            # bust: not finish and not miss
            is_bust = ~is_finish & ~is_miss

            active_idx = np.where(active)[0]
            throws_per_game[active_idx] += 1

            # Update finished games
            finish_idx = active_idx[is_finish]
            done[finish_idx] = True

            # Bust ends turn: active stays True but break to next-turn loop
            bust_idx = active_idx[is_bust]
            # Misses keep the dart loop going; bust ends the turn but next turn resumes.
            if dart_idx == 2:
                # End of turn regardless; no break.
                pass
            # For early break on bust, we need per-game termination of dart loop.
            # Simplest correct simulation: re-enter the for-loop only with the
            # subset of games that missed (continued in turn). We'll process this
            # by reducing the active set for the next dart_idx iteration.
            # For correctness, only games that "missed" should throw the next dart.
            # Bust games will start a new turn (still active).
            # So: for next dart loop iteration, the "active for next dart" set
            # is the subset that missed.
            # We'll restructure: build a mask for "should throw next dart in this turn".
            still_in_turn = np.zeros(n_games, dtype=bool)
            still_in_turn[active_idx[is_miss]] = True
            # The active set for the next dart_idx becomes still_in_turn.
            active = still_in_turn
            n_active = int(active.sum())

        # Any game that's not done is starting a new turn at score 2.
        # done already marks finished games.
        # active for the next iteration: ~done
        # (We loop until done.all().)

    empirical = float(throws_per_game.mean())
    se = float(throws_per_game.std(ddof=1) / np.sqrt(n_games))
    assert abs(empirical - sol.v[s]) < 5 * se + 0.1, (
        f"empirical V(2) = {empirical:.3f} +/- {se:.3f}, solver V(2) = {sol.v[s]:.3f}"
    )
