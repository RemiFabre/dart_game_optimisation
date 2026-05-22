"""End-game dynamic programming.

Two modes are exposed:

- :func:`solve_simple` matches the reference implementation's semantics:
  one-throw-at-a-time, no double-out, missing the board counts as a bust
  (i.e. stays at current score, throw lost). This exists so we can verify
  against the existing pickled results.

- :func:`solve_official` implements proper 501 rules: 3-dart turns,
  double-out, busting (overshoot, leave score == 1, or land on 0 without
  hitting a double) ends the turn and rewinds the score to its value at the
  start of the turn.

Both consume a precomputed :class:`darts.proba.ProbabilityCube` and produce
``V[s]`` (expected throws to finish from score ``s``) and the optimal aim
pixel for each state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .outcomes import IS_DOUBLE, N_OUTCOMES, VALUES
from .proba import ProbabilityCube


@dataclass
class SimpleSolution:
    """Result of the simple-mode solver."""

    goal: int
    sigma_x: float
    sigma_y: float
    resolution: int
    v: np.ndarray                 # shape (goal+1,), v[s] = expected throws
    best_aim: np.ndarray          # shape (goal+1, 2), pixel (i, j)


@dataclass
class OfficialSolution:
    """Result of the official-mode solver.

    Within a turn, the optimal aim depends on three things: the score at the
    start of the turn (for bust rollback), the current remaining score, and
    the number of darts still to throw. We store aim per ``(s_turn, s_curr,
    darts_left)`` only for darts_left in {1, 2}; the dart-3 aim depends on
    the same state. For dart 1 of a turn (darts_left == 3) the turn-start
    score equals the current score, so we store it as a 1D table.
    """

    goal: int
    sigma_x: float
    sigma_y: float
    resolution: int
    v: np.ndarray                                              # shape (goal+1,)
    best_first_aim: np.ndarray                                 # (goal+1, 2)
    # mid_turn_aim[(s_turn, s_curr, darts_left)] -> (i, j)
    mid_turn_aim: dict[tuple[int, int, int], tuple[int, int]]


# ---------------------------------------------------------------------------
# Simple mode (reference semantics)
# ---------------------------------------------------------------------------


def solve_simple(pc: ProbabilityCube, goal: int = 301) -> SimpleSolution:
    """Solve the simple-mode end game from score ``goal`` down to 0.

    ``V[s]`` is the expected number of single-dart throws needed to bring
    the running score from ``s`` to 0, treating misses (value-0 throws) and
    overshoots (value > s) as busts (score unchanged, throw lost).
    """
    cube = pc.cube  # (n_outcomes, N, N)
    n = pc.resolution
    flat_cube = cube.reshape(N_OUTCOMES, -1)  # (n_outcomes, N*N)

    v = np.zeros(goal + 1, dtype=np.float64)
    best_aim = np.zeros((goal + 1, 2), dtype=np.int64)

    for s in range(1, goal + 1):
        new_scores = s - VALUES  # (n_outcomes,)
        finish_mask = new_scores == 0
        bust_mask = (new_scores < 0) | (VALUES == 0)
        continue_mask = ~finish_mask & ~bust_mask

        v_continue = np.zeros(N_OUTCOMES, dtype=np.float64)
        v_continue[continue_mask] = v[new_scores[continue_mask]]

        # Per-aim-pixel: bust probability and continue-cost expectation.
        bust_prob = flat_cube.T @ bust_mask.astype(flat_cube.dtype)         # (N*N,)
        continue_term = flat_cube.T @ v_continue.astype(flat_cube.dtype)     # (N*N,)

        # V_p = (1 + continue_term) / (1 - bust_prob), with safe handling
        # of degenerate aim points where everything busts.
        with np.errstate(divide="ignore", invalid="ignore"):
            v_cand = (1.0 + continue_term) / (1.0 - bust_prob)
        v_cand = np.where(np.isfinite(v_cand) & (bust_prob < 1.0), v_cand, np.inf)

        idx = int(np.argmin(v_cand))
        v[s] = float(v_cand[idx])
        best_aim[s] = (idx // n, idx % n)

    return SimpleSolution(
        goal=goal,
        sigma_x=pc.sigma_x,
        sigma_y=pc.sigma_y,
        resolution=pc.resolution,
        v=v,
        best_aim=best_aim,
    )


# ---------------------------------------------------------------------------
# Official mode: 3-dart turns + double-out
# ---------------------------------------------------------------------------


def _classify(s_curr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (finish_mask, bust_mask, continue_mask) for outcomes at score ``s_curr``."""
    new_scores = s_curr - VALUES
    finish_mask = (new_scores == 0) & IS_DOUBLE
    bust_mask = (
        (new_scores < 0)
        | (new_scores == 1)
        | ((new_scores == 0) & ~IS_DOUBLE)
    )
    continue_mask = ~finish_mask & ~bust_mask
    return finish_mask, bust_mask, continue_mask


def solve_official(
    pc: ProbabilityCube,
    goal: int = 501,
    fixed_point_tol: float = 1e-7,
    fixed_point_max_iter: int = 50,
) -> OfficialSolution:
    """Solve official 501 (or any goal) with double-out and 3-dart turns."""
    cube = pc.cube
    n = pc.resolution
    flat_cube = cube.reshape(N_OUTCOMES, -1)
    n_pixels = flat_cube.shape[1]

    v = np.full(goal + 1, np.inf, dtype=np.float64)
    v[0] = 0.0
    # V(1) intentionally stays +inf (no double-out from 1).
    best_first_aim = np.zeros((goal + 1, 2), dtype=np.int64)
    mid_turn_aim: dict[tuple[int, int, int], tuple[int, int]] = {}

    values_row = VALUES[None, :].astype(np.int64)
    is_double_row = IS_DOUBLE[None, :]

    for s in range(2, goal + 1):
        v_self = 3.0  # initial guess

        # All s_curr from 2 to s in one batch.
        scurr_arr = np.arange(2, s + 1, dtype=np.int64)[:, None]   # (n_scurr, 1)
        new_scores = scurr_arr - values_row                         # (n_scurr, K)
        finish_mat = (new_scores == 0) & is_double_row
        bust_mat = (
            (new_scores < 0)
            | (new_scores == 1)
            | ((new_scores == 0) & ~is_double_row)
        )
        continue_mat = ~finish_mat & ~bust_mat
        cont_to_s = continue_mat & (new_scores == s)
        cont_normal = continue_mat & ~cont_to_s

        ns_clipped = np.clip(new_scores, 0, s)                      # safe lookup
        static_dart3 = np.where(cont_normal, v[ns_clipped], 0.0)
        mask_dart3 = bust_mat | cont_to_s                           # contributes v_self

        # Row for V(s) = W_0(s); same logic as the last s_curr row above,
        # so we can index directly into the matrices.
        s_idx = s - 2  # row index of s_curr == s in the batched arrays

        idx_first = 0
        aim_w2_row = np.zeros(s - 1, dtype=np.int64)
        aim_w1_row = np.zeros(s - 1, dtype=np.int64)
        for iter_idx in range(fixed_point_max_iter):
            # W_2: cost_mat[s_curr, k] = v[ns] (continue, ns<s) or v_self (bust or ns==s)
            cost_mat_w2 = static_dart3 + mask_dart3 * v_self        # (n_scurr, K)
            cost_fields_w2 = 1.0 + cost_mat_w2 @ flat_cube           # (n_scurr, N*N)
            w2_vec = cost_fields_w2.min(axis=1)
            aim_w2_row = cost_fields_w2.argmin(axis=1)

            # W_1: continue advances to W_2(new_score)
            w2_full = np.zeros(s + 1, dtype=np.float64)
            w2_full[2:] = w2_vec
            cost_mat_w1 = np.where(continue_mat, w2_full[ns_clipped], 0.0)
            cost_mat_w1 = np.where(bust_mat, v_self, cost_mat_w1)
            cost_fields_w1 = 1.0 + cost_mat_w1 @ flat_cube
            w1_vec = cost_fields_w1.min(axis=1)
            aim_w1_row = cost_fields_w1.argmin(axis=1)

            # V(s) = W_0(s): continue advances to W_1(new_score), only the s_curr == s row.
            w1_full = np.zeros(s + 1, dtype=np.float64)
            w1_full[2:] = w1_vec
            cost_vec_s = np.where(
                continue_mat[s_idx], w1_full[ns_clipped[s_idx]], 0.0
            )
            cost_vec_s = np.where(bust_mat[s_idx], v_self, cost_vec_s)
            cost_field_s = 1.0 + flat_cube.T @ cost_vec_s.astype(flat_cube.dtype)
            v_new = float(cost_field_s.min())
            idx_first = int(cost_field_s.argmin())

            diff = abs(v_new - v_self)
            v_self = v_new
            if diff < fixed_point_tol:
                break

        v[s] = v_self
        best_first_aim[s] = (idx_first // n, idx_first % n)
        for sci, s_curr in enumerate(range(2, s + 1)):
            mid_turn_aim[(s, s_curr, 1)] = divmod(int(aim_w2_row[sci]), n)
            mid_turn_aim[(s, s_curr, 2)] = divmod(int(aim_w1_row[sci]), n)

    return OfficialSolution(
        goal=goal,
        sigma_x=pc.sigma_x,
        sigma_y=pc.sigma_y,
        resolution=pc.resolution,
        v=v,
        best_first_aim=best_first_aim,
        mid_turn_aim=mid_turn_aim,
    )
