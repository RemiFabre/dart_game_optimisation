"""Round-trip tests for the solver I/O layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from darts.endgame import solve_official, solve_simple
from darts.ev import expected_value_map
from darts.io import (
    load_npy,
    load_npz,
    save_ev_map,
    save_official_solution,
    save_probability_cube,
    save_simple_solution,
)
from darts.proba import probability_cube


def test_ev_map_roundtrip(tmp_path: Path) -> None:
    result = expected_value_map(0.07, 0.07, resolution=32)
    npy_path = save_ev_map(result, tmp_path, name="ev_test")
    arr, meta = load_npy(npy_path)
    np.testing.assert_allclose(arr, result.ev, atol=1e-5)
    assert meta["kind"] == "ev_map"
    assert meta["sigma_x"] == 0.07
    assert meta["resolution"] == 32


def test_probability_cube_roundtrip(tmp_path: Path) -> None:
    pc = probability_cube(0.07, 0.07, resolution=32)
    npy_path = save_probability_cube(pc, tmp_path, name="cube_test")
    arr, meta = load_npy(npy_path)
    np.testing.assert_allclose(arr, pc.cube, atol=1e-5)
    assert meta["kind"] == "probability_cube"
    assert arr.shape == (63, 32, 32)


def test_simple_solution_roundtrip(tmp_path: Path) -> None:
    pc = probability_cube(0.07, 0.07, resolution=32)
    sol = solve_simple(pc, goal=20)
    npz_path = save_simple_solution(sol, tmp_path, name="sol_simple_test")
    data, meta = load_npz(npz_path)
    np.testing.assert_allclose(data["v"], sol.v)
    np.testing.assert_array_equal(data["best_aim"], sol.best_aim)
    assert meta["kind"] == "endgame_simple"


def test_official_solution_roundtrip(tmp_path: Path) -> None:
    pc = probability_cube(0.07, 0.07, resolution=32)
    sol = solve_official(pc, goal=10, fixed_point_max_iter=20)
    npz_path = save_official_solution(sol, tmp_path, name="sol_official_test")
    data, meta = load_npz(npz_path)
    np.testing.assert_allclose(data["v"], sol.v, equal_nan=True)
    np.testing.assert_array_equal(data["best_first_aim"], sol.best_first_aim)
    # mid_turn_aim has shape (n_entries, 5): [s, s_curr, darts_left, i, j]
    assert data["mid_turn_aim"].shape[1] == 5
    assert meta["kind"] == "endgame_official"
