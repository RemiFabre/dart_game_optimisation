"""On-disk format for solver outputs.

Each artifact is a pair of files:

- ``<name>.npy`` (or ``.npz``) containing the raw arrays.
- ``<name>.json`` describing what's inside: kind, sigma, resolution, board
  dimensions, axis conventions, units, etc.

Keeping metadata in a sidecar means the viz layer can render correctly
without having to hard-code the solver's choices.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import board
from .ev import EvResult
from .endgame import OfficialSolution, SimpleSolution
from .proba import ProbabilityCube


_COMMON_METADATA = {
    "board_diameter_mm": board.TOTAL_DIAM,
    "extent_normalized": [-0.5, 0.5],
    "axis_0": "x_board (horizontal; +x toward the 6, right)",
    "axis_1": "y_board (vertical; +y toward the 20, up)",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_ev_map(result: EvResult, out_dir: Path, name: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"ev_sx{result.sigma_x}_sy{result.sigma_y}_n{result.resolution}"
    np.save(out_dir / f"{name}.npy", result.ev.astype(np.float32))
    _write_json(
        out_dir / f"{name}.json",
        {
            "kind": "ev_map",
            "sigma_x": result.sigma_x,
            "sigma_y": result.sigma_y,
            "resolution": result.resolution,
            "value_units": "expected score per dart",
            **_COMMON_METADATA,
        },
    )
    return out_dir / f"{name}.npy"


def save_probability_cube(pc: ProbabilityCube, out_dir: Path, name: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"cube_sx{pc.sigma_x}_sy{pc.sigma_y}_n{pc.resolution}"
    np.save(out_dir / f"{name}.npy", pc.cube)
    _write_json(
        out_dir / f"{name}.json",
        {
            "kind": "probability_cube",
            "sigma_x": pc.sigma_x,
            "sigma_y": pc.sigma_y,
            "resolution": pc.resolution,
            "axis_0_kind": "outcome_index",
            "axis_1": _COMMON_METADATA["axis_0"],
            "axis_2": _COMMON_METADATA["axis_1"],
            "outcome_index_layout": "0=MISS; 1..20=S1..S20; 21..40=D1..D20; 41..60=T1..T20; 61=BULL_25; 62=BULL_50",
            "value_units": "probability",
            "board_diameter_mm": board.TOTAL_DIAM,
            "extent_normalized": [-0.5, 0.5],
        },
    )
    return out_dir / f"{name}.npy"


def save_simple_solution(sol: SimpleSolution, out_dir: Path, name: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"endgame_simple_sx{sol.sigma_x}_sy{sol.sigma_y}_g{sol.goal}_n{sol.resolution}"
    np.savez(out_dir / f"{name}.npz", v=sol.v, best_aim=sol.best_aim)
    _write_json(
        out_dir / f"{name}.json",
        {
            "kind": "endgame_simple",
            "sigma_x": sol.sigma_x,
            "sigma_y": sol.sigma_y,
            "goal": sol.goal,
            "resolution": sol.resolution,
            "v_units": "expected throws to finish (single-dart, no double-out, misses count as busts)",
            "best_aim_format": "pixel (i, j) into the (N, N) board grid",
            **_COMMON_METADATA,
        },
    )
    return out_dir / f"{name}.npz"


def save_official_solution(
    sol: OfficialSolution, out_dir: Path, name: str | None = None
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"endgame_official_sx{sol.sigma_x}_sy{sol.sigma_y}_g{sol.goal}_n{sol.resolution}"

    keys = sorted(sol.mid_turn_aim.keys())
    mid_array = np.array(
        [[k[0], k[1], k[2], *sol.mid_turn_aim[k]] for k in keys], dtype=np.int32
    )
    np.savez(
        out_dir / f"{name}.npz",
        v=sol.v,
        best_first_aim=sol.best_first_aim,
        mid_turn_aim=mid_array,
    )
    _write_json(
        out_dir / f"{name}.json",
        {
            "kind": "endgame_official",
            "sigma_x": sol.sigma_x,
            "sigma_y": sol.sigma_y,
            "goal": sol.goal,
            "resolution": sol.resolution,
            "v_units": "expected throws to finish (501-style, 3-dart turns, double-out)",
            "best_first_aim_format": "pixel (i, j) for dart 1 of a fresh turn at score s",
            "mid_turn_aim_columns": "[s_turn_start, s_curr, darts_left, aim_i, aim_j]",
            **_COMMON_METADATA,
        },
    )
    return out_dir / f"{name}.npz"


def load_npy(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an ``.npy`` artifact together with its sidecar metadata."""
    arr = np.load(path)
    meta_path = path.with_suffix(".json")
    with open(meta_path) as f:
        meta = json.load(f)
    return arr, meta


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load an ``.npz`` artifact together with its sidecar metadata."""
    data = dict(np.load(path))
    meta_path = path.with_suffix(".json")
    with open(meta_path) as f:
        meta = json.load(f)
    return data, meta
