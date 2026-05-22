"""Enumeration of the discrete outcomes of a single dart throw.

A dart can land in one of ~63 distinguishable regions on the board (or off
the board entirely). We give each one a stable index so the probability
cube has a fixed, well-known layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Outcome:
    name: str          # e.g. "S20", "D5", "T19", "BULL_25", "BULL_50", "MISS"
    value: int         # the score this outcome contributes
    is_double: bool    # True for D1..D20 and the bullseye (counts as double-bull)


def _build_outcomes() -> list[Outcome]:
    outs: list[Outcome] = [Outcome("MISS", 0, False)]
    for n in range(1, 21):
        outs.append(Outcome(f"S{n}", n, False))
    for n in range(1, 21):
        outs.append(Outcome(f"D{n}", 2 * n, True))
    for n in range(1, 21):
        outs.append(Outcome(f"T{n}", 3 * n, False))
    outs.append(Outcome("BULL_25", 25, False))
    outs.append(Outcome("BULL_50", 50, True))
    return outs


OUTCOMES: list[Outcome] = _build_outcomes()
N_OUTCOMES: int = len(OUTCOMES)
OUTCOME_INDEX: dict[str, int] = {o.name: i for i, o in enumerate(OUTCOMES)}

VALUES: np.ndarray = np.array([o.value for o in OUTCOMES], dtype=np.int64)
IS_DOUBLE: np.ndarray = np.array([o.is_double for o in OUTCOMES], dtype=bool)
