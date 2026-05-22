"""Probability cubes via FFT convolution.

For a given player aim error (diagonal-covariance Gaussian) this module
computes ``P[k, i, j] = Prob(outcome k | aim at pixel (i, j))`` for every
single-dart outcome in :mod:`darts.outcomes`. The implementation is a stack
of FFT convolutions, one per outcome, of the binary region mask with the
Gaussian kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from . import board
from .ev import gaussian_kernel
from .outcomes import IS_DOUBLE, N_OUTCOMES, VALUES


@dataclass(frozen=True)
class ProbabilityCube:
    """Probability of each single-dart outcome at every aim pixel."""

    cube: np.ndarray              # shape (N_OUTCOMES, resolution, resolution)
    resolution: int
    sigma_x: float
    sigma_y: float

    @property
    def pixel_size(self) -> float:
        return 1.0 / (self.resolution - 1)

    def pixel_to_board(self, i: int, j: int) -> tuple[float, float]:
        return -0.5 + i * self.pixel_size, -0.5 + j * self.pixel_size


def outcome_masks(resolution: int) -> np.ndarray:
    """Binary masks ``M[k, i, j] = 1`` iff pixel ``(i, j)`` is in region ``k``."""
    region = board.rasterize_outcome_index(resolution)
    masks = np.zeros((N_OUTCOMES, resolution, resolution), dtype=np.float32)
    for k in range(N_OUTCOMES):
        masks[k] = region == k
    return masks


def probability_cube(
    sigma_x: float,
    sigma_y: float,
    resolution: int = 256,
    truncate: float = 5.0,
    masks: np.ndarray | None = None,
) -> ProbabilityCube:
    """Compute the per-outcome probability cube for a given player."""
    if masks is None:
        masks = outcome_masks(resolution)
    elif masks.shape != (N_OUTCOMES, resolution, resolution):
        raise ValueError(
            f"masks shape {masks.shape} does not match expected "
            f"({N_OUTCOMES}, {resolution}, {resolution})"
        )

    pixel_size = 1.0 / (resolution - 1)
    kernel = gaussian_kernel(sigma_x, sigma_y, pixel_size, truncate=truncate).astype(
        np.float32
    )
    cube = np.empty_like(masks, dtype=np.float32)
    for k in range(N_OUTCOMES):
        cube[k] = fftconvolve(masks[k], kernel, mode="same")

    # Mass that the convolution leaked off the grid corresponds to throws that
    # landed off-board (a MISS). Add it back to the MISS layer so the cube is
    # exactly normalized at every aim point.
    leak = 1.0 - cube.sum(axis=0)
    cube[0] += leak  # outcome 0 = MISS

    return ProbabilityCube(
        cube=cube, resolution=resolution, sigma_x=sigma_x, sigma_y=sigma_y
    )


def ev_from_cube(pc: ProbabilityCube) -> np.ndarray:
    """Derive the EV map from a probability cube (sanity check / utility)."""
    return np.einsum("kij,k->ij", pc.cube, VALUES.astype(pc.cube.dtype))


def double_finishing_prob(pc: ProbabilityCube) -> dict[int, np.ndarray]:
    """For each finishing value v (even 2..40 or 50), return its prob-map.

    A finishing value is a score that can be reached via a *double* (or the
    bullseye, which counts as double-bull). These are the only outcomes that
    legally close out the game under the double-out rule.
    """
    out: dict[int, np.ndarray] = {}
    for k in range(N_OUTCOMES):
        if not IS_DOUBLE[k]:
            continue
        v = int(VALUES[k])
        if v in out:
            out[v] = out[v] + pc.cube[k]
        else:
            out[v] = pc.cube[k].copy()
    return out
