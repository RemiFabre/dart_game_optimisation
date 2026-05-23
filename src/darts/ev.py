"""Expected-value computation via FFT convolution.

Given a player's Gaussian aim error (diagonal covariance) the expected score
when aiming at a point ``p`` is the convolution of the dartboard score field
with the Gaussian kernel evaluated at ``p``. Computing the convolution on a
grid yields the EV for every aim point on that grid in a single shot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from . import board


@dataclass(frozen=True)
class EvResult:
    """EV map plus metadata describing the grid it lives on."""

    ev: np.ndarray            # (resolution, resolution): axis 0 = x (horizontal), axis 1 = y (vertical)
    resolution: int
    sigma_x: float
    sigma_y: float

    @property
    def pixel_size(self) -> float:
        """Side of one pixel in normalized board units."""
        return 1.0 / (self.resolution - 1)

    def pixel_to_board(self, i: int, j: int) -> tuple[float, float]:
        return -0.5 + i * self.pixel_size, -0.5 + j * self.pixel_size

    def argmax_board(self) -> tuple[tuple[float, float], float]:
        idx = int(np.argmax(self.ev))
        i, j = divmod(idx, self.ev.shape[1])
        return self.pixel_to_board(i, j), float(self.ev[i, j])


def gaussian_kernel(
    sigma_x: float,
    sigma_y: float,
    pixel_size: float,
    truncate: float = 5.0,
) -> np.ndarray:
    """Build a normalized 2D Gaussian kernel sampled on the same pixel grid.

    ``sigma_x`` / ``sigma_y`` are in normalized board units, matching the
    board coordinate system. ``truncate`` is how many sigmas the kernel
    extends in each direction (5σ keeps the tail mass below 1e-6).
    """
    sx_px = max(sigma_x / pixel_size, 1e-9)
    sy_px = max(sigma_y / pixel_size, 1e-9)

    half_x = int(math.ceil(truncate * sx_px))
    half_y = int(math.ceil(truncate * sy_px))

    kx = np.arange(-half_x, half_x + 1, dtype=np.float64)
    ky = np.arange(-half_y, half_y + 1, dtype=np.float64)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kernel = np.exp(-0.5 * ((KX / sx_px) ** 2 + (KY / sy_px) ** 2))
    kernel /= kernel.sum()
    return kernel


def expected_value_map(
    sigma_x: float,
    sigma_y: float,
    resolution: int = 512,
    truncate: float = 5.0,
    score_field: np.ndarray | None = None,
) -> EvResult:
    """Compute the EV map for a diagonal-covariance Gaussian miss model.

    ``score_field`` is optional; if not provided it is rasterized from the
    canonical board geometry at the requested resolution.
    """
    if score_field is None:
        score_field = board.rasterize_score(resolution)
    elif score_field.shape != (resolution, resolution):
        raise ValueError(
            f"score_field shape {score_field.shape} does not match resolution {resolution}"
        )

    pixel_size = 1.0 / (resolution - 1)
    kernel = gaussian_kernel(sigma_x, sigma_y, pixel_size, truncate=truncate)
    ev = fftconvolve(score_field, kernel, mode="same")
    return EvResult(ev=ev, resolution=resolution, sigma_x=sigma_x, sigma_y=sigma_y)


def expected_value_at(
    aim_x: float,
    aim_y: float,
    sigma_x: float,
    sigma_y: float,
    resolution: int = 1024,
    truncate: float = 5.0,
) -> float:
    """Convenience: EV at a single aim point, with bilinear sampling of the grid.

    Slow if called many times; prefer ``expected_value_map`` for sweeps.
    """
    result = expected_value_map(sigma_x, sigma_y, resolution=resolution, truncate=truncate)
    inv = 1.0 / result.pixel_size
    fi = (aim_x + 0.5) * inv
    fj = (aim_y + 0.5) * inv
    i0 = int(np.floor(fi))
    j0 = int(np.floor(fj))
    di = fi - i0
    dj = fj - j0
    n = resolution
    i0 = max(0, min(i0, n - 2))
    j0 = max(0, min(j0, n - 2))
    ev = result.ev
    return float(
        (1 - di) * (1 - dj) * ev[i0, j0]
        + di * (1 - dj) * ev[i0 + 1, j0]
        + (1 - di) * dj * ev[i0, j0 + 1]
        + di * dj * ev[i0 + 1, j0 + 1]
    )
