"""Fit a 2D Gaussian to a set of dart hits.

Given the (x, y) positions of darts in normalized board coordinates plus
(optionally) the aim point the player was using, we compute the
maximum-likelihood mean and covariance. The covariance summarises the
player's accuracy in the same units the solver consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianFit:
    mean: np.ndarray            # shape (2,), in normalized board coords
    cov: np.ndarray             # shape (2, 2)

    @property
    def sigma_x(self) -> float:
        return float(np.sqrt(self.cov[0, 0]))

    @property
    def sigma_y(self) -> float:
        return float(np.sqrt(self.cov[1, 1]))

    @property
    def correlation(self) -> float:
        denom = self.sigma_x * self.sigma_y
        if denom == 0:
            return 0.0
        return float(self.cov[0, 1] / denom)


def fit_gaussian(points: np.ndarray, aim: np.ndarray | None = None) -> GaussianFit:
    """Maximum-likelihood 2D Gaussian fit to ``points``.

    ``points`` has shape ``(N, 2)`` — each row is a (x, y) dart hit in
    normalized board coordinates. If ``aim`` is given (shape ``(2,)``), the
    covariance is computed around the aim point rather than the empirical
    mean — useful when you know the player was aiming at a specific spot
    and want to attribute all deviation to error (not aim bias).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2); got {pts.shape}")
    if pts.shape[0] < 2:
        raise ValueError("need at least 2 hits to estimate a covariance")

    if aim is None:
        mean = pts.mean(axis=0)
    else:
        mean = np.asarray(aim, dtype=np.float64).reshape(2)

    centered = pts - mean
    # Population covariance (1/N) when mean is given; sample covariance
    # (1/(N-1)) when mean is estimated. Both are valid MLE choices.
    if aim is None:
        cov = np.cov(centered.T, ddof=0)
    else:
        cov = centered.T @ centered / pts.shape[0]
    # Force symmetry against numerical drift.
    cov = 0.5 * (cov + cov.T)
    return GaussianFit(mean=mean, cov=cov)
