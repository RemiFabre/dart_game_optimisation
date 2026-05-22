"""Tests for the covariance estimator."""

from __future__ import annotations

import numpy as np

from darts.covariance import fit_gaussian


def test_recovers_known_sigma() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    true_mean = np.array([0.1, -0.05])
    true_cov = np.diag([0.07**2, 0.05**2])
    samples = rng.multivariate_normal(true_mean, true_cov, n)
    fit = fit_gaussian(samples)
    np.testing.assert_allclose(fit.mean, true_mean, atol=0.005)
    assert abs(fit.sigma_x - 0.07) < 0.005
    assert abs(fit.sigma_y - 0.05) < 0.005
    assert abs(fit.correlation) < 0.05


def test_aim_provided_uses_aim_as_mean() -> None:
    rng = np.random.default_rng(1)
    aim = np.array([0.3, 0.0])
    # Player is biased: aims at 0.3 but actually shoots at 0.32 on average.
    samples = rng.multivariate_normal([0.32, 0.0], np.diag([0.07**2, 0.07**2]), 2000)
    fit_with_aim = fit_gaussian(samples, aim=aim)
    fit_no_aim = fit_gaussian(samples)
    # When aim is provided, mean is the aim itself; covariance absorbs the bias.
    np.testing.assert_array_equal(fit_with_aim.mean, aim)
    # sigma_x with aim should be slightly larger because the bias is absorbed.
    assert fit_with_aim.sigma_x > fit_no_aim.sigma_x


def test_too_few_points_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        fit_gaussian(np.array([[0.1, 0.2]]))


def test_recovers_correlated_covariance() -> None:
    rng = np.random.default_rng(2)
    true_cov = np.array([[0.07**2, 0.5 * 0.07 * 0.05], [0.5 * 0.07 * 0.05, 0.05**2]])
    samples = rng.multivariate_normal([0.0, 0.0], true_cov, 5000)
    fit = fit_gaussian(samples)
    np.testing.assert_allclose(fit.cov, true_cov, atol=1e-3)
