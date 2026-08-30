"""
Reusable estimator-statistics helpers -- CORRECTED aggregation math.

This module is the single source of truth for pooling per-episode
position-estimation error statistics into aggregate (multi-episode)
mean-error and RMSE figures.

Background / bug this module fixes (see
experiments/run_measurement_noise_estimator_correction.py and
test_measurement_noise_estimator_aggregation.py):

Each episode's rollout (experiments/measurement_noise_eval_utils.py)
records, over its own n_i post-detection timesteps:

    m_i = mean_position_estimation_error_i = (1/n_i) * sum_t ||e_t||
    r_i = position_estimation_rmse_i        = sqrt((1/n_i) * sum_t ||e_t||^2)
    n_i = n_position_estimation_samples_i

An earlier analysis script incorrectly computed an aggregate RMSE as
sqrt(mean(m_i^2)) -- i.e. it squared the PER-EPISODE MEAN errors instead of
reconstructing the pooled sum of squared timestep errors. Since
Var(||e_t||) > 0, sqrt(mean(m_i^2)) systematically UNDER-estimates the true
pooled RMSE (Jensen's inequality: E[m_i^2] = E[m_i]^2 + Var(m_i) >= E[m_i]^2,
so this specific error only partly explains the gap -- the deeper issue is
that m_i is a MEAN-of-norms while the true pooled RMSE requires the
MEAN-of-squared-norms, and mean-of-norms != sqrt(mean-of-squared-norms) even
within a single episode).

The correct reconstruction uses n_i and r_i (already exactly known per
episode) to recover the total sum of squared timestep errors:

    sum_t e_t^2 (episode i) = n_i * r_i^2

so the POOLED (sample-weighted) RMSE across episodes is:

    rmse_sample_weighted = sqrt( sum_i(n_i * r_i^2) / sum_i(n_i) )

and the POOLED (sample-weighted) mean error is:

    mean_error_sample_weighted = sum_i(n_i * m_i) / sum_i(n_i)
"""

from __future__ import annotations

import math

import numpy as np


# =============================================================================
# Theoretical benchmarks for isotropic 3-D Gaussian sensor noise
# epsilon ~ N(0, variance * I_3), sigma = sqrt(variance)
# =============================================================================
def theoretical_raw_measurement_mean_error(variance: float) -> float:
    """
    Theoretical expected Euclidean norm E[||epsilon||] of isotropic 3-D
    Gaussian noise (chi distribution, 3 dof, scaled by sigma):

        E[||epsilon||] = 2 * sqrt(2/pi) * sigma  ~= 1.5957691216 * sigma
    """
    sigma = math.sqrt(variance)
    return 2.0 * math.sqrt(2.0 / math.pi) * sigma


def theoretical_raw_measurement_rmse(variance: float) -> float:
    """
    Theoretical 3-D vector RMSE sqrt(E[||epsilon||^2]) = sqrt(3) * sigma
    = sqrt(3 * variance).
    """
    return math.sqrt(3.0 * variance)


# =============================================================================
# Sample-weighted (pooled, timestep-correct) aggregation
# =============================================================================
def sample_weighted_mean_error(n: np.ndarray, mean_error: np.ndarray) -> float:
    """
    sum_i(n_i * m_i) / sum_i(n_i) -- the exact pooled mean Euclidean error
    across all underlying timesteps (n, mean_error are per-episode arrays).
    """
    n = np.asarray(n, dtype=np.float64)
    mean_error = np.asarray(mean_error, dtype=np.float64)
    total_n = n.sum()
    if total_n == 0:
        return float("nan")
    return float(np.sum(n * mean_error) / total_n)


def sample_weighted_rmse(n: np.ndarray, rmse: np.ndarray) -> float:
    """
    sqrt( sum_i(n_i * r_i^2) / sum_i(n_i) ) -- the exact pooled RMSE across
    all underlying timesteps, reconstructed from each episode's own RMSE and
    sample count (n_i * r_i^2 recovers that episode's sum-of-squared-errors).
    """
    n = np.asarray(n, dtype=np.float64)
    rmse = np.asarray(rmse, dtype=np.float64)
    total_n = n.sum()
    if total_n == 0:
        return float("nan")
    return float(np.sqrt(np.sum(n * np.square(rmse)) / total_n))


# =============================================================================
# Episode-weighted (simple, descriptive-only) aggregation -- NOT an RMSE
# reconstruction; every episode counts equally regardless of sample count.
# =============================================================================
def episode_weighted_mean_error(mean_error: np.ndarray) -> float:
    """mean_i(m_i) -- simple unweighted mean of per-episode mean errors."""
    mean_error = np.asarray(mean_error, dtype=np.float64)
    if mean_error.size == 0:
        return float("nan")
    return float(np.mean(mean_error))


def mean_episode_rmse(rmse: np.ndarray) -> float:
    """
    mean_i(r_i) -- simple unweighted mean of per-episode RMSE values. This is
    a valid DESCRIPTIVE statistic (average of episode-level RMSEs) but is NOT
    itself an aggregate/pooled RMSE -- never label it "RMSE" without
    qualifying it as episode-weighted.
    """
    rmse = np.asarray(rmse, dtype=np.float64)
    if rmse.size == 0:
        return float("nan")
    return float(np.mean(rmse))


# =============================================================================
# PPO / PPO-Kalman: equal-weight aggregation across the 3 training seeds
# =============================================================================
def equal_weight_across_seeds(per_seed_values: np.ndarray) -> tuple[float, float]:
    """
    Paper-level PPO statistic: equal-weight mean of the 3 per-training-seed
    values (NOT a pooled/sample-weighted mean across 3000 episodes -- the
    three trained policies are NOT independent controller replications; each
    seed's own sample-weighted statistic is computed first, then the 3
    resulting numbers are averaged with equal weight).

    Returns (mean_across_seeds, sd_across_seeds).
    """
    per_seed_values = np.asarray(per_seed_values, dtype=np.float64)
    return float(np.mean(per_seed_values)), float(np.std(per_seed_values))
