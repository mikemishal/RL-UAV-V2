"""
Statistics helpers for the locked final nominal evaluation.

All functions here are pure/stateless and operate on plain numpy arrays of
per-episode outcomes (1/0 success indicators), so they can be unit-tested
independently of the environment/policy machinery.

Terminology:
    - "Wilson score interval": the recommended binomial CI for a single
      fixed method's success rate (see experiments/final_experiment_protocol.py
      item 28) -- avoids the normal-approximation interval's known problems
      near 0/1.
    - "paired bootstrap": for two FIXED (single-instance) methods evaluated
      on the SAME matched episode seeds (common random numbers) -- resamples
      episode indices only.
    - "hierarchical bootstrap"/"hierarchical paired bootstrap": for methods
      with THREE independently trained PPO models -- resamples both the
      training-seed axis (3 models, with replacement) AND the episode axis
      (5000 episodes, with replacement), reflecting that PPO's paper-level
      success rate has two sources of sampling variation: environment
      scenario variability and training-seed variability. See
      experiments/final_experiment_protocol.py items 31/33/34.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p_hat = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))) / denom
    return (center - half, center + half)


def standard_error(successes: int, n: int) -> float:
    """Binomial standard error of the sample proportion."""
    if n == 0:
        return float("nan")
    p_hat = successes / n
    return float(np.sqrt(p_hat * (1 - p_hat) / n))


def mcnemar_exact(n01: int, n10: int) -> float:
    """
    Exact two-sided McNemar test p-value from discordant-pair counts.

    n01: method A fails, method B succeeds (A loses / B wins).
    n10: method A succeeds, method B fails (A wins / B loses).

    Uses the exact binomial test on the discordant pairs under H0: p=0.5
    (the standard exact McNemar formulation), NOT the chi-square approximation.
    """
    n = n01 + n10
    if n == 0:
        return 1.0
    k = int(min(n01, n10))
    result = stats.binomtest(k, n, 0.5, alternative="two-sided")
    return float(result.pvalue)


def paired_bootstrap_ci(
    success_a: np.ndarray,
    success_b: np.ndarray,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Paired bootstrap CI for mean(success_a - success_b) when BOTH methods
    are fixed (single instance) and evaluated on the SAME matched episode
    seeds (common random numbers). Resamples episode indices only (with
    replacement), applying the SAME resampled indices to both arrays.

    Returns (observed_diff, ci_low, ci_high).
    """
    success_a = np.asarray(success_a, dtype=np.float64)
    success_b = np.asarray(success_b, dtype=np.float64)
    n = len(success_a)
    assert len(success_b) == n, "paired arrays must have equal length (matched seeds)"

    rng = np.random.default_rng(seed)
    observed = float(np.mean(success_a) - np.mean(success_b))
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(success_a[idx]) - np.mean(success_b[idx])
    alpha = 1 - confidence
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return observed, float(lo), float(hi)


def hierarchical_bootstrap_track(
    success_matrix: np.ndarray,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Hierarchical bootstrap 95% CI for a single PPO track's paper-level
    (equal-weight, 3-training-seed) success rate.

    success_matrix: shape (n_training_seeds, n_episodes) of 0/1 outcomes.

    Each replicate: (1) resample training-seed rows with replacement,
    (2) resample episode columns with replacement, (3) compute the
    equal-weight mean success over the sampled rows/columns.

    Returns (observed_mean, ci_low, ci_high), where observed_mean is the
    unresampled equal-weight mean of the per-seed success rates.
    """
    success_matrix = np.asarray(success_matrix, dtype=np.float64)
    n_seeds, n_eps = success_matrix.shape
    rng = np.random.default_rng(seed)
    observed = float(np.nanmean(success_matrix, axis=1).mean())
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        ep_idx = rng.integers(0, n_eps, size=n_eps)
        means[i] = np.nanmean(success_matrix[np.ix_(seed_idx, ep_idx)])
    alpha = 1 - confidence
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return observed, float(lo), float(hi)


def hierarchical_paired_bootstrap_vs_scripted(
    ppo_matrix: np.ndarray,
    scripted_vec: np.ndarray,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Hierarchical paired bootstrap for PPO (3 training seeds) vs a fixed
    scripted comparator, using the SAME resampled episode indices for both
    (avoids pseudoreplicating the 5000 environment scenarios).

    ppo_matrix: shape (n_training_seeds, n_episodes).
    scripted_vec: shape (n_episodes,), SAME episode-seed order as ppo_matrix's columns.

    Each replicate: (1) resample PPO training-seed rows with replacement,
    (2) resample episode indices with replacement (shared by both sides),
    (3) diff = mean(PPO over sampled rows/cols) - mean(scripted over sampled cols).

    Returns (observed_diff, ci_low, ci_high).
    """
    ppo_matrix = np.asarray(ppo_matrix, dtype=np.float64)
    scripted_vec = np.asarray(scripted_vec, dtype=np.float64)
    n_seeds, n_eps = ppo_matrix.shape
    assert len(scripted_vec) == n_eps, "PPO and scripted comparator must share the same episode seeds"

    rng = np.random.default_rng(seed)
    observed = float(np.nanmean(ppo_matrix, axis=1).mean() - np.nanmean(scripted_vec))
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        ep_idx = rng.integers(0, n_eps, size=n_eps)
        ppo_sampled = np.nanmean(ppo_matrix[np.ix_(seed_idx, ep_idx)])
        scripted_sampled = np.nanmean(scripted_vec[ep_idx])
        diffs[i] = ppo_sampled - scripted_sampled
    alpha = 1 - confidence
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return observed, float(lo), float(hi)


def hierarchical_paired_bootstrap_matched_seeds(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, float, np.ndarray]:
    """
    Hierarchical paired bootstrap for two 3-training-seed PPO tracks
    compared with MATCHED training-seed labels (e.g. PPO-Kalman seed 42 vs
    PPO seed 42), per experiments/final_experiment_protocol.py item 34.

    a_matrix, b_matrix: shape (n_training_seeds, n_episodes), SAME seed-label
        order (row i of a_matrix and row i of b_matrix are the SAME training
        seed) and SAME episode-seed order.

    Each replicate: (1) resample training-seed-label indices with
    replacement (SAME indices applied to both a_matrix and b_matrix -- the
    matching is preserved), (2) resample episode indices with replacement
    (shared by both sides), (3) diff = mean(a over sampled) - mean(b over sampled).

    Returns (observed_mean_diff, ci_low, ci_high, per_seed_diffs) where
    per_seed_diffs is the array of the 3 UNRESAMPLED per-training-seed
    differences (a_matrix.mean(axis=1) - b_matrix.mean(axis=1)).
    """
    a_matrix = np.asarray(a_matrix, dtype=np.float64)
    b_matrix = np.asarray(b_matrix, dtype=np.float64)
    assert a_matrix.shape == b_matrix.shape, "matched-seed comparison requires identical shapes"
    n_seeds, n_eps = a_matrix.shape

    per_seed_diffs = np.nanmean(a_matrix, axis=1) - np.nanmean(b_matrix, axis=1)
    observed = float(per_seed_diffs.mean())

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)  # same labels for both tracks
        ep_idx = rng.integers(0, n_eps, size=n_eps)
        a_sampled = np.nanmean(a_matrix[np.ix_(seed_idx, ep_idx)])
        b_sampled = np.nanmean(b_matrix[np.ix_(seed_idx, ep_idx)])
        diffs[i] = a_sampled - b_sampled
    alpha = 1 - confidence
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return observed, float(lo), float(hi), per_seed_diffs


def hierarchical_bootstrap_independent_families(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    n_boot: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Hierarchical bootstrap comparing two multi-training-seed families whose
    training roots are NOT matched/comparable (e.g. LR-PPO roots 55/56/57 vs
    standalone PPO roots 45/46/47 -- there is no meaningful pairing between
    the two families' training roots, unlike hierarchical_paired_bootstrap_
    matched_seeds' same-label design). Each family's training-root rows are
    resampled INDEPENDENTLY; the SAME resampled environment-seed indices are
    applied to BOTH families (common random numbers preserved at the
    environment-scenario level only).

    a_matrix: shape (n_a_roots, n_episodes).
    b_matrix: shape (n_b_roots, n_episodes), SAME episode-seed column order
        as a_matrix (n_a_roots and n_b_roots may differ).

    Each replicate: (1) resample a's training-root rows with replacement,
    (2) INDEPENDENTLY resample b's training-root rows with replacement,
    (3) resample episode indices ONCE with replacement (shared by both),
    (4) diff = mean(a over sampled rows/cols) - mean(b over sampled rows,
    SAME cols).

    Returns (observed_diff, ci_low, ci_high), where observed_diff is the
    UNRESAMPLED equal-weight-per-family mean difference
    (mean(a_matrix, axis=1).mean() - mean(b_matrix, axis=1).mean()).
    """
    a_matrix = np.asarray(a_matrix, dtype=np.float64)
    b_matrix = np.asarray(b_matrix, dtype=np.float64)
    n_a_roots, n_eps = a_matrix.shape
    n_b_roots, n_eps_b = b_matrix.shape
    assert n_eps == n_eps_b, "both families must share the same episode-seed columns"

    observed = float(np.nanmean(a_matrix, axis=1).mean() - np.nanmean(b_matrix, axis=1).mean())

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        a_idx = rng.integers(0, n_a_roots, size=n_a_roots)
        b_idx = rng.integers(0, n_b_roots, size=n_b_roots)  # independent of a_idx
        ep_idx = rng.integers(0, n_eps, size=n_eps)  # SAME episode indices for both families
        a_sampled = np.nanmean(a_matrix[np.ix_(a_idx, ep_idx)])
        b_sampled = np.nanmean(b_matrix[np.ix_(b_idx, ep_idx)])
        diffs[i] = a_sampled - b_sampled
    alpha = 1 - confidence
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return observed, float(lo), float(hi)
