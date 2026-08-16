"""Regression tests for the measurement-noise estimator-aggregation correction.

Covers (per the correction task spec):
    A. Reconstructed sample-weighted RMSE equals exact pooled timestep RMSE
       for synthetic episodes with known individual timestep errors.
    B. The old incorrect sqrt(mean(per_episode_mean_error^2)) differs from
       the correct RMSE on a constructed example.
    C. Sample-count weighting is correct for unequal episode lengths.
    D. theoretical mean-error formula equals 2*sqrt(2/pi)*sigma.
    E. theoretical RMSE equals sqrt(3)*sigma.
    F. For synthetic isotropic Gaussian samples, empirical mean norm is
       close to the theoretical mean.
    G. For synthetic isotropic Gaussian samples, empirical RMS norm is
       close to sqrt(3)*sigma.
    H. PPO aggregation retains three training-seed identities and
       equal-weights seed-level results.

Run directly:
    python test_measurement_noise_estimator_aggregation.py
"""

from __future__ import annotations

import math

import numpy as np

from experiments.estimator_stats import (
    theoretical_raw_measurement_mean_error,
    theoretical_raw_measurement_rmse,
    sample_weighted_mean_error,
    sample_weighted_rmse,
    episode_weighted_mean_error,
    mean_episode_rmse,
    equal_weight_across_seeds,
)


def _episode_stats_from_timestep_errors(timestep_errors: list[np.ndarray]) -> tuple[int, float, float]:
    """Given raw per-timestep error VECTORS for one episode, compute (n, m, r) exactly
    as experiments/measurement_noise_eval_utils.py does."""
    norms = np.array([np.linalg.norm(e) for e in timestep_errors])
    n = len(norms)
    m = float(np.mean(norms))
    r = float(np.sqrt(np.mean(np.square(norms))))
    return n, m, r


# ---------------------------------------------------------------------------
# A. Reconstructed sample-weighted RMSE equals exact pooled timestep RMSE
# ---------------------------------------------------------------------------
def test_sample_weighted_rmse_equals_pooled_timestep_rmse():
    rng = np.random.default_rng(0)
    episodes_raw = [rng.normal(0, 1.0, size=(7, 3)), rng.normal(0, 1.0, size=(23, 3)), rng.normal(0, 1.0, size=(5, 3))]
    n_list, m_list, r_list = [], [], []
    all_norms = []
    for ep in episodes_raw:
        n, m, r = _episode_stats_from_timestep_errors(list(ep))
        n_list.append(n)
        m_list.append(m)
        r_list.append(r)
        all_norms.extend(np.linalg.norm(ep, axis=1).tolist())

    true_pooled_rmse = float(np.sqrt(np.mean(np.square(all_norms))))
    reconstructed = sample_weighted_rmse(np.array(n_list, dtype=np.float64), np.array(r_list, dtype=np.float64))
    assert abs(reconstructed - true_pooled_rmse) < 1e-9, (reconstructed, true_pooled_rmse)

    true_pooled_mean = float(np.mean(all_norms))
    reconstructed_mean = sample_weighted_mean_error(np.array(n_list, dtype=np.float64), np.array(m_list, dtype=np.float64))
    assert abs(reconstructed_mean - true_pooled_mean) < 1e-9, (reconstructed_mean, true_pooled_mean)


# ---------------------------------------------------------------------------
# B. Old incorrect sqrt(mean(m_i^2)) differs from the correct RMSE
# ---------------------------------------------------------------------------
def test_old_buggy_formula_differs_from_correct():
    # Construct episodes with deliberately different within-episode variance
    # so that sqrt(mean(m_i^2)) diverges from the true pooled RMSE.
    n = np.array([100.0, 100.0])
    m = np.array([1.0, 3.0])
    r = np.array([1.05, 3.4])  # r_i >= m_i always (RMS >= mean for nonneg values)

    old_buggy_rmse = float(np.sqrt(np.mean(np.square(m))))
    correct_rmse = sample_weighted_rmse(n, r)
    assert abs(old_buggy_rmse - correct_rmse) > 0.05, "old and corrected formulas should differ on this example"


# ---------------------------------------------------------------------------
# C. Sample-count weighting correct for unequal episode lengths
# ---------------------------------------------------------------------------
def test_sample_weighting_unequal_lengths():
    # One huge episode should dominate the weighted MEAN, but a single episode
    # with a large RMSE still contributes heavily to the pooled RMSE via its
    # squared error (even with a small sample count) -- verify both against
    # their exact analytic values rather than assuming a particular ordering.
    n = np.array([1.0, 999.0])
    m = np.array([100.0, 1.0])
    r = np.array([100.0, 1.0])

    weighted_mean = sample_weighted_mean_error(n, m)
    weighted_rmse = sample_weighted_rmse(n, r)
    expected_mean = (1.0 * 100.0 + 999.0 * 1.0) / 1000.0  # = 1.099
    expected_rmse = math.sqrt((1.0 * 100.0 ** 2 + 999.0 * 1.0 ** 2) / 1000.0)  # = sqrt(10.999)
    assert abs(weighted_mean - expected_mean) < 1e-9, weighted_mean
    assert abs(weighted_rmse - expected_rmse) < 1e-9, weighted_rmse

    # Simple (unweighted) episode mean would have been (100+1)/2 = 50.5 -- confirm divergence.
    naive_mean = episode_weighted_mean_error(m)
    assert naive_mean == 50.5
    assert abs(weighted_mean - naive_mean) > 40

    # A second, more realistic case: two similarly-sized episodes with a
    # third much-longer episode -- weighted mean must sit much closer to the
    # long episode's own value than the naive unweighted mean does.
    n2 = np.array([10.0, 10.0, 990.0])
    m2 = np.array([5.0, 5.0, 2.0])
    weighted_mean2 = sample_weighted_mean_error(n2, m2)
    naive_mean2 = episode_weighted_mean_error(m2)
    expected_weighted_mean2 = (10.0 * 5.0 + 10.0 * 5.0 + 990.0 * 2.0) / 1010.0  # = 2.0594...
    assert abs(weighted_mean2 - expected_weighted_mean2) < 1e-9, weighted_mean2
    assert abs(naive_mean2 - 4.0) < 1e-9, naive_mean2
    assert abs(weighted_mean2 - 2.0) < abs(naive_mean2 - 2.0)


# ---------------------------------------------------------------------------
# D/E. Theoretical formulas
# ---------------------------------------------------------------------------
def test_theoretical_mean_error_formula():
    for variance in (0.05, 0.25, 0.5, 1.0, 2.0, 4.0):
        sigma = math.sqrt(variance)
        expected = 2.0 * math.sqrt(2.0 / math.pi) * sigma
        assert abs(theoretical_raw_measurement_mean_error(variance) - expected) < 1e-12
    assert abs(theoretical_raw_measurement_mean_error(1.0) - 1.5957691216057308) < 1e-9


def test_theoretical_rmse_formula():
    for variance in (0.05, 0.25, 0.5, 1.0, 2.0, 4.0):
        sigma = math.sqrt(variance)
        expected = math.sqrt(3.0) * sigma
        assert abs(theoretical_raw_measurement_rmse(variance) - expected) < 1e-12
        assert abs(theoretical_raw_measurement_rmse(variance) - math.sqrt(3.0 * variance)) < 1e-12


# ---------------------------------------------------------------------------
# F/G. Synthetic isotropic Gaussian: empirical mean norm / RMS norm match theory
# ---------------------------------------------------------------------------
def test_empirical_gaussian_mean_and_rms_norm():
    rng = np.random.default_rng(42)
    variance = 1.3
    sigma = math.sqrt(variance)
    samples = rng.normal(0, sigma, size=(200_000, 3))
    norms = np.linalg.norm(samples, axis=1)

    empirical_mean = float(np.mean(norms))
    theoretical_mean = theoretical_raw_measurement_mean_error(variance)
    assert abs(empirical_mean - theoretical_mean) / theoretical_mean < 0.01, (empirical_mean, theoretical_mean)

    empirical_rms = float(np.sqrt(np.mean(np.square(norms))))
    theoretical_rms = theoretical_raw_measurement_rmse(variance)
    assert abs(empirical_rms - theoretical_rms) / theoretical_rms < 0.01, (empirical_rms, theoretical_rms)


# ---------------------------------------------------------------------------
# H. PPO aggregation retains 3 training-seed identities, equal-weighted
# ---------------------------------------------------------------------------
def test_ppo_equal_weight_across_seeds():
    seed_rmse = np.array([1.0, 1.2, 1.4])
    mean_rmse, sd_rmse = equal_weight_across_seeds(seed_rmse)
    assert abs(mean_rmse - 1.2) < 1e-12
    assert abs(sd_rmse - np.std(seed_rmse)) < 1e-12

    # Equal-weighting must NOT be equivalent to pooling all seeds' raw episodes
    # together (which would implicitly weight by each seed's sample count).
    # Use RMSE values/sample counts chosen so the two aggregation strategies
    # give clearly different numbers.
    seed_rmse_skewed = np.array([1.0, 1.0, 3.0])
    n_per_seed = np.array([500.0, 500.0, 2000.0])  # third (highest-RMSE) seed dominates by sample count
    equal_weight_mean, _ = equal_weight_across_seeds(seed_rmse_skewed)
    pooled_across_all_episodes = sample_weighted_rmse(n_per_seed, seed_rmse_skewed)
    assert abs(equal_weight_mean - 1.6666666666666667) < 1e-9
    assert abs(pooled_across_all_episodes - math.sqrt(19000.0 / 3000.0)) < 1e-9
    assert abs(pooled_across_all_episodes - equal_weight_mean) > 0.5, (
        "equal-weight seed aggregation must differ from naive episode-count pooling "
        "when per-seed sample counts are unequal"
    )


def main() -> int:
    tests = [
        ("A. Sample-weighted RMSE equals pooled timestep RMSE", test_sample_weighted_rmse_equals_pooled_timestep_rmse),
        ("B. Old buggy formula differs from correct RMSE", test_old_buggy_formula_differs_from_correct),
        ("C. Sample weighting correct for unequal episode lengths", test_sample_weighting_unequal_lengths),
        ("D. Theoretical mean-error formula", test_theoretical_mean_error_formula),
        ("E. Theoretical RMSE formula", test_theoretical_rmse_formula),
        ("F/G. Empirical Gaussian mean/RMS norm match theory", test_empirical_gaussian_mean_and_rms_norm),
        ("H. PPO equal-weight across training seeds", test_ppo_equal_weight_across_seeds),
    ]

    print("=== Measurement-Noise Estimator-Aggregation Correction Tests ===\n")
    n_pass = 0
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            n_pass += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            n_fail += 1
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
            n_fail += 1

    print(f"\n{n_pass} passed, {n_fail} failed out of {len(tests)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
