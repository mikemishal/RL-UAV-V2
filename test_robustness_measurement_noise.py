"""Measurement-noise robustness sweep correctness tests -- locked sweep guardrails.

Script-style test consistent with the project's other test files. No
external test framework is used; each test function performs assertions
and the runner reports PASS/FAIL per test with a final summary.

Covers (per the measurement-noise robustness sweep task spec):
    A. Variance grid exactly 0.05, 0.25, 0.50, 1.00, 2.00, 4.00.
    B. Nominal variance 0.5 included and matches EnvConfig default.
    C. Per-axis sensor std = sqrt(measurement_var) helper is correct.
    D. Sweep seeds exactly 31000..31999.
    E. No overlap with 20000..24999, 25000..25999, 30000..30999, or future (>=32000).
    F. Exactly six primary paper methods, in the prescribed set.
    G. 10 policy instances exist (4 scripted + 3 PPO + 3 PPO-Kalman).
    H. PN-Kalman / Lead-Kalman are absent from this sweep.
    I. All six PPO hashes match frozen final-nominal hashes.
    J. Greedy policy instance uses the corrected measurement contract.
    K. Sweep config differs from nominal only in measurement_var/estimator mode.
    L. process_var and lead_time remain fixed at nominal everywhere.
    M. Kalman filter's R equals measurement_var * I for the configured variance.
    N. Empirical sensor noise std matches sqrt(measurement_var).
    O. Expected raw row count = 60000.
    P. Policy-info sanitization still excludes enemy_pos/enemy_vel.
    Q. Common seed set across all variance/method combinations (construction-level).
    R. RMSE sanity-check helper equals sqrt(3 * measurement_var).
    S. Bootstrap sampling pairs evaluation seeds/training-seed labels correctly.
    T. RNG fairness: true trajectories are unaffected by measurement_var.

Run directly:
    python test_robustness_measurement_noise.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy
from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env
from uav_defend.policies.baseline.greedy_intercept_policy import GreedyInterceptPolicy
from uav_defend.tracking.kalman_filter import EnemyKalmanFilter

from experiments.final_experiment_protocol import FINAL_NOMINAL_SEEDS
from experiments.acquisition_ablation_protocol import ACQUISITION_ABLATION_SEEDS
from experiments.robustness_detection_protocol import DETECTION_SWEEP_SEEDS
from experiments.final_eval_utils import sha256_of_file
from experiments.robustness_measurement_noise_protocol import (
    MEASUREMENT_VAR_VALUES,
    NOMINAL_MEASUREMENT_VAR,
    FIXED_PROCESS_VAR,
    FIXED_LEAD_TIME,
    MEASUREMENT_SWEEP_SEED_OFFSET,
    MEASUREMENT_SWEEP_EPISODES,
    MEASUREMENT_SWEEP_SEEDS,
    RESERVED_FUTURE_ROBUSTNESS_SEEDS,
    MOBILITY_SWEEP_SEED_OFFSET,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    NOISE_POLICY_INSTANCES,
    EXPECTED_N_POLICY_INSTANCES,
    EXPECTED_RAW_ROW_COUNT,
    measurement_std,
    theoretical_raw_measurement_rmse,
    build_sweep_env_config,
    assert_only_measurement_var_differs,
    assert_measurement_sweep_seed,
)
from experiments.final_stats import hierarchical_paired_bootstrap_matched_seeds

PROJECT_ROOT = Path(__file__).parent
FINAL_NOMINAL_HASHES_PATH = PROJECT_ROOT / "results" / "final_nominal" / "model_hashes.json"
_NOMINAL_ENV_CONFIG = EnvConfig()


# ---------------------------------------------------------------------------
# A. Variance grid exact ; B. nominal 0.5 included
# ---------------------------------------------------------------------------
def test_variance_values_exact():
    assert MEASUREMENT_VAR_VALUES == (0.05, 0.25, 0.50, 1.00, 2.00, 4.00)


def test_nominal_variance_included():
    assert NOMINAL_MEASUREMENT_VAR == 0.5
    assert NOMINAL_MEASUREMENT_VAR in MEASUREMENT_VAR_VALUES
    assert _NOMINAL_ENV_CONFIG.measurement_var == NOMINAL_MEASUREMENT_VAR


# ---------------------------------------------------------------------------
# C. Per-axis sensor std helper
# ---------------------------------------------------------------------------
def test_measurement_std_helper():
    for v in MEASUREMENT_VAR_VALUES:
        assert abs(measurement_std(v) - np.sqrt(v)) < 1e-12
    assert abs(measurement_std(0.5) - 0.7071067811865476) < 1e-9


# ---------------------------------------------------------------------------
# D. Sweep seeds exactly 31000..31999
# ---------------------------------------------------------------------------
def test_sweep_seed_range():
    assert MEASUREMENT_SWEEP_SEED_OFFSET == 31_000
    assert MEASUREMENT_SWEEP_EPISODES == 1_000
    assert list(MEASUREMENT_SWEEP_SEEDS) == list(range(31_000, 32_000))


# ---------------------------------------------------------------------------
# E. No overlap with prior ranges or future ranges
# ---------------------------------------------------------------------------
def test_no_overlap_with_prior_and_future_ranges():
    sweep_set = set(MEASUREMENT_SWEEP_SEEDS)
    assert sweep_set.isdisjoint(set(FINAL_NOMINAL_SEEDS))
    assert sweep_set.isdisjoint(set(ACQUISITION_ABLATION_SEEDS))
    assert sweep_set.isdisjoint(set(DETECTION_SWEEP_SEEDS))
    assert MOBILITY_SWEEP_SEED_OFFSET == 32_000
    assert max(MEASUREMENT_SWEEP_SEEDS) < MOBILITY_SWEEP_SEED_OFFSET
    assert sweep_set.isdisjoint(set(RESERVED_FUTURE_ROBUSTNESS_SEEDS))

    assert_measurement_sweep_seed(31_000)
    assert_measurement_sweep_seed(31_999)
    for bad_seed in (24_999, 20_000, 25_500, 30_999, 32_000):
        raised = False
        try:
            assert_measurement_sweep_seed(bad_seed)
        except ValueError:
            raised = True
        assert raised, f"assert_measurement_sweep_seed must reject seed {bad_seed}"


# ---------------------------------------------------------------------------
# F. Exactly six primary paper methods
# ---------------------------------------------------------------------------
def test_six_primary_methods_only():
    assert PRIMARY_PAPER_METHODS == ("greedy", "kalman_greedy", "pn", "lead", "ppo", "ppo_kalman")
    assert len(PRIMARY_PAPER_METHODS) == 6
    assert set(SCRIPTED_METHODS) == {"greedy", "kalman_greedy", "pn", "lead"}
    assert set(PPO_METHODS) == {"ppo", "ppo_kalman"}


# ---------------------------------------------------------------------------
# G. 10 policy instances ; H. PN-Kalman/Lead-Kalman absent
# ---------------------------------------------------------------------------
def test_10_policy_instances_exist():
    assert len(NOISE_POLICY_INSTANCES) == EXPECTED_N_POLICY_INSTANCES == 10
    ppo_instances = [p for p in NOISE_POLICY_INSTANCES if p.model_path is not None]
    scripted_instances = [p for p in NOISE_POLICY_INSTANCES if p.model_path is None]
    assert len(ppo_instances) == 6
    assert len(scripted_instances) == 4


def test_pn_kalman_lead_kalman_absent():
    methods_present = {p.paper_method for p in NOISE_POLICY_INSTANCES}
    assert "pn_kalman" not in methods_present
    assert "lead_kalman" not in methods_present
    assert methods_present == set(PRIMARY_PAPER_METHODS)


# ---------------------------------------------------------------------------
# I. All six PPO hashes match frozen final-nominal hashes
# ---------------------------------------------------------------------------
def test_ppo_hashes_match_final_nominal():
    if not FINAL_NOMINAL_HASHES_PATH.exists():
        print("  (final_nominal/model_hashes.json not found; skipping)")
        return
    nominal_by_path = {
        row["model_path"].replace("\\", "/"): row["sha256"]
        for row in json.loads(FINAL_NOMINAL_HASHES_PATH.read_text())
    }
    checked = 0
    for instance in NOISE_POLICY_INSTANCES:
        if instance.model_path is None:
            continue
        rel = str(instance.model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        assert rel in nominal_by_path, f"{rel} missing from nominal manifest"
        digest = sha256_of_file(instance.model_path)
        assert digest == nominal_by_path[rel], f"hash mismatch for {rel}"
        checked += 1
    assert checked == 6


# ---------------------------------------------------------------------------
# J. Greedy policy instance uses the corrected measurement contract
# ---------------------------------------------------------------------------
def test_greedy_uses_corrected_contract():
    policy = get_policy("greedy")
    assert isinstance(policy, GreedyInterceptPolicy)
    # Pre-detection: escorts soldier, ignores any stray e_hat/enemy_measurement.
    info_pre = {"enemy_detected": False, "soldier_pos": np.array([10.0, 0.0, 0.0]),
                "defender_pos": np.array([0.0, 0.0, 0.0]), "defender_vel": np.array([0.0, 0.0, 0.0]),
                "e_hat": np.array([-99.0, -99.0, -99.0])}
    action_pre = policy.act(np.zeros(16, dtype=np.float32), info_pre)
    assert action_pre[0] > 0, "pre-detection must move toward soldier (+x), not toward stray e_hat"

    # Post-detection: pursues enemy_measurement, NOT silently continuing to escort.
    info_post = {"enemy_detected": True, "soldier_pos": np.array([10.0, 0.0, 0.0]),
                 "defender_pos": np.array([0.0, 0.0, 0.0]), "defender_vel": np.array([0.0, 0.0, 0.0]),
                 "enemy_measurement": np.array([-10.0, 0.0, 0.0])}
    action_post = policy.act(np.zeros(16, dtype=np.float32), info_post)
    assert action_post[0] < 0, "post-detection must pursue enemy_measurement (-x), not keep escorting (+x)"

    # Detected but neither field present -> must raise, not silently escort.
    info_bad = {"enemy_detected": True, "soldier_pos": np.array([10.0, 0.0, 0.0]),
                "defender_pos": np.array([0.0, 0.0, 0.0]), "defender_vel": np.array([0.0, 0.0, 0.0])}
    raised = False
    try:
        policy.act(np.zeros(16, dtype=np.float32), info_bad)
    except ValueError:
        raised = True
    assert raised, "Greedy must raise, not silently escort, when detected but no estimate is present"


# ---------------------------------------------------------------------------
# K. Sweep config differs only in measurement_var/estimator ; L. process_var/lead_time fixed
# ---------------------------------------------------------------------------
def test_sweep_config_differs_only_in_measurement_var():
    for variance in MEASUREMENT_VAR_VALUES:
        for estimator in ("measurement", "kalman"):
            config = build_sweep_env_config(estimator, variance)
            assert_only_measurement_var_differs(config, _NOMINAL_ENV_CONFIG)
            assert config.process_var == FIXED_PROCESS_VAR == 1.0
            assert config.lead_time == FIXED_LEAD_TIME == 0.0

    # At nominal variance + measurement estimator, config must be IDENTICAL to nominal.
    nominal_like = build_sweep_env_config("measurement", NOMINAL_MEASUREMENT_VAR)
    assert dataclasses.asdict(nominal_like) == dataclasses.asdict(_NOMINAL_ENV_CONFIG)

    # A genuinely different (unrelated) field must be caught.
    bad_config = EnvConfig(use_kalman_tracking=False, measurement_var=0.5, v_d=99.0)
    raised = False
    try:
        assert_only_measurement_var_differs(bad_config, _NOMINAL_ENV_CONFIG)
    except AssertionError:
        raised = True
    assert raised, "assert_only_measurement_var_differs must catch an unrelated field change"

    # process_var drift must ALSO be caught, even if it's in allowed_diff_keys-adjacent territory.
    bad_process_var = EnvConfig(use_kalman_tracking=True, measurement_var=0.5, process_var=2.0)
    raised = False
    try:
        assert_only_measurement_var_differs(bad_process_var, _NOMINAL_ENV_CONFIG)
    except AssertionError:
        raised = True
    assert raised, "assert_only_measurement_var_differs must catch process_var drift"


# ---------------------------------------------------------------------------
# M. Kalman filter's R equals measurement_var * I
# ---------------------------------------------------------------------------
def test_kalman_R_equals_measurement_var_times_identity():
    for variance in MEASUREMENT_VAR_VALUES:
        kf = EnemyKalmanFilter(dt=0.5, process_var=FIXED_PROCESS_VAR, measurement_var=variance)
        expected_R = np.eye(kf.R.shape[0]) * variance
        assert np.allclose(kf.R, expected_R), (variance, kf.R)


# ---------------------------------------------------------------------------
# N. Empirical sensor noise std matches sqrt(measurement_var)
# ---------------------------------------------------------------------------
def test_empirical_sensor_noise_std():
    variance = 2.0
    expected_std = measurement_std(variance)
    config = build_sweep_env_config("measurement", variance)
    env = SoldierEnv(config=config)
    policy = get_policy("greedy")
    errors = []
    for s in range(50):
        obs, info = env.reset(seed=100_000 + s)
        policy.reset()
        mode = estimator_mode_for_env(env)
        done = False
        while not done:
            sanitized = build_policy_info(info, mode)
            action = policy.act(obs, sanitized)
            obs, reward, done, truncated, info = env.step(action)
            if info["enemy_detected"]:
                measurement = info["enemy_measurement"]
                if measurement is not None:
                    errors.append(np.asarray(measurement) - np.asarray(info["enemy_pos"]))
            if done or truncated:
                break
    env.close()
    errors = np.asarray(errors)
    assert errors.shape[0] > 100, "not enough post-detection samples collected"
    empirical_std = float(np.std(errors))
    assert abs(empirical_std - expected_std) / expected_std < 0.15, (empirical_std, expected_std)


# ---------------------------------------------------------------------------
# O. Expected raw row count = 60000
# ---------------------------------------------------------------------------
def test_expected_raw_row_count():
    assert EXPECTED_RAW_ROW_COUNT == 10 * 6 * 1000 == 60_000


# ---------------------------------------------------------------------------
# P. Policy-info sanitization still excludes enemy_pos/enemy_vel
# ---------------------------------------------------------------------------
def test_sanitization_excludes_ground_truth():
    fake_info = {
        "defender_pos": [0.0, 0.0, 5.0], "defender_vel": [1.0, 0.0, 0.0],
        "soldier_pos": [-10.0, 0.0, 0.0], "enemy_detected": True,
        "enemy_measurement": [5.0, 0.0, 0.0], "enemy_measurement_velocity": [0.0, 0.0, 0.0],
        "enemy_measurement_velocity_valid": True,
        "e_hat": [5.0, 0.0, 0.0], "v_hat": [0.0, 0.0, 0.0],
        "enemy_pos": [5.0, 0.0, 0.0], "enemy_vel": [0.0, 0.0, 0.0],
    }
    for mode in ("measurement", "kalman"):
        sanitized = build_policy_info(fake_info, mode)
        assert "enemy_pos" not in sanitized
        assert "enemy_vel" not in sanitized


# ---------------------------------------------------------------------------
# Q. Common seed set across all variance/method combinations (construction-level)
# ---------------------------------------------------------------------------
def test_common_seed_set_construction():
    seeds = list(MEASUREMENT_SWEEP_SEEDS)
    for variance in MEASUREMENT_VAR_VALUES:
        for _ in NOISE_POLICY_INSTANCES:
            assert seeds == list(MEASUREMENT_SWEEP_SEEDS)
    assert len(set(seeds)) == 1000


# ---------------------------------------------------------------------------
# R. RMSE sanity-check helper equals sqrt(3 * measurement_var)
# ---------------------------------------------------------------------------
def test_theoretical_rmse_helper():
    for v in MEASUREMENT_VAR_VALUES:
        assert abs(theoretical_raw_measurement_rmse(v) - np.sqrt(3.0 * v)) < 1e-12
    assert abs(theoretical_raw_measurement_rmse(0.5) - np.sqrt(1.5)) < 1e-9


# ---------------------------------------------------------------------------
# S. Bootstrap sampling pairs evaluation seeds/training-seed labels correctly
# ---------------------------------------------------------------------------
def test_bootstrap_pairs_seeds_correctly():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=(3, 100)).astype(np.float64)
    b = rng.integers(0, 2, size=(3, 100)).astype(np.float64)
    observed, lo, hi, per_seed_diffs = hierarchical_paired_bootstrap_matched_seeds(a, b, 300, 42)
    assert per_seed_diffs.shape == (3,)
    assert np.allclose(per_seed_diffs, a.mean(axis=1) - b.mean(axis=1))


# ---------------------------------------------------------------------------
# T. RNG fairness: true trajectories are unaffected by measurement_var
# ---------------------------------------------------------------------------
def test_rng_fairness_across_measurement_var():
    seed = 55_000
    n_steps = 40
    fixed_action = np.zeros(4, dtype=np.float32)  # deterministic action sequence

    def rollout(variance):
        config = build_sweep_env_config("measurement", variance)
        env = SoldierEnv(config=config)
        obs, info = env.reset(seed=seed)
        soldier_traj = [np.asarray(info["soldier_pos"]).copy()]
        enemy_traj = [np.asarray(info["enemy_pos"]).copy()]
        defender_traj = [np.asarray(info["defender_pos"]).copy()]
        for _ in range(n_steps):
            obs, reward, done, truncated, info = env.step(fixed_action)
            soldier_traj.append(np.asarray(info["soldier_pos"]).copy())
            enemy_traj.append(np.asarray(info["enemy_pos"]).copy())
            defender_traj.append(np.asarray(info["defender_pos"]).copy())
            if done or truncated:
                break
        env.close()
        return np.array(soldier_traj), np.array(enemy_traj), np.array(defender_traj)

    ref_soldier, ref_enemy, ref_defender = rollout(MEASUREMENT_VAR_VALUES[0])
    for variance in MEASUREMENT_VAR_VALUES[1:]:
        soldier, enemy, defender = rollout(variance)
        assert soldier.shape == ref_soldier.shape
        assert enemy.shape == ref_enemy.shape
        # Defender trajectory is action-driven (fixed action) so it too must be
        # identical regardless of measurement_var, since the fixed action is
        # not derived from any policy/estimate here.
        assert np.allclose(soldier, ref_soldier, atol=1e-9), variance
        assert np.allclose(enemy, ref_enemy, atol=1e-9), variance
        assert np.allclose(defender, ref_defender, atol=1e-9), variance


def main() -> int:
    tests = [
        ("A. Variance grid exact", test_variance_values_exact),
        ("B. Nominal 0.5 included", test_nominal_variance_included),
        ("C. measurement_std helper correct", test_measurement_std_helper),
        ("D. Sweep seeds exactly 31000..31999", test_sweep_seed_range),
        ("E. No overlap with prior/future ranges", test_no_overlap_with_prior_and_future_ranges),
        ("F. Six primary methods only", test_six_primary_methods_only),
        ("G. 10 policy instances exist", test_10_policy_instances_exist),
        ("H. PN-Kalman/Lead-Kalman absent", test_pn_kalman_lead_kalman_absent),
        ("I. PPO hashes match final-nominal hashes", test_ppo_hashes_match_final_nominal),
        ("J. Greedy uses corrected contract", test_greedy_uses_corrected_contract),
        ("K/L. Sweep config differs only in measurement_var; process_var/lead_time fixed",
         test_sweep_config_differs_only_in_measurement_var),
        ("M. Kalman R equals measurement_var * I", test_kalman_R_equals_measurement_var_times_identity),
        ("N. Empirical sensor noise std matches sqrt(measurement_var)", test_empirical_sensor_noise_std),
        ("O. Expected raw row count = 60000", test_expected_raw_row_count),
        ("P. Sanitization excludes ground truth", test_sanitization_excludes_ground_truth),
        ("Q. Common seed set construction", test_common_seed_set_construction),
        ("R. Theoretical RMSE helper = sqrt(3*variance)", test_theoretical_rmse_helper),
        ("S. Bootstrap pairs seeds correctly", test_bootstrap_pairs_seeds_correctly),
        ("T. RNG fairness across measurement_var", test_rng_fairness_across_measurement_var),
    ]

    print("=== Measurement-Noise Robustness Sweep Tests ===\n")
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
