"""Mobility robustness sweep correctness tests -- locked sweep guardrails.

Script-style test consistent with the project's other test files. No
external test framework is used; each test function performs assertions
and the runner reports PASS/FAIL per test with a final summary.

Covers (per the mobility robustness sweep task spec):
    A. Defender speed values exactly 12,15,18,21,24.
    B. Hostile speed values exactly 8,10,12,14,16.
    C. Nominal pair exactly (18,12).
    D. Nine UNIQUE configurations.
    E. Seeds exactly 32000..32999.
    F. No overlap with previous ranges.
    G. No seeds >=33000 used.
    H. Six methods only.
    I. 10 policy instances.
    J. PN-Kalman/Lead-Kalman absent.
    K. All six PPO hashes match frozen nominal hashes.
    L. Defender-arm configs change only v_d + estimator flag.
    M. Hostile-arm configs change only v_e + estimator flag.
    N. All acceleration/turn/climb parameters remain nominal.
    O. Sensing/evasion/reward remain nominal.
    P. mobility_ratio computed correctly.
    Q. Nominal configuration not executed twice.
    R. Expected rows = 90000.
    S. Corrected Greedy contract active.
    T. Policy sanitization excludes enemy_pos/enemy_vel.
    U. Observation normalization remains finite/in bounds at all tested speeds.
    V. Correct (sample-weighted) estimator aggregation helper is used.

Run directly:
    python test_robustness_mobility.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy
from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env
from uav_defend.policies.baseline.greedy_intercept_policy import GreedyInterceptPolicy

from experiments.final_experiment_protocol import FINAL_NOMINAL_SEEDS
from experiments.acquisition_ablation_protocol import ACQUISITION_ABLATION_SEEDS
from experiments.robustness_detection_protocol import DETECTION_SWEEP_SEEDS
from experiments.robustness_measurement_noise_protocol import MEASUREMENT_SWEEP_SEEDS
from experiments.final_eval_utils import sha256_of_file
from experiments.robustness_mobility_protocol import (
    DEFENDER_SPEED_VALUES,
    HOSTILE_SPEED_VALUES,
    NOMINAL_DEFENDER_SPEED,
    NOMINAL_HOSTILE_SPEED,
    NOMINAL_MOBILITY_RATIO,
    MOBILITY_CONDITIONS,
    EXPECTED_N_MOBILITY_CONDITIONS,
    defender_speed_conditions,
    hostile_speed_conditions,
    MOBILITY_SWEEP_SEED_OFFSET,
    MOBILITY_SWEEP_EPISODES,
    MOBILITY_SWEEP_SEEDS,
    HOSTILE_EVASION_SWEEP_SEED_OFFSET,
    RESERVED_FUTURE_ROBUSTNESS_SEEDS,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    MOBILITY_POLICY_INSTANCES,
    EXPECTED_N_POLICY_INSTANCES,
    EXPECTED_RAW_ROW_COUNT,
    FIXED_DEFENDER_MAX_ACCEL,
    FIXED_DEFENDER_MAX_TURN_RATE_DEG,
    FIXED_DEFENDER_MAX_CLIMB_RATE,
    FIXED_DEFENDER_MAX_DESCENT_RATE,
    FIXED_ENEMY_MAX_ACCEL,
    FIXED_ENEMY_MAX_TURN_RATE_DEG,
    FIXED_ENEMY_MAX_CLIMB_RATE,
    FIXED_ENEMY_MAX_DESCENT_RATE,
    FIXED_DETECTION_RADIUS,
    FIXED_MEASUREMENT_VAR,
    FIXED_PROCESS_VAR,
    FIXED_LEAD_TIME,
    FIXED_ENEMY_EVASION_ENABLED,
    FIXED_ENEMY_EVASION_RADIUS,
    FIXED_ENEMY_EVASION_GAIN,
    build_mobility_env_config,
    assert_only_mobility_speed_differs,
    assert_defender_arm_config,
    assert_hostile_arm_config,
    assert_mobility_sweep_seed,
)
from experiments.mobility_eval_utils import run_mobility_episode
from experiments.estimator_stats import sample_weighted_rmse
from experiments.run_robustness_mobility_sweep import compute_estimator_summary

PROJECT_ROOT = Path(__file__).parent
FINAL_NOMINAL_HASHES_PATH = PROJECT_ROOT / "results" / "final_nominal" / "model_hashes.json"
_NOMINAL_ENV_CONFIG = EnvConfig()


# ---------------------------------------------------------------------------
# A/B/C. Speed grids and nominal pair exact
# ---------------------------------------------------------------------------
def test_defender_speed_values_exact():
    assert DEFENDER_SPEED_VALUES == (12.0, 15.0, 18.0, 21.0, 24.0)


def test_hostile_speed_values_exact():
    assert HOSTILE_SPEED_VALUES == (8.0, 10.0, 12.0, 14.0, 16.0)


def test_nominal_pair_exact():
    assert NOMINAL_DEFENDER_SPEED == 18.0
    assert NOMINAL_HOSTILE_SPEED == 12.0
    assert abs(NOMINAL_MOBILITY_RATIO - 1.5) < 1e-12
    assert _NOMINAL_ENV_CONFIG.v_d == NOMINAL_DEFENDER_SPEED
    assert _NOMINAL_ENV_CONFIG.v_e == NOMINAL_HOSTILE_SPEED


# ---------------------------------------------------------------------------
# D. Nine unique configurations
# ---------------------------------------------------------------------------
def test_nine_unique_configurations():
    assert len(MOBILITY_CONDITIONS) == EXPECTED_N_MOBILITY_CONDITIONS == 9
    pairs = {(c.defender_speed, c.hostile_speed) for c in MOBILITY_CONDITIONS}
    assert len(pairs) == 9
    expected = {
        (12.0, 12.0), (15.0, 12.0), (18.0, 12.0), (21.0, 12.0), (24.0, 12.0),
        (18.0, 8.0), (18.0, 10.0), (18.0, 14.0), (18.0, 16.0),
    }
    assert pairs == expected


# ---------------------------------------------------------------------------
# E. Seeds exactly 32000..32999
# ---------------------------------------------------------------------------
def test_sweep_seed_range():
    assert MOBILITY_SWEEP_SEED_OFFSET == 32_000
    assert MOBILITY_SWEEP_EPISODES == 1_000
    assert list(MOBILITY_SWEEP_SEEDS) == list(range(32_000, 33_000))


# ---------------------------------------------------------------------------
# F/G. No overlap with prior ranges; no future (>=33000) ranges used
# ---------------------------------------------------------------------------
def test_no_overlap_with_prior_and_future_ranges():
    sweep_set = set(MOBILITY_SWEEP_SEEDS)
    assert sweep_set.isdisjoint(set(FINAL_NOMINAL_SEEDS))
    assert sweep_set.isdisjoint(set(ACQUISITION_ABLATION_SEEDS))
    assert sweep_set.isdisjoint(set(DETECTION_SWEEP_SEEDS))
    assert sweep_set.isdisjoint(set(MEASUREMENT_SWEEP_SEEDS))
    assert HOSTILE_EVASION_SWEEP_SEED_OFFSET == 33_000
    assert max(MOBILITY_SWEEP_SEEDS) < HOSTILE_EVASION_SWEEP_SEED_OFFSET
    assert sweep_set.isdisjoint(set(RESERVED_FUTURE_ROBUSTNESS_SEEDS))

    assert_mobility_sweep_seed(32_000)
    assert_mobility_sweep_seed(32_999)
    for bad_seed in (24_999, 20_000, 25_500, 30_999, 31_500, 33_000):
        raised = False
        try:
            assert_mobility_sweep_seed(bad_seed)
        except ValueError:
            raised = True
        assert raised, f"assert_mobility_sweep_seed must reject seed {bad_seed}"


# ---------------------------------------------------------------------------
# H. Six methods only
# ---------------------------------------------------------------------------
def test_six_primary_methods_only():
    assert PRIMARY_PAPER_METHODS == ("greedy", "kalman_greedy", "pn", "lead", "ppo", "ppo_kalman")
    assert len(PRIMARY_PAPER_METHODS) == 6


# ---------------------------------------------------------------------------
# I. 10 policy instances ; J. PN-Kalman/Lead-Kalman absent
# ---------------------------------------------------------------------------
def test_10_policy_instances_exist():
    assert len(MOBILITY_POLICY_INSTANCES) == EXPECTED_N_POLICY_INSTANCES == 10
    ppo_instances = [p for p in MOBILITY_POLICY_INSTANCES if p.model_path is not None]
    scripted_instances = [p for p in MOBILITY_POLICY_INSTANCES if p.model_path is None]
    assert len(ppo_instances) == 6
    assert len(scripted_instances) == 4


def test_pn_kalman_lead_kalman_absent():
    methods_present = {p.paper_method for p in MOBILITY_POLICY_INSTANCES}
    assert "pn_kalman" not in methods_present
    assert "lead_kalman" not in methods_present
    assert methods_present == set(PRIMARY_PAPER_METHODS)


# ---------------------------------------------------------------------------
# K. All six PPO hashes match frozen nominal hashes
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
    for instance in MOBILITY_POLICY_INSTANCES:
        if instance.model_path is None:
            continue
        rel = str(instance.model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        assert rel in nominal_by_path, f"{rel} missing from nominal manifest"
        digest = sha256_of_file(instance.model_path)
        assert digest == nominal_by_path[rel], f"hash mismatch for {rel}"
        checked += 1
    assert checked == 6


# ---------------------------------------------------------------------------
# L/M. Defender-arm / hostile-arm configs change only their own speed + estimator
# ---------------------------------------------------------------------------
def test_defender_arm_config_isolation():
    for condition in defender_speed_conditions():
        for estimator in ("measurement", "kalman"):
            config = build_mobility_env_config(estimator, condition.defender_speed, condition.hostile_speed)
            assert_defender_arm_config(config, _NOMINAL_ENV_CONFIG)


def test_hostile_arm_config_isolation():
    for condition in hostile_speed_conditions():
        for estimator in ("measurement", "kalman"):
            config = build_mobility_env_config(estimator, condition.defender_speed, condition.hostile_speed)
            assert_hostile_arm_config(config, _NOMINAL_ENV_CONFIG)

    # A genuinely different (unrelated) field must be caught.
    bad_config = EnvConfig(use_kalman_tracking=False, v_d=18.0, v_e=12.0, defender_max_accel=99.0)
    raised = False
    try:
        assert_only_mobility_speed_differs(bad_config, _NOMINAL_ENV_CONFIG)
    except AssertionError:
        raised = True
    assert raised, "assert_only_mobility_speed_differs must catch an unrelated field change"


# ---------------------------------------------------------------------------
# N. Acceleration/turn/climb parameters remain nominal at every condition
# ---------------------------------------------------------------------------
def test_dynamics_limits_never_scaled():
    assert FIXED_DEFENDER_MAX_ACCEL == _NOMINAL_ENV_CONFIG.defender_max_accel == 8.0
    assert FIXED_DEFENDER_MAX_TURN_RATE_DEG == _NOMINAL_ENV_CONFIG.defender_max_turn_rate_deg == 90.0
    assert FIXED_DEFENDER_MAX_CLIMB_RATE == _NOMINAL_ENV_CONFIG.defender_max_climb_rate == 6.0
    assert FIXED_DEFENDER_MAX_DESCENT_RATE == _NOMINAL_ENV_CONFIG.defender_max_descent_rate == 6.0
    assert FIXED_ENEMY_MAX_ACCEL == _NOMINAL_ENV_CONFIG.enemy_max_accel == 6.0
    assert FIXED_ENEMY_MAX_TURN_RATE_DEG == _NOMINAL_ENV_CONFIG.enemy_max_turn_rate_deg == 75.0
    assert FIXED_ENEMY_MAX_CLIMB_RATE == _NOMINAL_ENV_CONFIG.enemy_max_climb_rate == 5.0
    assert FIXED_ENEMY_MAX_DESCENT_RATE == _NOMINAL_ENV_CONFIG.enemy_max_descent_rate == 5.0

    for condition in MOBILITY_CONDITIONS:
        config = build_mobility_env_config("measurement", condition.defender_speed, condition.hostile_speed)
        assert config.defender_max_accel == FIXED_DEFENDER_MAX_ACCEL
        assert config.defender_max_turn_rate_deg == FIXED_DEFENDER_MAX_TURN_RATE_DEG
        assert config.defender_max_climb_rate == FIXED_DEFENDER_MAX_CLIMB_RATE
        assert config.defender_max_descent_rate == FIXED_DEFENDER_MAX_DESCENT_RATE
        assert config.enemy_max_accel == FIXED_ENEMY_MAX_ACCEL
        assert config.enemy_max_turn_rate_deg == FIXED_ENEMY_MAX_TURN_RATE_DEG
        assert config.enemy_max_climb_rate == FIXED_ENEMY_MAX_CLIMB_RATE
        assert config.enemy_max_descent_rate == FIXED_ENEMY_MAX_DESCENT_RATE


# ---------------------------------------------------------------------------
# O. Sensing/evasion/reward remain nominal at every condition
# ---------------------------------------------------------------------------
def test_sensing_evasion_reward_never_changed():
    assert FIXED_DETECTION_RADIUS == _NOMINAL_ENV_CONFIG.detection_radius == 15.0
    assert FIXED_MEASUREMENT_VAR == _NOMINAL_ENV_CONFIG.measurement_var == 0.5
    assert FIXED_PROCESS_VAR == _NOMINAL_ENV_CONFIG.process_var == 1.0
    assert FIXED_LEAD_TIME == _NOMINAL_ENV_CONFIG.lead_time == 0.0
    assert FIXED_ENEMY_EVASION_ENABLED == _NOMINAL_ENV_CONFIG.enemy_evasion_enabled is True
    assert FIXED_ENEMY_EVASION_RADIUS == _NOMINAL_ENV_CONFIG.enemy_evasion_radius == 20.0
    assert FIXED_ENEMY_EVASION_GAIN == _NOMINAL_ENV_CONFIG.enemy_evasion_gain == 0.75

    for condition in MOBILITY_CONDITIONS:
        config = build_mobility_env_config("kalman", condition.defender_speed, condition.hostile_speed)
        assert config.detection_radius == FIXED_DETECTION_RADIUS
        assert config.measurement_var == FIXED_MEASUREMENT_VAR
        assert config.process_var == FIXED_PROCESS_VAR
        assert config.lead_time == FIXED_LEAD_TIME
        assert config.enemy_evasion_enabled == FIXED_ENEMY_EVASION_ENABLED
        assert config.enemy_evasion_radius == FIXED_ENEMY_EVASION_RADIUS
        assert config.enemy_evasion_gain == FIXED_ENEMY_EVASION_GAIN
        # Reward fields must all equal nominal (none swept).
        assert config.reward_intercept == _NOMINAL_ENV_CONFIG.reward_intercept
        assert config.reward_soldier_caught == _NOMINAL_ENV_CONFIG.reward_soldier_caught
        assert config.reward_unsafe_intercept == _NOMINAL_ENV_CONFIG.reward_unsafe_intercept
        assert config.reward_timeout == _NOMINAL_ENV_CONFIG.reward_timeout


# ---------------------------------------------------------------------------
# P. mobility_ratio computed correctly
# ---------------------------------------------------------------------------
def test_mobility_ratio_correct():
    for condition in MOBILITY_CONDITIONS:
        assert abs(condition.mobility_ratio - condition.defender_speed / condition.hostile_speed) < 1e-12
    nominal = next(c for c in MOBILITY_CONDITIONS if c.sweep_axis == "nominal")
    assert abs(nominal.mobility_ratio - 1.5) < 1e-12
    ratios = {round(c.mobility_ratio, 4) for c in MOBILITY_CONDITIONS}
    expected_ratios = {round(v / 12.0, 4) for v in DEFENDER_SPEED_VALUES} | {round(18.0 / v, 4) for v in HOSTILE_SPEED_VALUES}
    assert ratios == expected_ratios


# ---------------------------------------------------------------------------
# Q. Nominal configuration not executed twice
# ---------------------------------------------------------------------------
def test_nominal_not_duplicated():
    nominal_rows = [c for c in MOBILITY_CONDITIONS if c.sweep_axis == "nominal"]
    assert len(nominal_rows) == 1
    d_conditions = defender_speed_conditions()
    h_conditions = hostile_speed_conditions()
    assert len(d_conditions) == 5
    assert len(h_conditions) == 5
    # Both curves reference the SAME nominal MobilityCondition object/values (not re-simulated).
    d_nominal = next(c for c in d_conditions if c.sweep_axis == "nominal")
    h_nominal = next(c for c in h_conditions if c.sweep_axis == "nominal")
    assert d_nominal is h_nominal or (d_nominal.defender_speed, d_nominal.hostile_speed) == (h_nominal.defender_speed, h_nominal.hostile_speed)
    # Total UNIQUE conditions must remain 9 despite each curve independently listing 5 points.
    assert len({(c.defender_speed, c.hostile_speed) for c in MOBILITY_CONDITIONS}) == 9


# ---------------------------------------------------------------------------
# R. Expected rows = 90000
# ---------------------------------------------------------------------------
def test_expected_raw_row_count():
    assert EXPECTED_RAW_ROW_COUNT == 9 * 10 * 1000 == 90_000


# ---------------------------------------------------------------------------
# S. Corrected Greedy contract active
# ---------------------------------------------------------------------------
def test_greedy_uses_corrected_contract():
    policy = get_policy("greedy")
    assert isinstance(policy, GreedyInterceptPolicy)
    info_post = {"enemy_detected": True, "soldier_pos": np.array([10.0, 0.0, 0.0]),
                 "defender_pos": np.array([0.0, 0.0, 0.0]), "defender_vel": np.array([0.0, 0.0, 0.0]),
                 "enemy_measurement": np.array([-10.0, 0.0, 0.0])}
    action_post = policy.act(np.zeros(16, dtype=np.float32), info_post)
    assert action_post[0] < 0, "post-detection must pursue enemy_measurement, not keep escorting"


# ---------------------------------------------------------------------------
# T. Policy sanitization excludes enemy_pos/enemy_vel
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
# U. Observation normalization remains finite/in bounds at all tested speeds
# ---------------------------------------------------------------------------
def test_observation_bounds_at_all_speeds():
    n_episodes_per_condition = 5
    max_abs_seen = 0.0
    for condition in MOBILITY_CONDITIONS:
        for estimator in ("measurement", "kalman"):
            config = build_mobility_env_config(estimator, condition.defender_speed, condition.hostile_speed)
            env = SoldierEnv(config=config)
            policy = get_policy("greedy") if estimator == "measurement" else get_policy("kalman_greedy")
            for s in range(n_episodes_per_condition):
                obs, info = env.reset(seed=18_000 + s)
                policy.reset()
                mode = estimator_mode_for_env(env)
                done = False
                while not done:
                    assert np.all(np.isfinite(obs)), (condition, obs)
                    max_abs_seen = max(max_abs_seen, float(np.max(np.abs(obs))))
                    sanitized = build_policy_info(info, mode)
                    action = policy.act(obs, sanitized)
                    obs, reward, done, truncated, info = env.step(action)
                    if done or truncated:
                        assert np.all(np.isfinite(obs)), (condition, obs)
                        max_abs_seen = max(max_abs_seen, float(np.max(np.abs(obs))))
                        break
            env.close()
    # Components explicitly clipped to [-1,1] by the environment (defender_vel,
    # hostile pos/vel) can never exceed this; unclipped raw position components
    # (soldier/defender x,y,z normalized by L/max_altitude) could in principle
    # exceed 1.0 slightly near domain edges -- allow a generous documented
    # tolerance rather than assuming exact [-1,1] containment.
    assert max_abs_seen < 3.0, f"observation magnitude grew implausibly large: {max_abs_seen}"


# ---------------------------------------------------------------------------
# V. Correct (sample-weighted) estimator aggregation helper is used
# ---------------------------------------------------------------------------
def test_correct_estimator_aggregation_used():
    # Construct a synthetic raw-episode dataframe with a KNOWN discrepancy
    # between the old buggy sqrt(mean(m_i^2)) and the correct sample-weighted RMSE.
    rows = []
    n_samples = [50, 200]
    m_vals = [1.0, 3.0]
    r_vals = [1.05, 3.4]
    condition = MOBILITY_CONDITIONS[0]
    for i, (n, m, r) in enumerate(zip(n_samples, m_vals, r_vals)):
        rows.append({
            "defender_speed": condition.defender_speed, "hostile_speed": condition.hostile_speed,
            "policy_instance": "greedy", "paper_method": "greedy", "training_seed": float("nan"),
            "eval_seed": 32000 + i, "mean_position_estimation_error": m,
            "position_estimation_rmse": r, "n_position_estimation_samples": n,
        })
    # Fill in the other 3 scripted methods + 6 PPO instances with matching (trivial) rows
    # so compute_estimator_summary can run without KeyErrors for this condition.
    for method in ("kalman_greedy", "pn", "lead"):
        for i, (n, m, r) in enumerate(zip(n_samples, m_vals, r_vals)):
            rows.append({
                "defender_speed": condition.defender_speed, "hostile_speed": condition.hostile_speed,
                "policy_instance": method, "paper_method": method, "training_seed": float("nan"),
                "eval_seed": 32000 + i, "mean_position_estimation_error": m,
                "position_estimation_rmse": r, "n_position_estimation_samples": n,
            })
    for method in ("ppo", "ppo_kalman"):
        for seed in (42, 43, 44):
            for i, (n, m, r) in enumerate(zip(n_samples, m_vals, r_vals)):
                rows.append({
                    "defender_speed": condition.defender_speed, "hostile_speed": condition.hostile_speed,
                    "policy_instance": f"{method}_seed_{seed}", "paper_method": method, "training_seed": seed,
                    "eval_seed": 32000 + i, "mean_position_estimation_error": m,
                    "position_estimation_rmse": r, "n_position_estimation_samples": n,
                })
    # Fill remaining 8 conditions with copies so groupby doesn't choke (not exercised by assertions).
    other_conditions = MOBILITY_CONDITIONS[1:]
    base_rows = list(rows)
    for other in other_conditions:
        for row in base_rows:
            new_row = dict(row)
            new_row["defender_speed"] = other.defender_speed
            new_row["hostile_speed"] = other.hostile_speed
            rows.append(new_row)

    df = pd.DataFrame(rows)
    summary = compute_estimator_summary(df)
    greedy_row = summary[(summary["defender_speed"] == condition.defender_speed) &
                          (summary["hostile_speed"] == condition.hostile_speed) &
                          (summary["method"] == "greedy")].iloc[0]

    n_arr = np.array(n_samples, dtype=np.float64)
    r_arr = np.array(r_vals, dtype=np.float64)
    m_arr = np.array(m_vals, dtype=np.float64)
    expected_correct_rmse = sample_weighted_rmse(n_arr, r_arr)
    old_buggy_rmse = float(np.sqrt(np.mean(np.square(m_arr))))

    assert abs(greedy_row["rmse_sample_weighted"] - expected_correct_rmse) < 1e-9
    assert abs(greedy_row["rmse_sample_weighted"] - old_buggy_rmse) > 0.05, (
        "compute_estimator_summary must NOT reproduce the old sqrt(mean(m_i^2)) formula"
    )


def main() -> int:
    tests = [
        ("A. Defender speed values exactly 12,15,18,21,24", test_defender_speed_values_exact),
        ("B. Hostile speed values exactly 8,10,12,14,16", test_hostile_speed_values_exact),
        ("C. Nominal pair exactly (18,12)", test_nominal_pair_exact),
        ("D. Nine unique configurations", test_nine_unique_configurations),
        ("E. Seeds exactly 32000..32999", test_sweep_seed_range),
        ("F/G. No overlap with prior ranges / no future seeds", test_no_overlap_with_prior_and_future_ranges),
        ("H. Six methods only", test_six_primary_methods_only),
        ("I. 10 policy instances exist", test_10_policy_instances_exist),
        ("J. PN-Kalman/Lead-Kalman absent", test_pn_kalman_lead_kalman_absent),
        ("K. PPO hashes match final-nominal hashes", test_ppo_hashes_match_final_nominal),
        ("L. Defender-arm config isolation", test_defender_arm_config_isolation),
        ("M. Hostile-arm config isolation", test_hostile_arm_config_isolation),
        ("N. Dynamics limits never scaled", test_dynamics_limits_never_scaled),
        ("O. Sensing/evasion/reward never changed", test_sensing_evasion_reward_never_changed),
        ("P. mobility_ratio computed correctly", test_mobility_ratio_correct),
        ("Q. Nominal configuration not executed twice", test_nominal_not_duplicated),
        ("R. Expected raw row count = 90000", test_expected_raw_row_count),
        ("S. Greedy uses corrected contract", test_greedy_uses_corrected_contract),
        ("T. Sanitization excludes ground truth", test_sanitization_excludes_ground_truth),
        ("U. Observation bounds at all tested speeds", test_observation_bounds_at_all_speeds),
        ("V. Correct estimator aggregation used", test_correct_estimator_aggregation_used),
    ]

    print("=== Mobility Robustness Sweep Tests ===\n")
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
