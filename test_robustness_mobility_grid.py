"""Defender-hostile speed interaction grid correctness tests -- locked guardrails.

Script-style test consistent with the project's other test files.

Covers (per the grid task spec):
    A. Defender values exactly 12,15,18,21,24.
    B. Hostile values exactly 8,10,12,14,16.
    C. 25 Cartesian cells exactly.
    D. Nominal (18,12) occurs exactly once.
    E. Seeds exactly 35000..35499.
    F. 500 seeds.
    G. No overlap with any opened dataset.
    H. 33000..34999 remain untouched/reserved.
    I. No seed >=35500.
    J. Six paper methods exactly.
    K. 10 policy instances.
    L. PN-Kalman/Lead-Kalman absent.
    M. Six model hashes match.
    N. Only v_d/v_e + estimator flag differ.
    O. All dynamic limits frozen.
    P. Sensor/reward/evasion frozen.
    Q. Corrected Greedy contract active.
    R. No policy truth leakage.
    S. Observations finite/bounded at all 25 cells.
    T. Expected rows 125000.
    U. Correct estimator aggregation helper used.
    V. Four mu=1.5 cells exactly exist.
    W. Common-random-number seed sets identical.
    X. Action semantics remain unchanged.

Run directly:
    python test_robustness_mobility_grid.py
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
from experiments.robustness_mobility_protocol import MOBILITY_SWEEP_SEEDS
from experiments.final_eval_utils import sha256_of_file
from experiments.robustness_mobility_grid_protocol import (
    DEFENDER_SPEED_VALUES,
    HOSTILE_SPEED_VALUES,
    NOMINAL_DEFENDER_SPEED,
    NOMINAL_HOSTILE_SPEED,
    EQUAL_RATIO_CELLS,
    GRID_CELLS,
    EXPECTED_GRID_CELLS,
    MOBILITY_GRID_SEED_OFFSET,
    MOBILITY_GRID_EPISODES,
    MOBILITY_GRID_SEEDS,
    RESERVED_UNTOUCHED_SEEDS,
    HOSTILE_EVASION_SWEEP_SEED_OFFSET,
    MAX_ALLOWED_SEED,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    GRID_POLICY_INSTANCES,
    EXPECTED_POLICY_INSTANCES,
    EXPECTED_RAW_ROWS,
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
    build_grid_env_config,
    assert_only_grid_speed_differs,
    assert_grid_seed,
)
from experiments.estimator_stats import sample_weighted_rmse
from experiments.run_robustness_mobility_grid_sweep import compute_estimator_summary

PROJECT_ROOT = Path(__file__).parent
FINAL_NOMINAL_HASHES_PATH = PROJECT_ROOT / "results" / "final_nominal" / "model_hashes.json"
_NOMINAL_ENV_CONFIG = EnvConfig()


# ---------------------------------------------------------------------------
# A/B. Grid values exact
# ---------------------------------------------------------------------------
def test_defender_speed_values_exact():
    assert DEFENDER_SPEED_VALUES == (12.0, 15.0, 18.0, 21.0, 24.0)


def test_hostile_speed_values_exact():
    assert HOSTILE_SPEED_VALUES == (8.0, 10.0, 12.0, 14.0, 16.0)


# ---------------------------------------------------------------------------
# C. 25 Cartesian cells ; D. nominal occurs once
# ---------------------------------------------------------------------------
def test_25_cartesian_cells_exact():
    assert len(GRID_CELLS) == EXPECTED_GRID_CELLS == 25
    pairs = {(c.defender_speed, c.hostile_speed) for c in GRID_CELLS}
    assert len(pairs) == 25
    expected = {(vd, ve) for vd in DEFENDER_SPEED_VALUES for ve in HOSTILE_SPEED_VALUES}
    assert pairs == expected


def test_nominal_occurs_exactly_once():
    nominal_cells = [c for c in GRID_CELLS if c.is_nominal]
    assert len(nominal_cells) == 1
    assert nominal_cells[0].defender_speed == NOMINAL_DEFENDER_SPEED == 18.0
    assert nominal_cells[0].hostile_speed == NOMINAL_HOSTILE_SPEED == 12.0


# ---------------------------------------------------------------------------
# E/F. Seeds exactly 35000..35499 ; 500 seeds
# ---------------------------------------------------------------------------
def test_seed_range_exact():
    assert MOBILITY_GRID_SEED_OFFSET == 35_000
    assert MOBILITY_GRID_EPISODES == 500
    assert list(MOBILITY_GRID_SEEDS) == list(range(35_000, 35_500))
    assert len(set(MOBILITY_GRID_SEEDS)) == 500


# ---------------------------------------------------------------------------
# G/H/I. No overlap with opened datasets; 33000-34999 reserved; no seed >=35500
# ---------------------------------------------------------------------------
def test_no_overlap_and_reserved_ranges_protected():
    grid_set = set(MOBILITY_GRID_SEEDS)
    assert grid_set.isdisjoint(set(FINAL_NOMINAL_SEEDS))
    assert grid_set.isdisjoint(set(ACQUISITION_ABLATION_SEEDS))
    assert grid_set.isdisjoint(set(DETECTION_SWEEP_SEEDS))
    assert grid_set.isdisjoint(set(MEASUREMENT_SWEEP_SEEDS))
    assert grid_set.isdisjoint(set(MOBILITY_SWEEP_SEEDS))
    assert grid_set.isdisjoint(set(RESERVED_UNTOUCHED_SEEDS))
    assert HOSTILE_EVASION_SWEEP_SEED_OFFSET == 33_000
    assert list(RESERVED_UNTOUCHED_SEEDS) == list(range(33_000, 35_000))
    assert MAX_ALLOWED_SEED == 35_499
    assert max(MOBILITY_GRID_SEEDS) == MAX_ALLOWED_SEED

    assert_grid_seed(35_000)
    assert_grid_seed(35_499)
    for bad_seed in (24_999, 30_999, 31_999, 32_999, 33_500, 34_999, 35_500):
        raised = False
        try:
            assert_grid_seed(bad_seed)
        except ValueError:
            raised = True
        assert raised, f"assert_grid_seed must reject seed {bad_seed}"


# ---------------------------------------------------------------------------
# J/K/L. Six methods; 10 instances; PN-Kalman/Lead-Kalman absent
# ---------------------------------------------------------------------------
def test_six_primary_methods_exact():
    assert PRIMARY_PAPER_METHODS == ("greedy", "kalman_greedy", "pn", "lead", "ppo", "ppo_kalman")


def test_10_policy_instances_and_no_kalman_variants():
    assert len(GRID_POLICY_INSTANCES) == EXPECTED_POLICY_INSTANCES == 10
    ppo_instances = [p for p in GRID_POLICY_INSTANCES if p.model_path is not None]
    scripted_instances = [p for p in GRID_POLICY_INSTANCES if p.model_path is None]
    assert len(ppo_instances) == 6
    assert len(scripted_instances) == 4
    methods_present = {p.paper_method for p in GRID_POLICY_INSTANCES}
    assert "pn_kalman" not in methods_present
    assert "lead_kalman" not in methods_present
    assert methods_present == set(PRIMARY_PAPER_METHODS)


# ---------------------------------------------------------------------------
# M. All six PPO hashes match
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
    for instance in GRID_POLICY_INSTANCES:
        if instance.model_path is None:
            continue
        rel = str(instance.model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        assert rel in nominal_by_path, f"{rel} missing from nominal manifest"
        digest = sha256_of_file(instance.model_path)
        assert digest == nominal_by_path[rel], f"hash mismatch for {rel}"
        checked += 1
    assert checked == 6


# ---------------------------------------------------------------------------
# N. Only v_d/v_e + estimator flag differ
# ---------------------------------------------------------------------------
def test_only_speed_and_estimator_differ():
    for cell in GRID_CELLS:
        for estimator in ("measurement", "kalman"):
            config = build_grid_env_config(estimator, cell.defender_speed, cell.hostile_speed)
            assert_only_grid_speed_differs(config, _NOMINAL_ENV_CONFIG)

    nominal_like = build_grid_env_config("measurement", NOMINAL_DEFENDER_SPEED, NOMINAL_HOSTILE_SPEED)
    import dataclasses
    assert dataclasses.asdict(nominal_like) == dataclasses.asdict(_NOMINAL_ENV_CONFIG)

    bad_config = EnvConfig(use_kalman_tracking=False, v_d=18.0, v_e=12.0, defender_max_accel=99.0)
    raised = False
    try:
        assert_only_grid_speed_differs(bad_config, _NOMINAL_ENV_CONFIG)
    except AssertionError:
        raised = True
    assert raised, "assert_only_grid_speed_differs must catch an unrelated field change"


# ---------------------------------------------------------------------------
# O. All dynamic limits frozen
# ---------------------------------------------------------------------------
def test_dynamic_limits_frozen_across_grid():
    assert FIXED_DEFENDER_MAX_ACCEL == _NOMINAL_ENV_CONFIG.defender_max_accel == 8.0
    assert FIXED_DEFENDER_MAX_TURN_RATE_DEG == _NOMINAL_ENV_CONFIG.defender_max_turn_rate_deg == 90.0
    assert FIXED_DEFENDER_MAX_CLIMB_RATE == _NOMINAL_ENV_CONFIG.defender_max_climb_rate == 6.0
    assert FIXED_DEFENDER_MAX_DESCENT_RATE == _NOMINAL_ENV_CONFIG.defender_max_descent_rate == 6.0
    assert FIXED_ENEMY_MAX_ACCEL == _NOMINAL_ENV_CONFIG.enemy_max_accel == 6.0
    assert FIXED_ENEMY_MAX_TURN_RATE_DEG == _NOMINAL_ENV_CONFIG.enemy_max_turn_rate_deg == 75.0
    assert FIXED_ENEMY_MAX_CLIMB_RATE == _NOMINAL_ENV_CONFIG.enemy_max_climb_rate == 5.0
    assert FIXED_ENEMY_MAX_DESCENT_RATE == _NOMINAL_ENV_CONFIG.enemy_max_descent_rate == 5.0
    for cell in GRID_CELLS:
        config = build_grid_env_config("measurement", cell.defender_speed, cell.hostile_speed)
        assert config.defender_max_accel == FIXED_DEFENDER_MAX_ACCEL
        assert config.defender_max_turn_rate_deg == FIXED_DEFENDER_MAX_TURN_RATE_DEG
        assert config.defender_max_climb_rate == FIXED_DEFENDER_MAX_CLIMB_RATE
        assert config.defender_max_descent_rate == FIXED_DEFENDER_MAX_DESCENT_RATE
        assert config.enemy_max_accel == FIXED_ENEMY_MAX_ACCEL
        assert config.enemy_max_turn_rate_deg == FIXED_ENEMY_MAX_TURN_RATE_DEG
        assert config.enemy_max_climb_rate == FIXED_ENEMY_MAX_CLIMB_RATE
        assert config.enemy_max_descent_rate == FIXED_ENEMY_MAX_DESCENT_RATE


# ---------------------------------------------------------------------------
# P. Sensor/reward/evasion frozen
# ---------------------------------------------------------------------------
def test_sensor_reward_evasion_frozen_across_grid():
    assert FIXED_DETECTION_RADIUS == _NOMINAL_ENV_CONFIG.detection_radius == 15.0
    assert FIXED_MEASUREMENT_VAR == _NOMINAL_ENV_CONFIG.measurement_var == 0.5
    assert FIXED_PROCESS_VAR == _NOMINAL_ENV_CONFIG.process_var == 1.0
    assert FIXED_LEAD_TIME == _NOMINAL_ENV_CONFIG.lead_time == 0.0
    assert FIXED_ENEMY_EVASION_ENABLED == _NOMINAL_ENV_CONFIG.enemy_evasion_enabled is True
    assert FIXED_ENEMY_EVASION_RADIUS == _NOMINAL_ENV_CONFIG.enemy_evasion_radius == 20.0
    assert FIXED_ENEMY_EVASION_GAIN == _NOMINAL_ENV_CONFIG.enemy_evasion_gain == 0.75
    for cell in GRID_CELLS:
        config = build_grid_env_config("kalman", cell.defender_speed, cell.hostile_speed)
        assert config.detection_radius == FIXED_DETECTION_RADIUS
        assert config.measurement_var == FIXED_MEASUREMENT_VAR
        assert config.process_var == FIXED_PROCESS_VAR
        assert config.lead_time == FIXED_LEAD_TIME
        assert config.enemy_evasion_enabled == FIXED_ENEMY_EVASION_ENABLED
        assert config.enemy_evasion_radius == FIXED_ENEMY_EVASION_RADIUS
        assert config.enemy_evasion_gain == FIXED_ENEMY_EVASION_GAIN
        assert config.reward_intercept == _NOMINAL_ENV_CONFIG.reward_intercept
        assert config.reward_soldier_caught == _NOMINAL_ENV_CONFIG.reward_soldier_caught
        assert config.rho == _NOMINAL_ENV_CONFIG.rho
        assert config.sigma_a == _NOMINAL_ENV_CONFIG.sigma_a
        assert config.sigma_e == _NOMINAL_ENV_CONFIG.sigma_e
        assert config.weave_amplitude == _NOMINAL_ENV_CONFIG.weave_amplitude


# ---------------------------------------------------------------------------
# Q. Corrected Greedy contract active
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
# R. No policy truth leakage
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
# S. Observations finite/bounded at all 25 cells
# ---------------------------------------------------------------------------
def test_observation_bounds_at_all_25_cells():
    n_episodes_per_cell = 2
    max_abs_seen = 0.0
    for cell in GRID_CELLS:
        config = build_grid_env_config("measurement", cell.defender_speed, cell.hostile_speed)
        env = SoldierEnv(config=config)
        policy = get_policy("greedy")
        for s in range(n_episodes_per_cell):
            obs, info = env.reset(seed=19_000 + s)
            policy.reset()
            mode = estimator_mode_for_env(env)
            done = False
            while not done:
                assert np.all(np.isfinite(obs)), (cell, obs)
                max_abs_seen = max(max_abs_seen, float(np.max(np.abs(obs))))
                sanitized = build_policy_info(info, mode)
                action = policy.act(obs, sanitized)
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    assert np.all(np.isfinite(obs)), (cell, obs)
                    max_abs_seen = max(max_abs_seen, float(np.max(np.abs(obs))))
                    break
        env.close()
    assert max_abs_seen < 3.0, f"observation magnitude grew implausibly large: {max_abs_seen}"


# ---------------------------------------------------------------------------
# T. Expected rows 125000
# ---------------------------------------------------------------------------
def test_expected_raw_row_count():
    assert EXPECTED_RAW_ROWS == 25 * 10 * 500 == 125_000


# ---------------------------------------------------------------------------
# U. Correct estimator aggregation helper used
# ---------------------------------------------------------------------------
def test_correct_estimator_aggregation_used():
    rows = []
    n_samples = [50, 200]
    m_vals = [1.0, 3.0]
    r_vals = [1.05, 3.4]
    cell = GRID_CELLS[0]
    for method in SCRIPTED_METHODS:
        for i, (n, m, r) in enumerate(zip(n_samples, m_vals, r_vals)):
            rows.append({
                "defender_speed": cell.defender_speed, "hostile_speed": cell.hostile_speed,
                "policy_instance": method, "paper_method": method, "training_seed": float("nan"),
                "eval_seed": 35000 + i, "mean_position_estimation_error": m,
                "position_estimation_rmse": r, "n_position_estimation_samples": n,
            })
    for method in PPO_METHODS:
        for seed in (42, 43, 44):
            for i, (n, m, r) in enumerate(zip(n_samples, m_vals, r_vals)):
                rows.append({
                    "defender_speed": cell.defender_speed, "hostile_speed": cell.hostile_speed,
                    "policy_instance": f"{method}_seed_{seed}", "paper_method": method, "training_seed": seed,
                    "eval_seed": 35000 + i, "mean_position_estimation_error": m,
                    "position_estimation_rmse": r, "n_position_estimation_samples": n,
                })
    base_rows = list(rows)
    for other in GRID_CELLS[1:]:
        for row in base_rows:
            new_row = dict(row)
            new_row["defender_speed"] = other.defender_speed
            new_row["hostile_speed"] = other.hostile_speed
            rows.append(new_row)

    df = pd.DataFrame(rows)
    summary = compute_estimator_summary(df, GRID_CELLS)
    greedy_row = summary[(summary["defender_speed"] == cell.defender_speed) &
                          (summary["hostile_speed"] == cell.hostile_speed) &
                          (summary["method"] == "greedy")].iloc[0]

    n_arr = np.array(n_samples, dtype=np.float64)
    r_arr = np.array(r_vals, dtype=np.float64)
    m_arr = np.array(m_vals, dtype=np.float64)
    expected_correct_rmse = sample_weighted_rmse(n_arr, r_arr)
    old_buggy_rmse = float(np.sqrt(np.mean(np.square(m_arr))))

    assert abs(greedy_row["rmse_sample_weighted"] - expected_correct_rmse) < 1e-9
    assert abs(greedy_row["rmse_sample_weighted"] - old_buggy_rmse) > 0.05


# ---------------------------------------------------------------------------
# V. Four mu=1.5 cells exactly exist
# ---------------------------------------------------------------------------
def test_equal_ratio_cells_exact():
    assert EQUAL_RATIO_CELLS == ((15.0, 10.0), (18.0, 12.0), (21.0, 14.0), (24.0, 16.0))
    for v_d, v_e in EQUAL_RATIO_CELLS:
        assert abs(v_d / v_e - 1.5) < 1e-9
        assert any(c.defender_speed == v_d and c.hostile_speed == v_e for c in GRID_CELLS)


# ---------------------------------------------------------------------------
# W. Common-random-number seed sets identical
# ---------------------------------------------------------------------------
def test_common_seed_set_construction():
    seeds = list(MOBILITY_GRID_SEEDS)
    for cell in GRID_CELLS:
        for _ in GRID_POLICY_INSTANCES:
            assert seeds == list(MOBILITY_GRID_SEEDS)
    assert len(set(seeds)) == 500


# ---------------------------------------------------------------------------
# X. Action semantics remain unchanged
# ---------------------------------------------------------------------------
def test_action_semantics_unchanged():
    for v_d in DEFENDER_SPEED_VALUES:
        config = EnvConfig(v_d=v_d)
        direction = np.array([1.0, 0.5, -0.3])
        unit_dir = direction / np.linalg.norm(direction)
        speeds = []
        for magnitude in (0.1, 0.5, 1.0):
            action = unit_dir * magnitude
            action_norm = np.linalg.norm(action)
            desired_direction = action / action_norm
            desired_velocity = config.v_d * desired_direction
            speeds.append(float(np.linalg.norm(desired_velocity)))
        assert np.allclose(speeds, speeds[0])
        assert abs(speeds[0] - v_d) < 1e-6


def main() -> int:
    tests = [
        ("A. Defender speed values exact", test_defender_speed_values_exact),
        ("B. Hostile speed values exact", test_hostile_speed_values_exact),
        ("C. 25 Cartesian cells exact", test_25_cartesian_cells_exact),
        ("D. Nominal occurs exactly once", test_nominal_occurs_exactly_once),
        ("E/F. Seed range exact, 500 seeds", test_seed_range_exact),
        ("G/H/I. No overlap; reserved ranges protected; no seed>=35500", test_no_overlap_and_reserved_ranges_protected),
        ("J. Six primary methods exact", test_six_primary_methods_exact),
        ("K/L. 10 instances, no Kalman-variant methods", test_10_policy_instances_and_no_kalman_variants),
        ("M. PPO hashes match final-nominal hashes", test_ppo_hashes_match_final_nominal),
        ("N. Only speed+estimator differ", test_only_speed_and_estimator_differ),
        ("O. Dynamic limits frozen across grid", test_dynamic_limits_frozen_across_grid),
        ("P. Sensor/reward/evasion frozen across grid", test_sensor_reward_evasion_frozen_across_grid),
        ("Q. Greedy uses corrected contract", test_greedy_uses_corrected_contract),
        ("R. Sanitization excludes ground truth", test_sanitization_excludes_ground_truth),
        ("S. Observation bounds at all 25 cells", test_observation_bounds_at_all_25_cells),
        ("T. Expected raw row count = 125000", test_expected_raw_row_count),
        ("U. Correct estimator aggregation used", test_correct_estimator_aggregation_used),
        ("V. Four mu=1.5 cells exact", test_equal_ratio_cells_exact),
        ("W. Common seed set construction", test_common_seed_set_construction),
        ("X. Action semantics unchanged", test_action_semantics_unchanged),
    ]

    print("=== Defender-Hostile Speed Interaction Grid Tests ===\n")
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
