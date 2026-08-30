"""Defender-maneuverability (accel x turn-rate) interaction grid correctness tests
-- locked guardrails. Script-style test consistent with the project's other test
files (no pytest).

Covers (per the maneuverability task spec):
    A. Accel values exactly 4,8,12.
    B. Turn values exactly 45,90,135.
    C. 9 Cartesian cells.
    D. Nominal (8,90) occurs exactly once.
    E. Seeds exactly 34000..34999.
    F. 1000 seeds.
    G. No overlap with any previously opened seed range.
    H. 35000..35499 remains prior opened mobility-grid data, never rerun.
    I. >=35500 untouched.
    J. Six methods exactly.
    K. 10 instances.
    L. PN-Kalman/Lead-Kalman absent.
    M. Model hashes exact.
    N. Only accel/turn + estimator flag differ.
    O. v_d=18 and v_e=12 fixed.
    P. Vertical rates fixed.
    Q. Hostile dynamics fixed.
    R. Sensor fixed.
    S. Evasion gain=.75/radius20 fixed.
    T. Reward fixed.
    U. Corrected Greedy contract.
    V. No truth leakage.
    W. Observations remain finite/bounded.
    X. Correct estimator aggregation.
    Y. Expected rows 90000.
    Z. Dynamic physical checker uses CELL-SPECIFIC accel/turn limits.

Run directly:
    python test_robustness_maneuverability.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env
from uav_defend.policies.baseline.greedy_intercept_policy import GreedyInterceptPolicy

from experiments.final_experiment_protocol import FINAL_NOMINAL_SEEDS
from experiments.acquisition_ablation_protocol import ACQUISITION_ABLATION_SEEDS
from experiments.robustness_detection_protocol import DETECTION_SWEEP_SEEDS
from experiments.robustness_measurement_noise_protocol import MEASUREMENT_SWEEP_SEEDS
from experiments.robustness_mobility_protocol import MOBILITY_SWEEP_SEEDS
from experiments.robustness_hostile_evasion_protocol import HOSTILE_EVASION_SEEDS
from experiments.robustness_mobility_grid_protocol import MOBILITY_GRID_SEEDS
from experiments.final_eval_utils import sha256_of_file, PhysicalLimitViolation, _check_physical_limits
from experiments.robustness_maneuverability_protocol import (
    DEFENDER_MAX_ACCEL_VALUES,
    DEFENDER_MAX_TURN_RATE_VALUES,
    NOMINAL_DEFENDER_MAX_ACCEL,
    NOMINAL_DEFENDER_MAX_TURN_RATE_DEG,
    NOMINAL_DEFENDER_SPEED,
    NOMINAL_HOSTILE_SPEED,
    MANEUVERABILITY_CELLS,
    EXPECTED_MANEUVERABILITY_CELLS,
    NOMINAL_CELL,
    MANEUVERABILITY_SEED_OFFSET,
    MANEUVERABILITY_EPISODES,
    MANEUVERABILITY_SEEDS,
    MOBILITY_GRID_SEED_OFFSET as MG_SEED_OFFSET,
    MOBILITY_GRID_EPISODES as MG_EPISODES,
    MAX_ALLOWED_SEED,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    MANEUVERABILITY_POLICY_INSTANCES,
    EXPECTED_N_POLICY_INSTANCES,
    EXPECTED_RAW_ROWS,
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
    build_maneuverability_env_config,
    assert_only_maneuverability_differs,
    assert_maneuverability_seed,
)
from experiments.maneuverability_eval_utils import run_maneuverability_episode
from experiments.estimator_stats import sample_weighted_rmse
from experiments.run_robustness_maneuverability_sweep import compute_estimator_summary

PROJECT_ROOT = Path(__file__).parent
FINAL_NOMINAL_HASHES_PATH = PROJECT_ROOT / "results" / "final_nominal" / "model_hashes.json"
_NOMINAL_ENV_CONFIG = EnvConfig()

_DIAGNOSTIC_SEEDS = range(19_600, 19_620)  # non-publication diagnostic seeds, below all opened ranges


# ---------------------------------------------------------------------------
# A/B. Grid values exact
# ---------------------------------------------------------------------------
def test_accel_values_exact():
    assert DEFENDER_MAX_ACCEL_VALUES == (4.0, 8.0, 12.0)


def test_turn_values_exact():
    assert DEFENDER_MAX_TURN_RATE_VALUES == (45.0, 90.0, 135.0)


# ---------------------------------------------------------------------------
# C/D. 9 Cartesian cells ; nominal occurs once
# ---------------------------------------------------------------------------
def test_9_cartesian_cells_exact():
    assert len(MANEUVERABILITY_CELLS) == EXPECTED_MANEUVERABILITY_CELLS == 9
    pairs = {(c.defender_max_accel, c.defender_max_turn_rate_deg) for c in MANEUVERABILITY_CELLS}
    assert len(pairs) == 9
    expected = {(a, t) for a in DEFENDER_MAX_ACCEL_VALUES for t in DEFENDER_MAX_TURN_RATE_VALUES}
    assert pairs == expected


def test_nominal_cell_occurs_once():
    assert NOMINAL_DEFENDER_MAX_ACCEL == 8.0
    assert NOMINAL_DEFENDER_MAX_TURN_RATE_DEG == 90.0
    nominal_matches = [c for c in MANEUVERABILITY_CELLS if c.is_nominal]
    assert len(nominal_matches) == 1
    assert nominal_matches[0] == NOMINAL_CELL


# ---------------------------------------------------------------------------
# E/F. Seeds exact
# ---------------------------------------------------------------------------
def test_seeds_exactly_34000_to_34999():
    assert MANEUVERABILITY_SEED_OFFSET == 34_000
    assert MANEUVERABILITY_SEEDS.start == 34_000
    assert MANEUVERABILITY_SEEDS.stop == 35_000
    assert_maneuverability_seed(34_000)
    assert_maneuverability_seed(34_999)
    try:
        assert_maneuverability_seed(35_000)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_1000_seeds():
    assert MANEUVERABILITY_EPISODES == 1_000
    assert len(MANEUVERABILITY_SEEDS) == 1_000


# ---------------------------------------------------------------------------
# G. No overlap with previously opened ranges
# ---------------------------------------------------------------------------
def test_no_overlap_with_previous_ranges():
    maneuverability_set = set(MANEUVERABILITY_SEEDS)
    other_ranges = {
        "FINAL_NOMINAL_SEEDS": FINAL_NOMINAL_SEEDS,
        "ACQUISITION_ABLATION_SEEDS": ACQUISITION_ABLATION_SEEDS,
        "DETECTION_SWEEP_SEEDS": DETECTION_SWEEP_SEEDS,
        "MEASUREMENT_SWEEP_SEEDS": MEASUREMENT_SWEEP_SEEDS,
        "MOBILITY_SWEEP_SEEDS": MOBILITY_SWEEP_SEEDS,
        "HOSTILE_EVASION_SEEDS": HOSTILE_EVASION_SEEDS,
        "MOBILITY_GRID_SEEDS": MOBILITY_GRID_SEEDS,
    }
    for name, other in other_ranges.items():
        overlap = maneuverability_set & set(other)
        assert not overlap, f"MANEUVERABILITY_SEEDS overlaps {name}: {overlap}"


# ---------------------------------------------------------------------------
# H. 35000..35499 remains prior opened mobility-grid data, never rerun
# ---------------------------------------------------------------------------
def test_mobility_grid_range_untouched_by_this_module():
    assert MG_SEED_OFFSET == 35_000
    assert MG_EPISODES == 500
    assert set(MOBILITY_GRID_SEEDS) == set(range(35_000, 35_500))
    assert set(MOBILITY_GRID_SEEDS).isdisjoint(set(MANEUVERABILITY_SEEDS))


# ---------------------------------------------------------------------------
# I. >=35500 untouched
# ---------------------------------------------------------------------------
def test_max_allowed_seed():
    assert MAX_ALLOWED_SEED == 34_999
    assert max(MANEUVERABILITY_SEEDS) < 35_500
    for seed in MANEUVERABILITY_SEEDS:
        assert seed <= MAX_ALLOWED_SEED


# ---------------------------------------------------------------------------
# J. Six methods exact
# ---------------------------------------------------------------------------
def test_six_paper_methods_exact():
    assert PRIMARY_PAPER_METHODS == ("greedy", "kalman_greedy", "pn", "lead", "ppo", "ppo_kalman")
    assert len(SCRIPTED_METHODS) == 4
    assert len(PPO_METHODS) == 2


# ---------------------------------------------------------------------------
# K. 10 policy instances
# ---------------------------------------------------------------------------
def test_ten_policy_instances():
    assert EXPECTED_N_POLICY_INSTANCES == 10
    assert len(MANEUVERABILITY_POLICY_INSTANCES) == 10


# ---------------------------------------------------------------------------
# L. PN-Kalman / Lead-Kalman absent
# ---------------------------------------------------------------------------
def test_no_pn_kalman_or_lead_kalman():
    names = {i.policy_instance for i in MANEUVERABILITY_POLICY_INSTANCES}
    methods = {i.paper_method for i in MANEUVERABILITY_POLICY_INSTANCES}
    assert "pn_kalman" not in methods
    assert "lead_kalman" not in methods
    assert not any("pn_kalman" in n or "lead_kalman" in n for n in names)


# ---------------------------------------------------------------------------
# M. Six PPO model hashes match final-nominal manifest
# ---------------------------------------------------------------------------
def test_ppo_model_hashes_match_final_nominal():
    assert FINAL_NOMINAL_HASHES_PATH.exists(), "final_nominal/model_hashes.json missing"
    nominal_hashes = {row["model_path"].replace("\\", "/"): row["sha256"] for row in json.loads(FINAL_NOMINAL_HASHES_PATH.read_text())}
    ppo_instances = [i for i in MANEUVERABILITY_POLICY_INSTANCES if i.model_path is not None]
    assert len(ppo_instances) == 6
    for instance in ppo_instances:
        rel = str(instance.model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        assert rel in nominal_hashes, f"{rel} not found in final nominal manifest"
        digest = sha256_of_file(instance.model_path)
        assert digest == nominal_hashes[rel], f"hash mismatch for {rel}"


# ---------------------------------------------------------------------------
# N. Only accel/turn (+estimator flag) differs -- config isolation
# ---------------------------------------------------------------------------
def test_only_maneuverability_differs_from_nominal():
    for accel, turn in [(c.defender_max_accel, c.defender_max_turn_rate_deg) for c in MANEUVERABILITY_CELLS]:
        config = build_maneuverability_env_config("measurement", accel, turn)
        assert_only_maneuverability_differs(config, _NOMINAL_ENV_CONFIG)
    config_kalman = build_maneuverability_env_config("kalman", NOMINAL_DEFENDER_MAX_ACCEL, NOMINAL_DEFENDER_MAX_TURN_RATE_DEG)
    swept_dict = dataclasses.asdict(config_kalman)
    nominal_dict = dataclasses.asdict(_NOMINAL_ENV_CONFIG)
    diffs = {k for k in swept_dict if swept_dict[k] != nominal_dict[k]}
    assert diffs <= {"use_kalman_tracking", "defender_max_accel", "defender_max_turn_rate_deg"}

    try:
        bad_config = dataclasses.replace(_NOMINAL_ENV_CONFIG, v_d=99.0, defender_max_accel=4.0)
        assert_only_maneuverability_differs(bad_config, _NOMINAL_ENV_CONFIG)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "assert_only_maneuverability_differs failed to catch an extraneous v_d change"


# ---------------------------------------------------------------------------
# O. v_d/v_e fixed
# ---------------------------------------------------------------------------
def test_mobility_fixed():
    assert NOMINAL_DEFENDER_SPEED == 18.0
    assert NOMINAL_HOSTILE_SPEED == 12.0
    for accel, turn in [(c.defender_max_accel, c.defender_max_turn_rate_deg) for c in MANEUVERABILITY_CELLS]:
        config = build_maneuverability_env_config("measurement", accel, turn)
        assert config.v_d == 18.0
        assert config.v_e == 12.0


# ---------------------------------------------------------------------------
# P. Vertical rates fixed
# ---------------------------------------------------------------------------
def test_vertical_rates_fixed():
    assert FIXED_DEFENDER_MAX_CLIMB_RATE == 6.0
    assert FIXED_DEFENDER_MAX_DESCENT_RATE == 6.0
    for accel, turn in [(c.defender_max_accel, c.defender_max_turn_rate_deg) for c in MANEUVERABILITY_CELLS]:
        config = build_maneuverability_env_config("measurement", accel, turn)
        assert config.defender_max_climb_rate == 6.0
        assert config.defender_max_descent_rate == 6.0


# ---------------------------------------------------------------------------
# Q. Hostile dynamics fixed
# ---------------------------------------------------------------------------
def test_hostile_dynamics_fixed():
    for accel, turn in [(c.defender_max_accel, c.defender_max_turn_rate_deg) for c in MANEUVERABILITY_CELLS]:
        config = build_maneuverability_env_config("measurement", accel, turn)
        assert config.enemy_max_accel == FIXED_ENEMY_MAX_ACCEL == 6.0
        assert config.enemy_max_turn_rate_deg == FIXED_ENEMY_MAX_TURN_RATE_DEG == 75.0
        assert config.enemy_max_climb_rate == FIXED_ENEMY_MAX_CLIMB_RATE == 5.0
        assert config.enemy_max_descent_rate == FIXED_ENEMY_MAX_DESCENT_RATE == 5.0


# ---------------------------------------------------------------------------
# R. Sensor fixed
# ---------------------------------------------------------------------------
def test_sensing_fixed():
    assert FIXED_DETECTION_RADIUS == 15.0
    assert FIXED_MEASUREMENT_VAR == 0.5
    assert FIXED_PROCESS_VAR == 1.0
    assert FIXED_LEAD_TIME == 0.0
    for accel, turn in [(c.defender_max_accel, c.defender_max_turn_rate_deg) for c in MANEUVERABILITY_CELLS]:
        config = build_maneuverability_env_config("measurement", accel, turn)
        assert config.detection_radius == 15.0
        assert config.measurement_var == 0.5
        assert config.process_var == 1.0
        assert config.lead_time == 0.0


# ---------------------------------------------------------------------------
# S. Evasion gain=.75/radius20 fixed
# ---------------------------------------------------------------------------
def test_evasion_fixed():
    assert FIXED_ENEMY_EVASION_ENABLED is True
    assert FIXED_ENEMY_EVASION_RADIUS == 20.0
    assert FIXED_ENEMY_EVASION_GAIN == 0.75
    for accel, turn in [(c.defender_max_accel, c.defender_max_turn_rate_deg) for c in MANEUVERABILITY_CELLS]:
        config = build_maneuverability_env_config("measurement", accel, turn)
        assert config.enemy_evasion_enabled is True
        assert config.enemy_evasion_radius == 20.0
        assert config.enemy_evasion_gain == 0.75


# ---------------------------------------------------------------------------
# T. Reward/termination fixed
# ---------------------------------------------------------------------------
def test_reward_and_termination_geometry_fixed():
    for accel, turn in [(c.defender_max_accel, c.defender_max_turn_rate_deg) for c in MANEUVERABILITY_CELLS]:
        config = build_maneuverability_env_config("measurement", accel, turn)
        assert config.intercept_radius == _NOMINAL_ENV_CONFIG.intercept_radius
        assert config.threat_radius == _NOMINAL_ENV_CONFIG.threat_radius
        assert config.unsafe_intercept_radius == _NOMINAL_ENV_CONFIG.unsafe_intercept_radius
        assert config.max_steps == _NOMINAL_ENV_CONFIG.max_steps
        assert config.dt == _NOMINAL_ENV_CONFIG.dt
        assert config.L == _NOMINAL_ENV_CONFIG.L
        assert config.max_altitude == _NOMINAL_ENV_CONFIG.max_altitude


# ---------------------------------------------------------------------------
# U. Corrected Greedy contract active
# ---------------------------------------------------------------------------
def test_corrected_greedy_contract_active():
    config = build_maneuverability_env_config("measurement", NOMINAL_DEFENDER_MAX_ACCEL, NOMINAL_DEFENDER_MAX_TURN_RATE_DEG)
    env = SoldierEnv(config=config)
    policy = GreedyInterceptPolicy()
    obs, info = env.reset(seed=19_600)
    policy.reset()
    estimator_mode = estimator_mode_for_env(env)
    policy_info = build_policy_info(info, estimator_mode)
    action = policy.act(obs, policy_info)
    assert action is not None
    assert np.all(np.isfinite(np.asarray(action, dtype=np.float64)))
    assert hasattr(policy, "act")


# ---------------------------------------------------------------------------
# V. No truth leakage
# ---------------------------------------------------------------------------
def test_no_policy_truth_leakage():
    config = build_maneuverability_env_config("measurement", NOMINAL_DEFENDER_MAX_ACCEL, NOMINAL_DEFENDER_MAX_TURN_RATE_DEG)
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=19_601)
    estimator_mode = estimator_mode_for_env(env)
    assert estimator_mode == "measurement"
    policy_info = build_policy_info(info, estimator_mode)
    assert "e_hat" not in policy_info or policy_info.get("e_hat") is None
    assert "enemy_pos" not in policy_info

    config_k = build_maneuverability_env_config("kalman", NOMINAL_DEFENDER_MAX_ACCEL, NOMINAL_DEFENDER_MAX_TURN_RATE_DEG)
    env_k = SoldierEnv(config=config_k)
    obs_k, info_k = env_k.reset(seed=19_601)
    estimator_mode_k = estimator_mode_for_env(env_k)
    assert estimator_mode_k == "kalman"
    policy_info_k = build_policy_info(info_k, estimator_mode_k)
    assert "enemy_pos" not in policy_info_k


# ---------------------------------------------------------------------------
# W. Observations remain finite/bounded
# ---------------------------------------------------------------------------
def test_observations_finite_at_extreme_cells():
    for accel, turn in [(4.0, 45.0), (12.0, 135.0)]:
        config = build_maneuverability_env_config("measurement", accel, turn)
        env = SoldierEnv(config=config)
        policy = GreedyInterceptPolicy()
        obs, info = env.reset(seed=19_602)
        policy.reset()
        estimator_mode = estimator_mode_for_env(env)
        assert np.all(np.isfinite(obs))
        for _ in range(50):
            policy_info = build_policy_info(info, estimator_mode)
            action = policy.act(obs, policy_info)
            obs, reward, done, trunc, info = env.step(action)
            assert np.all(np.isfinite(obs)), f"non-finite obs at cell ({accel},{turn})"
            assert np.isfinite(reward)
            if done or trunc:
                break


# ---------------------------------------------------------------------------
# X. Correct estimator aggregation
# ---------------------------------------------------------------------------
def test_corrected_estimator_aggregation_used():
    import pandas as pd
    n = np.array([50.0, 80.0, 30.0])
    r = np.array([1.0, 2.0, 3.0])
    correct = sample_weighted_rmse(n, r)
    naive_wrong = np.sqrt(np.mean(r ** 2))
    assert abs(correct - naive_wrong) > 1e-9

    df = pd.DataFrame({
        "defender_max_accel": [8.0] * 3,
        "defender_max_turn_rate_deg": [90.0] * 3,
        "policy_instance": ["greedy"] * 3,
        "paper_method": ["greedy"] * 3,
        "training_seed": [float("nan")] * 3,
        "success": [1, 1, 0],
        "n_position_estimation_samples": n,
        "mean_position_estimation_error": [0.9, 1.8, 2.7],
        "position_estimation_rmse": r,
    })
    summary = compute_estimator_summary(df, [(8.0, 90.0)])
    row = summary[(summary["method"] == "greedy") & (summary["defender_max_accel"] == 8.0)].iloc[0]
    assert abs(row["rmse_sample_weighted"] - correct) < 1e-9
    assert abs(row["rmse_sample_weighted"] - naive_wrong) > 1e-9


# ---------------------------------------------------------------------------
# Y. Expected rows = 90,000
# ---------------------------------------------------------------------------
def test_expected_rows_90000():
    assert EXPECTED_RAW_ROWS == len(MANEUVERABILITY_CELLS) * EXPECTED_N_POLICY_INSTANCES * MANEUVERABILITY_EPISODES
    assert EXPECTED_RAW_ROWS == 90_000


# ---------------------------------------------------------------------------
# Z. Dynamic physical checker uses CELL-SPECIFIC accel/turn limits
# ---------------------------------------------------------------------------
def test_physical_checker_uses_cell_specific_limits():
    config_restrictive = build_maneuverability_env_config("measurement", 4.0, 45.0)
    config_permissive = build_maneuverability_env_config("measurement", 12.0, 135.0)
    assert config_restrictive.defender_max_accel == 4.0
    assert config_permissive.defender_max_accel == 12.0

    env = SoldierEnv(config=config_restrictive)
    obs, info = env.reset(seed=19_603)
    _check_physical_limits(info, config_restrictive)

    # A restrictive-cell info dict artificially compared against the
    # PERMISSIVE config's higher limits must NOT raise (proves the checker
    # actually reads config-specific limits rather than a hardcoded global).
    fake_info = dict(info)
    fake_info["defender_accel"] = 10.0  # exceeds restrictive limit (4.0) but not permissive (12.0)
    try:
        _check_physical_limits(fake_info, config_restrictive)
        raised = False
    except PhysicalLimitViolation:
        raised = True
    assert raised, "physical checker failed to flag a violation of the restrictive cell's accel limit"
    _check_physical_limits(fake_info, config_permissive)  # must NOT raise under the permissive cell's limit


# ---------------------------------------------------------------------------
# Bonus: live rollout smoke check across the full grid (config isolation +
# episode integrity), reusing non-publication diagnostic seeds only.
# ---------------------------------------------------------------------------
def test_live_rollout_smoke_across_grid():
    for accel, turn in [(4.0, 45.0), (8.0, 90.0), (12.0, 135.0)]:
        config = build_maneuverability_env_config("measurement", accel, turn)
        env = SoldierEnv(config=config)
        policy = GreedyInterceptPolicy()
        row = run_maneuverability_episode(env, policy, seed=19_604, dt=config.dt)
        assert row["defender_max_accel"] == accel
        assert row["defender_max_turn_rate_deg"] == turn
        assert row["success"] in (0, 1)
        assert np.isfinite(row["mean_accel_utilization"])
        assert np.isfinite(row["mean_turn_rate_utilization"])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    tests.sort(key=lambda kv: kv[0])
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
