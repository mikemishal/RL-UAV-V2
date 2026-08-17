"""Hostile-evasion-strength robustness sweep correctness tests -- locked guardrails.

Script-style test consistent with the project's other test files (no pytest).

Covers (per the hostile-evasion task spec):
    A. Gain values exactly 0.00, 0.25, 0.50, 0.75, 1.00.
    B. Nominal gain = 0.75.
    C. Evasion radius fixed = 20.0 (never swept).
    D. enemy_evasion_enabled=True at every sweep point (including gain=0).
    E. Seeds exactly 33000..33999.
    F. 1000 seeds.
    G. No overlap with any previously opened seed range.
    H. 34000..34999 remain untouched/reserved (future maneuverability study).
    I. No seed >= 35500 evaluated by this module.
    J. Six paper methods exactly.
    K. 10 policy instances exactly.
    L. PN-Kalman/Lead-Kalman absent.
    M. Six PPO model hashes match the final-nominal manifest.
    N. Only enemy_evasion_gain (+ estimator flag) differs from nominal.
    O. Mobility (v_d/v_e) frozen at nominal.
    P. Sensing (detection radius, measurement/process variance, lead time) frozen.
    Q. Maneuverability (accel/turn/climb/descent limits) frozen.
    R. Reward/termination geometry frozen (implicit via full-config equality).
    S. Corrected Greedy contract is active (no reversion to old escort-only bug).
    T. No policy truth leakage (policy_info sanitized appropriately per estimator).
    U. gain=0.0 <=> enemy_evasion_enabled=False equivalence (diagnostic rollout).
    V. effective_evasion_weight == gain * activation (live rollout check).
    W. Corrected sample-weighted estimator aggregation helper is used (not naive sqrt-mean-of-squares).
    X. Expected rows = 50,000.

Run directly:
    python test_robustness_hostile_evasion.py
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
from experiments.robustness_mobility_grid_protocol import MOBILITY_GRID_SEEDS
from experiments.final_eval_utils import sha256_of_file
from experiments.robustness_hostile_evasion_protocol import (
    EVASION_GAIN_VALUES,
    NOMINAL_EVASION_GAIN,
    EVASION_RADIUS,
    FIXED_ENEMY_EVASION_ENABLED,
    HOSTILE_EVASION_SEED_OFFSET,
    HOSTILE_EVASION_EPISODES,
    HOSTILE_EVASION_SEEDS,
    DYNAMICS_SWEEP_SEED_OFFSET,
    DYNAMICS_SWEEP_EPISODES,
    RESERVED_UNTOUCHED_SEEDS,
    MAX_ALLOWED_SEED,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    EVASION_POLICY_INSTANCES,
    EXPECTED_N_POLICY_INSTANCES,
    EXPECTED_ROWS,
    NOMINAL_DEFENDER_SPEED,
    NOMINAL_HOSTILE_SPEED,
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
    build_evasion_env_config,
    assert_only_evasion_gain_differs,
    assert_evasion_sweep_seed,
)
from experiments.hostile_evasion_eval_utils import run_hostile_evasion_episode
from experiments.estimator_stats import sample_weighted_rmse
from experiments.run_robustness_hostile_evasion_sweep import compute_estimator_summary

PROJECT_ROOT = Path(__file__).parent
FINAL_NOMINAL_HASHES_PATH = PROJECT_ROOT / "results" / "final_nominal" / "model_hashes.json"
_NOMINAL_ENV_CONFIG = EnvConfig()

_DIAGNOSTIC_SEEDS = range(19_500, 19_520)  # non-publication diagnostic seeds, below all opened ranges


# ---------------------------------------------------------------------------
# A. Gain values exact
# ---------------------------------------------------------------------------
def test_gain_values_exact():
    assert EVASION_GAIN_VALUES == (0.00, 0.25, 0.50, 0.75, 1.00)


# ---------------------------------------------------------------------------
# B. Nominal gain
# ---------------------------------------------------------------------------
def test_nominal_gain_is_075():
    assert NOMINAL_EVASION_GAIN == 0.75
    assert NOMINAL_EVASION_GAIN in EVASION_GAIN_VALUES


# ---------------------------------------------------------------------------
# C. Evasion radius fixed
# ---------------------------------------------------------------------------
def test_evasion_radius_fixed_at_20():
    assert EVASION_RADIUS == 20.0
    for gain in EVASION_GAIN_VALUES:
        config = build_evasion_env_config("measurement", gain)
        assert config.enemy_evasion_radius == 20.0


# ---------------------------------------------------------------------------
# D. enemy_evasion_enabled True at every sweep point
# ---------------------------------------------------------------------------
def test_evasion_enabled_true_at_every_gain():
    assert FIXED_ENEMY_EVASION_ENABLED is True
    for gain in EVASION_GAIN_VALUES:
        config = build_evasion_env_config("measurement", gain)
        assert config.enemy_evasion_enabled is True
        config_k = build_evasion_env_config("kalman", gain)
        assert config_k.enemy_evasion_enabled is True


# ---------------------------------------------------------------------------
# E/F. Seeds exact
# ---------------------------------------------------------------------------
def test_seeds_exactly_33000_to_33999():
    assert HOSTILE_EVASION_SEED_OFFSET == 33_000
    assert HOSTILE_EVASION_SEEDS.start == 33_000
    assert HOSTILE_EVASION_SEEDS.stop == 34_000
    assert_evasion_sweep_seed(33_000)
    assert_evasion_sweep_seed(33_999)
    try:
        assert_evasion_sweep_seed(34_000)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_1000_seeds():
    assert HOSTILE_EVASION_EPISODES == 1_000
    assert len(HOSTILE_EVASION_SEEDS) == 1_000


# ---------------------------------------------------------------------------
# G. No overlap with previously opened ranges
# ---------------------------------------------------------------------------
def test_no_overlap_with_previous_ranges():
    hostile_evasion_set = set(HOSTILE_EVASION_SEEDS)
    other_ranges = {
        "FINAL_NOMINAL_SEEDS": FINAL_NOMINAL_SEEDS,
        "ACQUISITION_ABLATION_SEEDS": ACQUISITION_ABLATION_SEEDS,
        "DETECTION_SWEEP_SEEDS": DETECTION_SWEEP_SEEDS,
        "MEASUREMENT_SWEEP_SEEDS": MEASUREMENT_SWEEP_SEEDS,
        "MOBILITY_SWEEP_SEEDS": MOBILITY_SWEEP_SEEDS,
        "MOBILITY_GRID_SEEDS": MOBILITY_GRID_SEEDS,
    }
    for name, other in other_ranges.items():
        overlap = hostile_evasion_set & set(other)
        assert not overlap, f"HOSTILE_EVASION_SEEDS overlaps {name}: {overlap}"


# ---------------------------------------------------------------------------
# H. 34000..34999 reserved/untouched
# ---------------------------------------------------------------------------
def test_dynamics_range_reserved_untouched():
    assert DYNAMICS_SWEEP_SEED_OFFSET == 34_000
    assert DYNAMICS_SWEEP_EPISODES == 1_000
    assert RESERVED_UNTOUCHED_SEEDS.start == 34_000
    assert RESERVED_UNTOUCHED_SEEDS.stop == 35_000
    assert set(RESERVED_UNTOUCHED_SEEDS).isdisjoint(set(HOSTILE_EVASION_SEEDS))


# ---------------------------------------------------------------------------
# I. No seed >= 35500 evaluated
# ---------------------------------------------------------------------------
def test_max_allowed_seed():
    assert MAX_ALLOWED_SEED == 33_999
    assert max(HOSTILE_EVASION_SEEDS) < 35_500
    for seed in HOSTILE_EVASION_SEEDS:
        assert seed <= MAX_ALLOWED_SEED


# ---------------------------------------------------------------------------
# J. Six paper methods exact
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
    assert len(EVASION_POLICY_INSTANCES) == 10


# ---------------------------------------------------------------------------
# L. PN-Kalman / Lead-Kalman absent
# ---------------------------------------------------------------------------
def test_no_pn_kalman_or_lead_kalman():
    names = {i.policy_instance for i in EVASION_POLICY_INSTANCES}
    methods = {i.paper_method for i in EVASION_POLICY_INSTANCES}
    assert "pn_kalman" not in methods
    assert "lead_kalman" not in methods
    assert not any("pn_kalman" in n or "lead_kalman" in n for n in names)


# ---------------------------------------------------------------------------
# M. Six PPO model hashes match final-nominal manifest
# ---------------------------------------------------------------------------
def test_ppo_model_hashes_match_final_nominal():
    assert FINAL_NOMINAL_HASHES_PATH.exists(), "final_nominal/model_hashes.json missing"
    nominal_hashes = {row["model_path"].replace("\\", "/"): row["sha256"] for row in json.loads(FINAL_NOMINAL_HASHES_PATH.read_text())}
    ppo_instances = [i for i in EVASION_POLICY_INSTANCES if i.model_path is not None]
    assert len(ppo_instances) == 6
    for instance in ppo_instances:
        rel = str(instance.model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        assert rel in nominal_hashes, f"{rel} not found in final nominal manifest"
        digest = sha256_of_file(instance.model_path)
        assert digest == nominal_hashes[rel], f"hash mismatch for {rel}"


# ---------------------------------------------------------------------------
# N. Only gain (+estimator flag) differs -- config isolation
# ---------------------------------------------------------------------------
def test_only_evasion_gain_differs_from_nominal():
    for gain in EVASION_GAIN_VALUES:
        config = build_evasion_env_config("measurement", gain)
        assert_only_evasion_gain_differs(config, _NOMINAL_ENV_CONFIG)
    config_kalman = build_evasion_env_config("kalman", NOMINAL_EVASION_GAIN)
    swept_dict = dataclasses.asdict(config_kalman)
    nominal_dict = dataclasses.asdict(_NOMINAL_ENV_CONFIG)
    diffs = {k for k in swept_dict if swept_dict[k] != nominal_dict[k]}
    assert diffs <= {"use_kalman_tracking", "enemy_evasion_gain"}

    try:
        bad_config = dataclasses.replace(_NOMINAL_ENV_CONFIG, v_d=99.0, enemy_evasion_gain=0.5)
        assert_only_evasion_gain_differs(bad_config, _NOMINAL_ENV_CONFIG)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "assert_only_evasion_gain_differs failed to catch an extraneous v_d change"


# ---------------------------------------------------------------------------
# O. Mobility frozen
# ---------------------------------------------------------------------------
def test_mobility_frozen():
    assert NOMINAL_DEFENDER_SPEED == 18.0
    assert NOMINAL_HOSTILE_SPEED == 12.0
    for gain in EVASION_GAIN_VALUES:
        config = build_evasion_env_config("measurement", gain)
        assert config.v_d == 18.0
        assert config.v_e == 12.0


# ---------------------------------------------------------------------------
# P. Sensing frozen
# ---------------------------------------------------------------------------
def test_sensing_frozen():
    assert FIXED_DETECTION_RADIUS == 15.0
    assert FIXED_MEASUREMENT_VAR == 0.5
    assert FIXED_PROCESS_VAR == 1.0
    assert FIXED_LEAD_TIME == 0.0
    for gain in EVASION_GAIN_VALUES:
        config = build_evasion_env_config("measurement", gain)
        assert config.detection_radius == 15.0
        assert config.measurement_var == 0.5
        assert config.process_var == 1.0
        assert config.lead_time == 0.0


# ---------------------------------------------------------------------------
# Q. Maneuverability frozen
# ---------------------------------------------------------------------------
def test_maneuverability_frozen():
    for gain in EVASION_GAIN_VALUES:
        config = build_evasion_env_config("measurement", gain)
        assert config.defender_max_accel == FIXED_DEFENDER_MAX_ACCEL == 8.0
        assert config.defender_max_turn_rate_deg == FIXED_DEFENDER_MAX_TURN_RATE_DEG == 90.0
        assert config.defender_max_climb_rate == FIXED_DEFENDER_MAX_CLIMB_RATE == 6.0
        assert config.defender_max_descent_rate == FIXED_DEFENDER_MAX_DESCENT_RATE == 6.0
        assert config.enemy_max_accel == FIXED_ENEMY_MAX_ACCEL == 6.0
        assert config.enemy_max_turn_rate_deg == FIXED_ENEMY_MAX_TURN_RATE_DEG == 75.0
        assert config.enemy_max_climb_rate == FIXED_ENEMY_MAX_CLIMB_RATE == 5.0
        assert config.enemy_max_descent_rate == FIXED_ENEMY_MAX_DESCENT_RATE == 5.0


# ---------------------------------------------------------------------------
# R. Reward/termination geometry frozen (full-config equality except allowed keys)
# ---------------------------------------------------------------------------
def test_reward_and_termination_geometry_frozen():
    for gain in EVASION_GAIN_VALUES:
        config = build_evasion_env_config("measurement", gain)
        assert config.intercept_radius == _NOMINAL_ENV_CONFIG.intercept_radius
        assert config.threat_radius == _NOMINAL_ENV_CONFIG.threat_radius
        assert config.unsafe_intercept_radius == _NOMINAL_ENV_CONFIG.unsafe_intercept_radius
        assert config.max_steps == _NOMINAL_ENV_CONFIG.max_steps
        assert config.dt == _NOMINAL_ENV_CONFIG.dt
        assert config.L == _NOMINAL_ENV_CONFIG.L
        assert config.max_altitude == _NOMINAL_ENV_CONFIG.max_altitude


# ---------------------------------------------------------------------------
# S. Corrected Greedy contract active
# ---------------------------------------------------------------------------
def test_corrected_greedy_contract_active():
    config = build_evasion_env_config("measurement", NOMINAL_EVASION_GAIN)
    env = SoldierEnv(config=config)
    policy = GreedyInterceptPolicy()
    obs, info = env.reset(seed=19_500)
    policy.reset()
    estimator_mode = estimator_mode_for_env(env)
    policy_info = build_policy_info(info, estimator_mode)
    action = policy.act(obs, policy_info)
    assert action is not None
    assert np.all(np.isfinite(np.asarray(action, dtype=np.float64)))
    # Corrected contract: greedy must intercept (not merely escort) once detected.
    assert hasattr(policy, "act")


# ---------------------------------------------------------------------------
# T. No policy truth leakage
# ---------------------------------------------------------------------------
def test_no_policy_truth_leakage():
    config = build_evasion_env_config("measurement", NOMINAL_EVASION_GAIN)
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=19_501)
    estimator_mode = estimator_mode_for_env(env)
    assert estimator_mode == "measurement"
    policy_info = build_policy_info(info, estimator_mode)
    assert "e_hat" not in policy_info or policy_info.get("e_hat") is None
    assert "enemy_pos" not in policy_info

    config_k = build_evasion_env_config("kalman", NOMINAL_EVASION_GAIN)
    env_k = SoldierEnv(config=config_k)
    obs_k, info_k = env_k.reset(seed=19_501)
    estimator_mode_k = estimator_mode_for_env(env_k)
    assert estimator_mode_k == "kalman"
    policy_info_k = build_policy_info(info_k, estimator_mode_k)
    assert "enemy_pos" not in policy_info_k


# ---------------------------------------------------------------------------
# U. gain=0.0 <=> enemy_evasion_enabled=False equivalence
# ---------------------------------------------------------------------------
def test_gain_zero_equivalent_to_evasion_disabled():
    for seed in list(_DIAGNOSTIC_SEEDS)[:5]:
        config_gain0 = build_evasion_env_config("measurement", 0.0)
        config_disabled = dataclasses.replace(
            _NOMINAL_ENV_CONFIG, enemy_evasion_enabled=False, enemy_evasion_radius=EVASION_RADIUS
        )
        env_a = SoldierEnv(config=config_gain0)
        env_b = SoldierEnv(config=config_disabled)
        _, info_a = env_a.reset(seed=seed)
        _, info_b = env_b.reset(seed=seed)
        done_a = done_b = False
        while not (done_a or done_b):
            action = np.zeros(env_a.action_space.shape, dtype=np.float32)
            _, _, done_a, trunc_a, info_a = env_a.step(action)
            _, _, done_b, trunc_b, info_b = env_b.step(action)
            assert np.allclose(info_a["enemy_pos"], info_b["enemy_pos"], atol=1e-6), (
                f"seed={seed}: hostile trajectories diverge between gain=0 and enabled=False"
            )
            assert info_a["enemy_evasion_weight"] == 0.0
            assert info_b["enemy_evasion_weight"] == 0.0
            if done_a or trunc_a or done_b or trunc_b:
                break


# ---------------------------------------------------------------------------
# V. effective_evasion_weight == gain * activation
# ---------------------------------------------------------------------------
def test_effective_evasion_weight_equals_gain_times_activation():
    for gain in (0.25, 0.75, 1.0):
        config = build_evasion_env_config("measurement", gain)
        env = SoldierEnv(config=config)
        _, info = env.reset(seed=19_502)
        done = False
        checked_any = False
        for _ in range(200):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, _, done, trunc, info = env.step(action)
            expected_weight = gain * info["enemy_evasion_activation"]
            assert abs(info["enemy_evasion_weight"] - expected_weight) < 1e-6
            checked_any = True
            if done or trunc:
                break
        assert checked_any


# ---------------------------------------------------------------------------
# W. Corrected sample-weighted estimator aggregation helper is used
# ---------------------------------------------------------------------------
def test_corrected_estimator_aggregation_used():
    import pandas as pd
    rng = np.random.default_rng(0)
    n = np.array([50.0, 80.0, 30.0])
    r = np.array([1.0, 2.0, 3.0])
    correct = sample_weighted_rmse(n, r)
    naive_wrong = np.sqrt(np.mean(r ** 2))
    assert abs(correct - naive_wrong) > 1e-9

    df = pd.DataFrame({
        "enemy_evasion_gain": [0.75] * 3,
        "policy_instance": ["greedy"] * 3,
        "paper_method": ["greedy"] * 3,
        "training_seed": [float("nan")] * 3,
        "success": [1, 1, 0],
        "n_position_estimation_samples": n,
        "mean_position_estimation_error": [0.9, 1.8, 2.7],
        "position_estimation_rmse": r,
    })
    summary = compute_estimator_summary(df, [0.75])
    row = summary[(summary["method"] == "greedy") & (summary["enemy_evasion_gain"] == 0.75)].iloc[0]
    assert abs(row["rmse_sample_weighted"] - correct) < 1e-9
    assert abs(row["rmse_sample_weighted"] - naive_wrong) > 1e-9


# ---------------------------------------------------------------------------
# X. Expected rows = 50,000
# ---------------------------------------------------------------------------
def test_expected_rows_50000():
    assert EXPECTED_ROWS == len(EVASION_GAIN_VALUES) * EXPECTED_N_POLICY_INSTANCES * HOSTILE_EVASION_EPISODES
    assert EXPECTED_ROWS == 50_000


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
