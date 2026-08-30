"""Final-experiment-protocol correctness tests -- locked final nominal evaluation guardrails.

Script-style test consistent with the project's other test files. No
external test framework is used; each test function performs assertions
and the runner reports PASS/FAIL per test with a final summary.

Covers (per experiments/final_experiment_protocol.py's spec):
    A. Final nominal seeds exactly 20000..24999.
    B. 5000 unique seeds.
    C. No overlap with training/validation/diagnostic/robustness seeds.
    D. All 8 paper methods exist.
    E. All 6 PPO model files exist.
    F. All 6 model hashes are nonempty and unique.
    G. All PPO models have observation space 16 and action space 3.
    H. Direct models map only to measurement environments.
    I. Kalman models map only to Kalman environments.
    J. All non-estimator EnvConfig fields match (Direct vs Kalman).
    K. All prespecified comparison names exist.
    L. Results paths are under results/final_nominal/.
    M. No controller receives enemy_pos/enemy_vel through policy info.
    N. All three PPO training seeds are retained.

Run directly:
    python test_final_experiment_protocol.py
"""

from __future__ import annotations

import dataclasses

from experiments.training_protocol import TRAINING_SEEDS, VALIDATION_SEEDS, DIAGNOSTIC_SEEDS
from experiments.final_eval_utils import sha256_of_file
from experiments.final_experiment_protocol import (
    FINAL_NOMINAL_SEED_OFFSET,
    FINAL_NOMINAL_EPISODES,
    FINAL_NOMINAL_SEEDS,
    ROBUSTNESS_SEED_OFFSET,
    RESERVED_GAP_SEEDS,
    PPO_TRAINING_SEEDS,
    PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    POLICY_INSTANCES,
    PRIMARY_COMPARISONS,
    SECONDARY_COMPARISONS,
    ALL_COMPARISONS,
    FINAL_NOMINAL_OUTPUT_ROOT,
    assert_final_nominal_seed,
)


# ---------------------------------------------------------------------------
# A. Final nominal seeds exactly 20000..24999
# ---------------------------------------------------------------------------
def test_final_nominal_seed_range():
    assert FINAL_NOMINAL_SEED_OFFSET == 20_000
    assert FINAL_NOMINAL_EPISODES == 5_000
    assert list(FINAL_NOMINAL_SEEDS) == list(range(20_000, 25_000))
    assert min(FINAL_NOMINAL_SEEDS) == 20_000
    assert max(FINAL_NOMINAL_SEEDS) == 24_999


# ---------------------------------------------------------------------------
# B. 5000 unique seeds
# ---------------------------------------------------------------------------
def test_5000_unique_seeds():
    seeds = list(FINAL_NOMINAL_SEEDS)
    assert len(seeds) == 5000
    assert len(set(seeds)) == 5000


# ---------------------------------------------------------------------------
# C. No overlap with training/validation/diagnostic/robustness seeds
# ---------------------------------------------------------------------------
def test_no_overlap_with_other_seed_ranges():
    final_set = set(FINAL_NOMINAL_SEEDS)
    assert final_set.isdisjoint(set(TRAINING_SEEDS)), "overlaps training seeds"
    assert final_set.isdisjoint(set(VALIDATION_SEEDS)), "overlaps validation seeds"
    assert final_set.isdisjoint(set(DIAGNOSTIC_SEEDS)), "overlaps diagnostic seeds"
    assert final_set.isdisjoint(set(RESERVED_GAP_SEEDS)), "overlaps the reserved 25000-29999 gap"
    assert ROBUSTNESS_SEED_OFFSET == 30_000
    assert max(FINAL_NOMINAL_SEEDS) < ROBUSTNESS_SEED_OFFSET

    # Guard helper: rejects anything outside the locked range.
    assert_final_nominal_seed(20_000)
    assert_final_nominal_seed(24_999)
    for bad_seed in (19_999, 25_000, 30_000, 42):
        raised = False
        try:
            assert_final_nominal_seed(bad_seed)
        except ValueError:
            raised = True
        assert raised, f"assert_final_nominal_seed must reject seed {bad_seed}"


# ---------------------------------------------------------------------------
# D. All 8 paper methods exist
# ---------------------------------------------------------------------------
def test_all_8_paper_methods_exist():
    assert len(PAPER_METHODS) == 8
    assert set(PAPER_METHODS) == {
        "greedy", "pn", "lead", "kalman_greedy", "pn_kalman", "lead_kalman", "ppo", "ppo_kalman",
    }
    assert len(SCRIPTED_METHODS) == 6
    assert len(PPO_METHODS) == 2


# ---------------------------------------------------------------------------
# E. All 6 PPO model files exist
# ---------------------------------------------------------------------------
def test_all_6_ppo_model_files_exist():
    ppo_instances = [p for p in POLICY_INSTANCES if p.model_path is not None]
    assert len(ppo_instances) == 6
    for instance in ppo_instances:
        assert instance.model_path.exists(), f"missing frozen model: {instance.model_path}"
        assert instance.model_path.name == "best_model.zip", (
            f"{instance.policy_instance} must use best_model.zip, not {instance.model_path.name}"
        )
        assert "training_400k" in str(instance.model_path), (
            f"{instance.policy_instance} must come from the 400k campaign, got {instance.model_path}"
        )


# ---------------------------------------------------------------------------
# F. All 6 model hashes are nonempty and unique
# ---------------------------------------------------------------------------
def test_model_hashes_nonempty_and_unique():
    ppo_instances = [p for p in POLICY_INSTANCES if p.model_path is not None]
    hashes = [sha256_of_file(p.model_path) for p in ppo_instances]
    assert all(h and len(h) == 64 for h in hashes), "expected 64-char nonempty SHA-256 hex digests"
    assert len(set(hashes)) == len(hashes), "expected 6 unique model hashes (distinct trained models)"


# ---------------------------------------------------------------------------
# G. All PPO models have observation space 16 and action space 3
# ---------------------------------------------------------------------------
def test_ppo_model_io_shapes():
    from uav_defend.policies.rl.ppo_policy_wrapper import PPOPolicyWrapper
    from uav_defend.policies.rl_kalman.ppo_kalman_policy_wrapper import PPOKalmanPolicyWrapper

    for instance in POLICY_INSTANCES:
        if instance.model_path is None:
            continue
        wrapper_cls = PPOKalmanPolicyWrapper if instance.paper_method == "ppo_kalman" else PPOPolicyWrapper
        policy = wrapper_cls.load(str(instance.model_path), deterministic=True)
        obs_shape = policy.model.observation_space.shape
        act_shape = policy.model.action_space.shape
        assert obs_shape == (16,), f"{instance.policy_instance}: expected obs shape (16,), got {obs_shape}"
        assert act_shape == (3,), f"{instance.policy_instance}: expected action shape (3,), got {act_shape}"


# ---------------------------------------------------------------------------
# H, I. Direct models -> measurement env only; Kalman models -> Kalman env only
# ---------------------------------------------------------------------------
def test_direct_models_map_to_measurement_only():
    for instance in POLICY_INSTANCES:
        if instance.paper_method == "ppo":
            assert instance.estimator == "measurement"
            assert "direct" in str(instance.model_path)


def test_kalman_models_map_to_kalman_only():
    for instance in POLICY_INSTANCES:
        if instance.paper_method == "ppo_kalman":
            assert instance.estimator == "kalman"
            assert str(instance.model_path).count("kalman") >= 1


# ---------------------------------------------------------------------------
# J. All non-estimator EnvConfig fields match (Direct vs Kalman)
# ---------------------------------------------------------------------------
def test_env_config_matches_except_estimator():
    from experiments.training_runner import build_env_config

    direct_dict = dataclasses.asdict(build_env_config("direct"))
    kalman_dict = dataclasses.asdict(build_env_config("kalman"))
    assert direct_dict["use_kalman_tracking"] is False
    assert kalman_dict["use_kalman_tracking"] is True
    direct_dict.pop("use_kalman_tracking")
    kalman_dict.pop("use_kalman_tracking")
    assert direct_dict == kalman_dict


# ---------------------------------------------------------------------------
# K. All prespecified comparison names exist
# ---------------------------------------------------------------------------
def test_prespecified_comparisons_exist():
    assert [c[0] for c in PRIMARY_COMPARISONS] == list("ABCDEFG")
    assert [c[0] for c in SECONDARY_COMPARISONS] == list("HIJ")
    assert len(ALL_COMPARISONS) == 10
    valid_methods = set(PAPER_METHODS)
    for name, a, b in ALL_COMPARISONS:
        assert a in valid_methods and b in valid_methods, f"comparison {name} references unknown method"


# ---------------------------------------------------------------------------
# L. Results paths are under results/final_nominal/
# ---------------------------------------------------------------------------
def test_output_root_under_results_final_nominal():
    parts = FINAL_NOMINAL_OUTPUT_ROOT.parts
    assert parts[-2:] == ("results", "final_nominal")


# ---------------------------------------------------------------------------
# M. No controller receives enemy_pos/enemy_vel through policy info
# ---------------------------------------------------------------------------
def test_no_ground_truth_in_policy_info():
    from uav_defend.policies.sanitize import build_policy_info

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
# N. All three PPO training seeds are retained
# ---------------------------------------------------------------------------
def test_all_three_ppo_training_seeds_retained():
    assert PPO_TRAINING_SEEDS == (42, 43, 44)
    for method in PPO_METHODS:
        seeds_for_method = sorted(
            p.training_seed for p in POLICY_INSTANCES if p.paper_method == method
        )
        assert seeds_for_method == [42, 43, 44], (
            f"{method}: expected all three training seeds retained, got {seeds_for_method}"
        )
    # No "best seed" selection anywhere in the protocol -- exactly 3 instances per PPO method.
    assert len([p for p in POLICY_INSTANCES if p.paper_method == "ppo"]) == 3
    assert len([p for p in POLICY_INSTANCES if p.paper_method == "ppo_kalman"]) == 3
    assert len(POLICY_INSTANCES) == 12


def main() -> int:
    tests = [
        ("A. Final nominal seeds exactly 20000..24999", test_final_nominal_seed_range),
        ("B. 5000 unique seeds", test_5000_unique_seeds),
        ("C. No overlap with training/validation/diagnostic/robustness seeds", test_no_overlap_with_other_seed_ranges),
        ("D. All 8 paper methods exist", test_all_8_paper_methods_exist),
        ("E. All 6 PPO model files exist", test_all_6_ppo_model_files_exist),
        ("F. All 6 model hashes nonempty and unique", test_model_hashes_nonempty_and_unique),
        ("G. PPO models have obs=16, action=3", test_ppo_model_io_shapes),
        ("H. Direct models map only to measurement", test_direct_models_map_to_measurement_only),
        ("I. Kalman models map only to Kalman", test_kalman_models_map_to_kalman_only),
        ("J. Non-estimator EnvConfig fields match", test_env_config_matches_except_estimator),
        ("K. Prespecified comparison names exist", test_prespecified_comparisons_exist),
        ("L. Results paths under results/final_nominal/", test_output_root_under_results_final_nominal),
        ("M. No enemy_pos/enemy_vel in policy info", test_no_ground_truth_in_policy_info),
        ("N. All three PPO training seeds retained", test_all_three_ppo_training_seeds_retained),
    ]

    print("=== Final Experiment Protocol Tests ===\n")
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
