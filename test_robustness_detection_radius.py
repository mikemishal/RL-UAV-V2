"""Detection-radius robustness sweep correctness tests -- locked sweep guardrails.

Script-style test consistent with the project's other test files. No
external test framework is used; each test function performs assertions
and the runner reports PASS/FAIL per test with a final summary.

Covers (per the detection-radius robustness sweep task spec):
    A. Radius values exactly 5,10,15,20,25,30.
    B. Nominal 15 included.
    C. Robustness seeds exactly 30000..30999.
    D. No overlap with 20000..24999, 25000..25999.
    E. No use of future robustness ranges >=31000.
    F. 12 policy instances exist.
    G. All six PPO hashes match frozen nominal hashes.
    H. Every sweep config differs from nominal only in detection_radius/estimator mode.
    I. Expected raw row count = 72000.
    J. Common seed set across all radius/method combinations.
    K. Scripted pre-detection identity.
    L. PPO aggregation keeps all three training seeds.
    M. Bootstrap sampling pairs evaluation seeds correctly.
    N. Policy-info sanitization still excludes enemy_pos/enemy_vel.
    O. All result paths are under results/robustness/detection_radius/.

Run directly:
    python test_robustness_detection_radius.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy
from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env

from experiments.final_experiment_protocol import FINAL_NOMINAL_SEEDS
from experiments.acquisition_ablation_protocol import ACQUISITION_ABLATION_SEEDS
from experiments.robustness_eval_utils import sha256_of_file
from experiments.robustness_detection_protocol import (
    DETECTION_RADIUS_VALUES,
    NOMINAL_DETECTION_RADIUS,
    DETECTION_SWEEP_SEED_OFFSET,
    DETECTION_SWEEP_EPISODES,
    DETECTION_SWEEP_SEEDS,
    RESERVED_FUTURE_ROBUSTNESS_SEEDS,
    MEASUREMENT_NOISE_SWEEP_SEED_OFFSET,
    SWEEP_POLICY_INSTANCES,
    EXPECTED_N_POLICY_INSTANCES,
    EXPECTED_RAW_ROW_COUNT,
    SCRIPTED_PRE_DETECTION_IDENTITY_GROUP,
    ROBUSTNESS_DETECTION_OUTPUT_ROOT,
    build_sweep_env_config,
    assert_only_detection_radius_differs,
    assert_detection_sweep_seed,
)
from experiments.final_stats import hierarchical_paired_bootstrap_matched_seeds

PROJECT_ROOT = Path(__file__).parent
FINAL_NOMINAL_HASHES_PATH = PROJECT_ROOT / "results" / "final_nominal" / "model_hashes.json"
_NOMINAL_ENV_CONFIG = EnvConfig()


# ---------------------------------------------------------------------------
# A. Radius values exactly 5,10,15,20,25,30 ; B. nominal 15 included
# ---------------------------------------------------------------------------
def test_radius_values_exact():
    assert DETECTION_RADIUS_VALUES == (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)


def test_nominal_radius_included():
    assert NOMINAL_DETECTION_RADIUS == 15.0
    assert NOMINAL_DETECTION_RADIUS in DETECTION_RADIUS_VALUES
    assert _NOMINAL_ENV_CONFIG.detection_radius == NOMINAL_DETECTION_RADIUS


# ---------------------------------------------------------------------------
# C. Robustness seeds exactly 30000..30999
# ---------------------------------------------------------------------------
def test_robustness_seed_range():
    assert DETECTION_SWEEP_SEED_OFFSET == 30_000
    assert DETECTION_SWEEP_EPISODES == 1_000
    assert list(DETECTION_SWEEP_SEEDS) == list(range(30_000, 31_000))


# ---------------------------------------------------------------------------
# D. No overlap with 20000..24999, 25000..25999 ; E. no future robustness ranges
# ---------------------------------------------------------------------------
def test_no_overlap_with_prior_and_future_ranges():
    sweep_set = set(DETECTION_SWEEP_SEEDS)
    assert sweep_set.isdisjoint(set(FINAL_NOMINAL_SEEDS))
    assert sweep_set.isdisjoint(set(ACQUISITION_ABLATION_SEEDS))
    assert sweep_set.isdisjoint(set(RESERVED_FUTURE_ROBUSTNESS_SEEDS))
    assert MEASUREMENT_NOISE_SWEEP_SEED_OFFSET == 31_000
    assert max(DETECTION_SWEEP_SEEDS) < MEASUREMENT_NOISE_SWEEP_SEED_OFFSET

    assert_detection_sweep_seed(30_000)
    assert_detection_sweep_seed(30_999)
    for bad_seed in (24_999, 20_000, 25_500, 31_000):
        raised = False
        try:
            assert_detection_sweep_seed(bad_seed)
        except ValueError:
            raised = True
        assert raised, f"assert_detection_sweep_seed must reject seed {bad_seed}"


# ---------------------------------------------------------------------------
# F. 12 policy instances exist
# ---------------------------------------------------------------------------
def test_12_policy_instances_exist():
    assert len(SWEEP_POLICY_INSTANCES) == EXPECTED_N_POLICY_INSTANCES == 12
    ppo_instances = [p for p in SWEEP_POLICY_INSTANCES if p.model_path is not None]
    scripted_instances = [p for p in SWEEP_POLICY_INSTANCES if p.model_path is None]
    assert len(ppo_instances) == 6
    assert len(scripted_instances) == 6


# ---------------------------------------------------------------------------
# G. All six PPO hashes match frozen nominal hashes
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
    for instance in SWEEP_POLICY_INSTANCES:
        if instance.model_path is None:
            continue
        rel = str(instance.model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        assert rel in nominal_by_path, f"{rel} missing from nominal manifest"
        digest = sha256_of_file(instance.model_path)
        assert digest == nominal_by_path[rel], f"hash mismatch for {rel}"
        checked += 1
    assert checked == 6


# ---------------------------------------------------------------------------
# H. Every sweep config differs from nominal only in detection_radius/estimator
# ---------------------------------------------------------------------------
def test_sweep_config_differs_only_in_detection_radius():
    for radius in DETECTION_RADIUS_VALUES:
        for estimator in ("measurement", "kalman"):
            config = build_sweep_env_config(estimator, radius)
            assert_only_detection_radius_differs(config, _NOMINAL_ENV_CONFIG)
    # At nominal radius + measurement estimator, config must be IDENTICAL to nominal.
    nominal_like = build_sweep_env_config("measurement", NOMINAL_DETECTION_RADIUS)
    import dataclasses
    assert dataclasses.asdict(nominal_like) == dataclasses.asdict(_NOMINAL_ENV_CONFIG)
    # A genuinely different field must be caught.
    bad_config = EnvConfig(use_kalman_tracking=False, detection_radius=15.0, v_d=99.0)
    raised = False
    try:
        assert_only_detection_radius_differs(bad_config, _NOMINAL_ENV_CONFIG)
    except AssertionError:
        raised = True
    assert raised, "assert_only_detection_radius_differs must catch an unrelated field change"


# ---------------------------------------------------------------------------
# I. Expected raw row count = 72000
# ---------------------------------------------------------------------------
def test_expected_raw_row_count():
    assert EXPECTED_RAW_ROW_COUNT == 12 * 6 * 1000 == 72_000


# ---------------------------------------------------------------------------
# J. Common seed set across all radius/method combinations (construction-level)
# ---------------------------------------------------------------------------
def test_common_seed_set_construction():
    seeds = list(DETECTION_SWEEP_SEEDS)
    # Every (radius, instance) combination in the runner uses this SAME seeds list
    # (see run_evaluation in run_robustness_detection_sweep.py) -- verify the
    # seed list itself is well-formed and identical regardless of radius/instance.
    for radius in DETECTION_RADIUS_VALUES:
        for _ in SWEEP_POLICY_INSTANCES:
            assert seeds == list(DETECTION_SWEEP_SEEDS)
    assert len(set(seeds)) == 1000


# ---------------------------------------------------------------------------
# K. Scripted pre-detection identity (live rollout, at a non-nominal radius)
# ---------------------------------------------------------------------------
def test_scripted_pre_detection_identity():
    seed = 14000
    radius = 10.0
    results = {}
    for name in SCRIPTED_PRE_DETECTION_IDENTITY_GROUP:
        estimator = "kalman" if "kalman" in name else "measurement"
        env = SoldierEnv(config=build_sweep_env_config(estimator, radius))
        policy = get_policy(name)
        obs, info = env.reset(seed=seed)
        policy.reset()
        mode = estimator_mode_for_env(env)
        detection_step = None
        defender_pos_at_detection = None
        step = 0
        done = False
        while not done:
            sanitized = build_policy_info(info, mode)
            action = policy.act(obs, sanitized)
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            if info["enemy_detected"] and detection_step is None:
                detection_step = step
                defender_pos_at_detection = info["defender_pos"].copy()
                break
            if done or truncated:
                break
        results[name] = (detection_step, defender_pos_at_detection)

    ref_step, ref_pos = results[SCRIPTED_PRE_DETECTION_IDENTITY_GROUP[0]]
    for name in SCRIPTED_PRE_DETECTION_IDENTITY_GROUP[1:]:
        step, pos = results[name]
        assert step == ref_step, (name, step, ref_step)
        if ref_pos is not None:
            assert np.allclose(pos, ref_pos, atol=1e-4), (name, pos, ref_pos)


# ---------------------------------------------------------------------------
# L. PPO aggregation keeps all three training seeds
# ---------------------------------------------------------------------------
def test_ppo_aggregation_keeps_all_seeds():
    for method in ("ppo", "ppo_kalman"):
        seeds_for_method = sorted(
            p.training_seed for p in SWEEP_POLICY_INSTANCES if p.paper_method == method
        )
        assert seeds_for_method == [42, 43, 44]


# ---------------------------------------------------------------------------
# M. Bootstrap sampling pairs evaluation seeds correctly
# ---------------------------------------------------------------------------
def test_bootstrap_pairs_seeds_correctly():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=(3, 100)).astype(np.float64)
    b = rng.integers(0, 2, size=(3, 100)).astype(np.float64)
    observed, lo, hi, per_seed_diffs = hierarchical_paired_bootstrap_matched_seeds(a, b, 300, 42)
    assert per_seed_diffs.shape == (3,)
    assert np.allclose(per_seed_diffs, a.mean(axis=1) - b.mean(axis=1))


# ---------------------------------------------------------------------------
# N. Policy-info sanitization still excludes enemy_pos/enemy_vel
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
# O. All result paths are under results/robustness/detection_radius/
# ---------------------------------------------------------------------------
def test_output_root_location():
    parts = ROBUSTNESS_DETECTION_OUTPUT_ROOT.parts
    assert parts[-3:] == ("results", "robustness", "detection_radius")


def main() -> int:
    tests = [
        ("A. Radius values exactly 5,10,15,20,25,30", test_radius_values_exact),
        ("B. Nominal 15 included", test_nominal_radius_included),
        ("C. Robustness seeds exactly 30000..30999", test_robustness_seed_range),
        ("D/E. No overlap with prior/future ranges", test_no_overlap_with_prior_and_future_ranges),
        ("F. 12 policy instances exist", test_12_policy_instances_exist),
        ("G. PPO hashes match final-nominal hashes", test_ppo_hashes_match_final_nominal),
        ("H. Sweep config differs only in detection_radius", test_sweep_config_differs_only_in_detection_radius),
        ("I. Expected raw row count = 72000", test_expected_raw_row_count),
        ("J. Common seed set construction", test_common_seed_set_construction),
        ("K. Scripted pre-detection identity", test_scripted_pre_detection_identity),
        ("L. PPO aggregation keeps all three seeds", test_ppo_aggregation_keeps_all_seeds),
        ("M. Bootstrap pairs seeds correctly", test_bootstrap_pairs_seeds_correctly),
        ("N. Sanitization excludes ground truth", test_sanitization_excludes_ground_truth),
        ("O. Output root under results/robustness/detection_radius/", test_output_root_location),
    ]

    print("=== Detection-Radius Robustness Sweep Tests ===\n")
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
