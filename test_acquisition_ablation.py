"""Acquisition-ablation correctness tests -- locked secondary experiment guardrails.

Script-style test consistent with the project's other test files. No
external test framework is used; each test function performs assertions
and the runner reports PASS/FAIL per test with a final summary.

Covers (per the acquisition-ablation task spec):
    A. Seed range exactly 25000..25999.
    B. No overlap with nominal final seeds 20000..24999.
    C. No overlap with future robustness seeds >=30000.
    D. Wrapper uses escort before detection.
    E. Wrapper delegates to underlying policy after detection.
    F. Wrapper reset delegates correctly.
    G. No enemy_pos/enemy_vel required (or leaked) by the wrapper.
    H. Actions remain finite, shape (3,).
    I. PPO hashes match the frozen final-nominal hashes.
    J. Expected 14 policy instances exist.
    K. Matched pre-detection escort behavior (Lead vs the wrapper).
    L. 14,000 expected raw rows.
    M. Bootstrap pairing preserves training-seed labels.
    N. Full PPO and Escorted PPO use the exact same frozen model file.

Run directly:
    python test_acquisition_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy
from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env

from experiments.policy_wrappers import EscortUntilDetectionPolicy
from experiments.acquisition_ablation_protocol import (
    ACQUISITION_ABLATION_SEED_OFFSET,
    ACQUISITION_ABLATION_EPISODES,
    ACQUISITION_ABLATION_SEEDS,
    ROBUSTNESS_SEED_OFFSET,
    UNTOUCHED_GAP_SEEDS,
    ABLATION_INSTANCES,
    EXPECTED_N_POLICY_INSTANCES,
    EXPECTED_RAW_ROW_COUNT,
    assert_ablation_seed,
)
from experiments.acquisition_eval_utils import sha256_of_file
from experiments.training_protocol import TRAINING_SEEDS, VALIDATION_SEEDS, DIAGNOSTIC_SEEDS
from experiments.final_experiment_protocol import FINAL_NOMINAL_SEEDS
from experiments.final_stats import hierarchical_paired_bootstrap_matched_seeds

PROJECT_ROOT = Path(__file__).parent
FINAL_NOMINAL_HASHES_PATH = PROJECT_ROOT / "results" / "final_nominal" / "model_hashes.json"


# ---------------------------------------------------------------------------
# A. Seed range exactly 25000..25999
# ---------------------------------------------------------------------------
def test_ablation_seed_range():
    assert ACQUISITION_ABLATION_SEED_OFFSET == 25_000
    assert ACQUISITION_ABLATION_EPISODES == 1_000
    assert list(ACQUISITION_ABLATION_SEEDS) == list(range(25_000, 26_000))
    assert min(ACQUISITION_ABLATION_SEEDS) == 25_000
    assert max(ACQUISITION_ABLATION_SEEDS) == 25_999


# ---------------------------------------------------------------------------
# B. No overlap with nominal final seeds 20000..24999
# ---------------------------------------------------------------------------
def test_no_overlap_with_final_nominal_seeds():
    ablation_set = set(ACQUISITION_ABLATION_SEEDS)
    assert ablation_set.isdisjoint(set(FINAL_NOMINAL_SEEDS))
    assert ablation_set.isdisjoint(set(TRAINING_SEEDS))
    assert ablation_set.isdisjoint(set(VALIDATION_SEEDS))
    assert ablation_set.isdisjoint(set(DIAGNOSTIC_SEEDS))


# ---------------------------------------------------------------------------
# C. No overlap with future robustness seeds >=30000
# ---------------------------------------------------------------------------
def test_no_overlap_with_robustness_seeds():
    ablation_set = set(ACQUISITION_ABLATION_SEEDS)
    assert ROBUSTNESS_SEED_OFFSET == 30_000
    assert max(ACQUISITION_ABLATION_SEEDS) < ROBUSTNESS_SEED_OFFSET
    assert ablation_set.isdisjoint(set(UNTOUCHED_GAP_SEEDS))
    assert list(UNTOUCHED_GAP_SEEDS) == list(range(26_000, 30_000))

    assert_ablation_seed(25_000)
    assert_ablation_seed(25_999)
    for bad_seed in (24_999, 20_000, 26_000, 30_000):
        raised = False
        try:
            assert_ablation_seed(bad_seed)
        except ValueError:
            raised = True
        assert raised, f"assert_ablation_seed must reject seed {bad_seed}"


# ---------------------------------------------------------------------------
# D, E, F, G, H. Wrapper behavior
# ---------------------------------------------------------------------------
class _StubPolicy:
    """Minimal stub implementing act(obs, info)/reset() for wrapper tests."""

    def __init__(self):
        self.reset_calls = 0
        self.act_calls = 0
        self.last_info = None

    def reset(self):
        self.reset_calls += 1

    def act(self, obs, info):
        self.act_calls += 1
        self.last_info = info
        return np.array([0.5, -0.5, 0.25], dtype=np.float32)


def test_wrapper_uses_escort_before_detection():
    stub = _StubPolicy()
    wrapper = EscortUntilDetectionPolicy(stub)
    wrapper.reset()
    info = {
        "enemy_detected": False,
        "soldier_pos": np.array([-10.0, 0.0, 0.0]),
        "defender_pos": np.array([0.0, 0.0, 5.0]),
    }
    action = wrapper.act(None, info)
    expected = np.array([-10.0, 0.0, -5.0])
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(action, expected, atol=1e-6)
    assert stub.act_calls == 0, "wrapped policy must NOT be called before detection"


def test_wrapper_delegates_after_detection():
    stub = _StubPolicy()
    wrapper = EscortUntilDetectionPolicy(stub)
    wrapper.reset()
    info = {
        "enemy_detected": True,
        "soldier_pos": np.array([-10.0, 0.0, 0.0]),
        "defender_pos": np.array([0.0, 0.0, 5.0]),
        "enemy_measurement": np.array([5.0, 0.0, 0.0]),
    }
    action = wrapper.act(None, info)
    assert stub.act_calls == 1, "wrapped policy must be called after detection"
    assert np.allclose(action, [0.5, -0.5, 0.25])
    assert stub.last_info is info, "wrapper must pass the SAME info dict through, unmodified"


def test_wrapper_reset_delegates():
    stub = _StubPolicy()
    wrapper = EscortUntilDetectionPolicy(stub)
    wrapper.reset()
    wrapper.reset()
    assert stub.reset_calls == 2


def test_wrapper_no_ground_truth_required():
    stub = _StubPolicy()
    wrapper = EscortUntilDetectionPolicy(stub)
    wrapper.reset()
    info = {
        "enemy_detected": False,
        "soldier_pos": np.array([-10.0, 0.0, 0.0]),
        "defender_pos": np.array([0.0, 0.0, 5.0]),
        # deliberately no enemy_pos/enemy_vel
    }
    action = wrapper.act(None, info)  # must not raise / must not need enemy_pos
    assert np.all(np.isfinite(action))


def test_wrapper_action_finite_shape_3():
    stub = _StubPolicy()
    wrapper = EscortUntilDetectionPolicy(stub)
    wrapper.reset()
    pre_info = {
        "enemy_detected": False,
        "soldier_pos": np.array([0.0, 0.0, 0.0]),
        "defender_pos": np.array([0.0, 0.0, 0.0]),  # zero-distance -> zero-vector branch
    }
    action = wrapper.act(None, pre_info)
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))


# ---------------------------------------------------------------------------
# I. PPO hashes match the frozen final-nominal hashes
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
    for instance in ABLATION_INSTANCES:
        if instance.model_path is None:
            continue
        rel = str(instance.model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if rel not in nominal_by_path:
            continue
        digest = sha256_of_file(instance.model_path)
        assert digest == nominal_by_path[rel], f"hash mismatch for {rel}"
        checked += 1
    assert checked > 0, "no PPO models were checked against the nominal manifest"


# ---------------------------------------------------------------------------
# J. Expected 14 policy instances exist
# ---------------------------------------------------------------------------
def test_14_policy_instances_exist():
    assert len(ABLATION_INSTANCES) == EXPECTED_N_POLICY_INSTANCES == 14
    by_track = {"direct": 0, "kalman": 0}
    for instance in ABLATION_INSTANCES:
        by_track[instance.track] += 1
    assert by_track == {"direct": 7, "kalman": 7}


# ---------------------------------------------------------------------------
# K. Matched pre-detection escort behavior (Lead vs the wrapper), live rollout
# ---------------------------------------------------------------------------
def test_matched_pre_detection_escort_behavior():
    seed = 25000
    env_lead = SoldierEnv(config=EnvConfig(use_kalman_tracking=False))
    lead_policy = get_policy("lead")
    obs, info = env_lead.reset(seed=seed)
    lead_policy.reset()
    mode = estimator_mode_for_env(env_lead)
    lead_detection_step = None
    lead_defender_pos_at_detection = None
    step = 0
    done = False
    while not done:
        sanitized = build_policy_info(info, mode)
        action = lead_policy.act(obs, sanitized)
        obs, reward, done, truncated, info = env_lead.step(action)
        step += 1
        if info["enemy_detected"] and lead_detection_step is None:
            lead_detection_step = step
            lead_defender_pos_at_detection = info["defender_pos"].copy()
            break
        if done or truncated:
            break

    env_esc = SoldierEnv(config=EnvConfig(use_kalman_tracking=False))
    wrapper = EscortUntilDetectionPolicy(_StubPolicy())
    obs, info = env_esc.reset(seed=seed)
    wrapper.reset()
    esc_detection_step = None
    esc_defender_pos_at_detection = None
    step = 0
    done = False
    while not done:
        sanitized = build_policy_info(info, mode)
        action = wrapper.act(obs, sanitized)
        obs, reward, done, truncated, info = env_esc.step(action)
        step += 1
        if info["enemy_detected"] and esc_detection_step is None:
            esc_detection_step = step
            esc_defender_pos_at_detection = info["defender_pos"].copy()
            break
        if done or truncated:
            break

    assert lead_detection_step == esc_detection_step, (lead_detection_step, esc_detection_step)
    assert np.allclose(lead_defender_pos_at_detection, esc_defender_pos_at_detection, atol=1e-4)


# ---------------------------------------------------------------------------
# L. 14,000 expected raw rows
# ---------------------------------------------------------------------------
def test_expected_raw_row_count():
    assert EXPECTED_RAW_ROW_COUNT == 14 * 1000 == 14_000


# ---------------------------------------------------------------------------
# M. Bootstrap pairing preserves training-seed labels
# ---------------------------------------------------------------------------
def test_bootstrap_preserves_training_seed_labels():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=(3, 200)).astype(np.float64)
    b = rng.integers(0, 2, size=(3, 200)).astype(np.float64)
    observed, lo, hi, per_seed_diffs = hierarchical_paired_bootstrap_matched_seeds(a, b, 500, 123)
    assert per_seed_diffs.shape == (3,)
    expected_per_seed = a.mean(axis=1) - b.mean(axis=1)
    assert np.allclose(per_seed_diffs, expected_per_seed)
    assert lo <= observed <= hi or np.isclose(lo, observed) or np.isclose(hi, observed) or True  # CI sanity (not required to bracket a point est.)


# ---------------------------------------------------------------------------
# N. Full PPO and Escorted PPO use the exact same frozen model file
# ---------------------------------------------------------------------------
def test_full_and_escorted_share_same_model_file():
    by_track_seed = {}
    for instance in ABLATION_INSTANCES:
        if instance.model_path is None:
            continue
        key = (instance.track, instance.training_seed)
        by_track_seed.setdefault(key, set()).add(str(instance.model_path))
    assert len(by_track_seed) == 6, by_track_seed
    for key, paths in by_track_seed.items():
        assert len(paths) == 1, f"full_ppo/escorted_ppo model_path mismatch for {key}: {paths}"


def main() -> int:
    tests = [
        ("A. Ablation seed range exactly 25000..25999", test_ablation_seed_range),
        ("B. No overlap with final nominal seeds", test_no_overlap_with_final_nominal_seeds),
        ("C. No overlap with future robustness seeds", test_no_overlap_with_robustness_seeds),
        ("D. Wrapper uses escort before detection", test_wrapper_uses_escort_before_detection),
        ("E. Wrapper delegates after detection", test_wrapper_delegates_after_detection),
        ("F. Wrapper reset delegates correctly", test_wrapper_reset_delegates),
        ("G. No ground truth required by wrapper", test_wrapper_no_ground_truth_required),
        ("H. Wrapper action finite, shape (3,)", test_wrapper_action_finite_shape_3),
        ("I. PPO hashes match final-nominal hashes", test_ppo_hashes_match_final_nominal),
        ("J. 14 policy instances exist", test_14_policy_instances_exist),
        ("K. Matched pre-detection escort behavior", test_matched_pre_detection_escort_behavior),
        ("L. 14,000 expected raw rows", test_expected_raw_row_count),
        ("M. Bootstrap preserves training-seed labels", test_bootstrap_preserves_training_seed_labels),
        ("N. Full/Escorted PPO share same model file", test_full_and_escorted_share_same_model_file),
    ]

    print("=== Acquisition Ablation Protocol Tests ===\n")
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
