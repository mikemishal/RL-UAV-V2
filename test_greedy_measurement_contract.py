"""Greedy measurement-contract correctness tests (software-bug correction).

Positive semantic tests verifying Direct Greedy and Kalman-Greedy actually
DEPEND on their legitimate hostile-position estimate, not merely that they
don't leak ground truth (the older test_experiment_fairness.py tests proved
absence of leakage but did not catch that Direct Greedy was silently stuck
in escort mode -- see uav_defend/policies/baseline/greedy_intercept_policy.py's
module docstring for the full bug description).

Script-style test consistent with the project's other test files.

Covers:
    A. Direct Greedy pre-detection escorts soldier.
    B. Direct Greedy after detection pursues enemy_measurement.
    C. Changing enemy_measurement after detection changes Direct Greedy action.
    D. Direct Greedy does NOT require e_hat.
    E. Kalman-Greedy after detection pursues e_hat.
    F. Changing e_hat changes Kalman-Greedy action.
    G. Mutating enemy_pos/enemy_vel has no effect (not available via sanitized info).
    H. enemy_detected=True with neither enemy_measurement nor e_hat -> explicit error.
    I. Measurement-track sanitized info + Direct Greedy works end-to-end (live rollout).
    J. Kalman-track sanitized info + Kalman-Greedy works end-to-end (live rollout).

Run directly:
    python test_greedy_measurement_contract.py
"""

from __future__ import annotations

import numpy as np

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.baseline.greedy_intercept_policy import GreedyInterceptPolicy
from uav_defend.policies.baseline.kalman_greedy_intercept_policy import KalmanGreedyInterceptPolicy
from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env

DEFENDER_POS = np.array([0.0, 0.0, 5.0])
SOLDIER_POS = np.array([-20.0, 0.0, 0.0])


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# A. Direct Greedy pre-detection escorts soldier
# ---------------------------------------------------------------------------
def test_direct_greedy_pre_detection_escorts_soldier():
    policy = GreedyInterceptPolicy()
    info = {
        "defender_pos": DEFENDER_POS, "soldier_pos": SOLDIER_POS,
        "enemy_detected": False,
    }
    action = policy.act(None, info)
    expected = _normalize(SOLDIER_POS - DEFENDER_POS)
    assert np.allclose(action, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# B. Direct Greedy after detection pursues enemy_measurement
# ---------------------------------------------------------------------------
def test_direct_greedy_post_detection_pursues_measurement():
    policy = GreedyInterceptPolicy()
    measurement = np.array([10.0, 5.0, 2.0])
    info = {
        "defender_pos": DEFENDER_POS, "soldier_pos": SOLDIER_POS,
        "enemy_detected": True, "enemy_measurement": measurement,
    }
    action = policy.act(None, info)
    expected = _normalize(measurement - DEFENDER_POS)
    assert np.allclose(action, expected, atol=1e-6), (action, expected)


# ---------------------------------------------------------------------------
# C. Changing enemy_measurement after detection changes Direct Greedy action
# ---------------------------------------------------------------------------
def test_direct_greedy_action_responds_to_measurement_change():
    policy = GreedyInterceptPolicy()
    info_a = {
        "defender_pos": DEFENDER_POS, "soldier_pos": SOLDIER_POS,
        "enemy_detected": True, "enemy_measurement": np.array([10.0, 0.0, 0.0]),
    }
    info_b = dict(info_a)
    info_b["enemy_measurement"] = np.array([-10.0, 8.0, 3.0])
    action_a = policy.act(None, info_a)
    action_b = policy.act(None, info_b)
    assert not np.allclose(action_a, action_b), "action must change when enemy_measurement changes"


# ---------------------------------------------------------------------------
# D. Direct Greedy does NOT require e_hat
# ---------------------------------------------------------------------------
def test_direct_greedy_does_not_require_e_hat():
    policy = GreedyInterceptPolicy()
    measurement = np.array([10.0, 0.0, 0.0])
    info = {
        "defender_pos": DEFENDER_POS, "soldier_pos": SOLDIER_POS,
        "enemy_detected": True, "enemy_measurement": measurement,
        # deliberately no 'e_hat' key at all
    }
    action = policy.act(None, info)  # must not raise
    expected = _normalize(measurement - DEFENDER_POS)
    assert np.allclose(action, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# E. Kalman-Greedy after detection pursues e_hat
# ---------------------------------------------------------------------------
def test_kalman_greedy_post_detection_pursues_e_hat():
    policy = KalmanGreedyInterceptPolicy()
    e_hat = np.array([12.0, -4.0, 1.0])
    info = {
        "defender_pos": DEFENDER_POS, "soldier_pos": SOLDIER_POS,
        "enemy_detected": True, "e_hat": e_hat,
    }
    action = policy.act(None, info)
    expected = _normalize(e_hat - DEFENDER_POS)
    assert np.allclose(action, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# F. Changing e_hat changes Kalman-Greedy action
# ---------------------------------------------------------------------------
def test_kalman_greedy_action_responds_to_e_hat_change():
    policy = KalmanGreedyInterceptPolicy()
    info_a = {
        "defender_pos": DEFENDER_POS, "soldier_pos": SOLDIER_POS,
        "enemy_detected": True, "e_hat": np.array([10.0, 0.0, 0.0]),
    }
    info_b = dict(info_a)
    info_b["e_hat"] = np.array([-5.0, 9.0, 2.0])
    action_a = policy.act(None, info_a)
    action_b = policy.act(None, info_b)
    assert not np.allclose(action_a, action_b), "action must change when e_hat changes"


# ---------------------------------------------------------------------------
# G. Mutating enemy_pos/enemy_vel has no effect (unavailable via sanitized info)
# ---------------------------------------------------------------------------
def test_ground_truth_mutation_has_no_effect():
    greedy = GreedyInterceptPolicy()
    measurement = np.array([10.0, 0.0, 0.0])
    info_a = {
        "defender_pos": DEFENDER_POS, "soldier_pos": SOLDIER_POS,
        "enemy_detected": True, "enemy_measurement": measurement,
    }
    info_b = dict(info_a)
    info_b["enemy_pos"] = np.array([9999.0, -8888.0, 7777.0])
    info_b["enemy_vel"] = np.array([-500.0, 300.0, 200.0])
    assert np.allclose(greedy.act(None, info_a), greedy.act(None, info_b), atol=1e-9)

    kalman_greedy = KalmanGreedyInterceptPolicy()
    e_hat = np.array([12.0, -4.0, 1.0])
    info_c = {
        "defender_pos": DEFENDER_POS, "soldier_pos": SOLDIER_POS,
        "enemy_detected": True, "e_hat": e_hat,
    }
    info_d = dict(info_c)
    info_d["enemy_pos"] = np.array([-9999.0, 8888.0, -7777.0])
    info_d["enemy_vel"] = np.array([500.0, -300.0, -200.0])
    assert np.allclose(kalman_greedy.act(None, info_c), kalman_greedy.act(None, info_d), atol=1e-9)


# ---------------------------------------------------------------------------
# H. enemy_detected=True with neither field -> explicit error
# ---------------------------------------------------------------------------
def test_greedy_raises_on_missing_estimate_after_detection():
    policy = GreedyInterceptPolicy()
    info = {
        "defender_pos": DEFENDER_POS, "soldier_pos": SOLDIER_POS,
        "enemy_detected": True,
        # neither enemy_measurement nor e_hat present
    }
    raised = False
    try:
        policy.act(None, info)
    except ValueError:
        raised = True
    assert raised, "GreedyInterceptPolicy must raise ValueError, not silently escort"


# ---------------------------------------------------------------------------
# I. Measurement-track sanitized info + Direct Greedy works end-to-end
# ---------------------------------------------------------------------------
def test_direct_greedy_live_rollout_switches_after_detection():
    env = SoldierEnv(config=EnvConfig(use_kalman_tracking=False))
    policy = GreedyInterceptPolicy()
    obs, info = env.reset(seed=15000)
    policy.reset()
    mode = estimator_mode_for_env(env)
    assert mode == "measurement"

    pre_detection_action = None
    post_detection_action = None
    done = False
    step = 0
    while not done and step < 500:
        sanitized = build_policy_info(info, mode)
        action = policy.act(obs, sanitized)
        if not info.get("enemy_detected", False) and pre_detection_action is None:
            pre_detection_action = action.copy()
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        if info["enemy_detected"] and post_detection_action is None:
            sanitized_post = build_policy_info(info, mode)
            post_detection_action = policy.act(obs, sanitized_post)
            break
        if done or truncated:
            break

    assert pre_detection_action is not None
    if post_detection_action is not None:
        assert not np.allclose(pre_detection_action, post_detection_action), (
            "Direct Greedy must change behavior after detection"
        )


# ---------------------------------------------------------------------------
# J. Kalman-track sanitized info + Kalman-Greedy works end-to-end
# ---------------------------------------------------------------------------
def test_kalman_greedy_live_rollout_switches_after_detection():
    env = SoldierEnv(config=EnvConfig(use_kalman_tracking=True))
    policy = KalmanGreedyInterceptPolicy()
    obs, info = env.reset(seed=15000)
    policy.reset()
    mode = estimator_mode_for_env(env)
    assert mode == "kalman"

    pre_detection_action = None
    post_detection_action = None
    done = False
    step = 0
    while not done and step < 500:
        sanitized = build_policy_info(info, mode)
        action = policy.act(obs, sanitized)
        if not info.get("enemy_detected", False) and pre_detection_action is None:
            pre_detection_action = action.copy()
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        if info["enemy_detected"] and post_detection_action is None:
            sanitized_post = build_policy_info(info, mode)
            post_detection_action = policy.act(obs, sanitized_post)
            break
        if done or truncated:
            break

    assert pre_detection_action is not None
    if post_detection_action is not None:
        assert not np.allclose(pre_detection_action, post_detection_action), (
            "Kalman-Greedy must change behavior after detection"
        )


def main() -> int:
    tests = [
        ("A. Direct Greedy pre-detection escorts soldier", test_direct_greedy_pre_detection_escorts_soldier),
        ("B. Direct Greedy post-detection pursues enemy_measurement", test_direct_greedy_post_detection_pursues_measurement),
        ("C. Direct Greedy action responds to measurement change", test_direct_greedy_action_responds_to_measurement_change),
        ("D. Direct Greedy does not require e_hat", test_direct_greedy_does_not_require_e_hat),
        ("E. Kalman-Greedy post-detection pursues e_hat", test_kalman_greedy_post_detection_pursues_e_hat),
        ("F. Kalman-Greedy action responds to e_hat change", test_kalman_greedy_action_responds_to_e_hat_change),
        ("G. Ground-truth mutation has no effect", test_ground_truth_mutation_has_no_effect),
        ("H. Greedy raises on missing estimate after detection", test_greedy_raises_on_missing_estimate_after_detection),
        ("I. Direct Greedy live rollout switches after detection", test_direct_greedy_live_rollout_switches_after_detection),
        ("J. Kalman-Greedy live rollout switches after detection", test_kalman_greedy_live_rollout_switches_after_detection),
    ]

    print("=== Greedy Measurement-Contract Tests ===\n")
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
