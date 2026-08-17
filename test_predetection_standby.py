"""Pre-detection defender-standby correctness tests -- locked guardrails.

Script-style test consistent with the project's other test files (no pytest).

Covers (per the common-predetection-standby task spec):
    A. reset: defender_pos == soldier_pos exactly.
    B. reset: defender_vel == zero.
    C. soldier random walk changes x/y over repeated steps.
    D. soldier z always 0.
    E. same seed produces identical soldier random walk.
    F. while undetected: defender_pos == soldier_pos after every step.
    G. while undetected: defender_vel == zero.
    H. while undetected: arbitrarily different supplied actions produce identical states.
    I. while undetected: controller_action_executed == False.
    J. detection transition: just_detected == True.
    K. detection transition: defender remains co-located with soldier.
    L. detection transition: defender velocity remains zero.
    M. detection transition: supplied action was NOT executed.
    N. first step AFTER detection: controller action IS executed.
    O. first step after detection: subject to existing acceleration/speed/turn/vertical limits.
    P. different controller actions may diverge only AFTER detection.
    Q. same seed gives same detection step for different action sequences.
    R. Direct and Kalman tracks have identical pre-detection trajectories and detection step.
    S. hostile-related observation channels remain zero before detection.
    T. hostile measurement becomes available at first detection.
    U. Direct finite-difference hostile velocity remains invalid/zero on the first measurement.
    V. no hostile ground-truth leakage.
    W. hostile reactive evasion remains enabled and deterministic.
    X. legacy mode standby=False retains prior action-driven pre-detection behavior.
    Y. physical-limit tests continue to pass.
    Z. existing experiment-fairness/environment tests continue to pass under the new default.

Run directly:
    python test_predetection_standby.py
"""

from __future__ import annotations

import numpy as np

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from experiments.final_eval_utils import _check_physical_limits

_EPS = 1e-6


def _rollout_until_detected(config: EnvConfig, seed: int, action_fn):
    """Roll out until enemy_detected (or max_steps), recording per-step info dicts."""
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=seed)
    history = [info]
    step = 0
    while not info["enemy_detected"] and step < config.max_steps:
        action = action_fn(step)
        obs, reward, done, trunc, info = env.step(action)
        history.append(info)
        step += 1
        if done and not info["enemy_detected"]:
            break
    return env, history, obs


_ACTIONS = {
    "plus_x": lambda s: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "minus_x": lambda s: np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "plus_y": lambda s: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "zero": lambda s: np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "random": lambda rng: (lambda s: rng.uniform(-1, 1, size=3).astype(np.float32)),
}


def _make_random_action_fn(seed=0):
    rng = np.random.default_rng(seed)
    return lambda s: rng.uniform(-1, 1, size=3).astype(np.float32)


# ---------------------------------------------------------------------------
# A/B. reset semantics
# ---------------------------------------------------------------------------
def test_reset_defender_colocated_with_soldier():
    env = SoldierEnv(config=EnvConfig())
    obs, info = env.reset(seed=1)
    assert np.allclose(info["defender_pos"], info["soldier_pos"])


def test_reset_defender_velocity_zero():
    env = SoldierEnv(config=EnvConfig())
    obs, info = env.reset(seed=1)
    assert np.allclose(info["defender_vel"], 0.0)


# ---------------------------------------------------------------------------
# C/D/E. Soldier random walk unchanged
# ---------------------------------------------------------------------------
def test_soldier_random_walk_changes_xy():
    env = SoldierEnv(config=EnvConfig())
    obs, info = env.reset(seed=2)
    positions = [info["soldier_pos"].copy()]
    for _ in range(50):
        obs, reward, done, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        positions.append(info["soldier_pos"].copy())
        if done:
            break
    xy = np.array([p[:2] for p in positions])
    assert np.std(xy[:, 0]) > 1e-3 or np.std(xy[:, 1]) > 1e-3, "soldier should move stochastically in x/y"


def test_soldier_z_always_zero():
    env = SoldierEnv(config=EnvConfig())
    obs, info = env.reset(seed=2)
    assert info["soldier_pos"][2] == 0.0
    for _ in range(50):
        obs, reward, done, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        assert info["soldier_pos"][2] == 0.0
        if done:
            break


def test_same_seed_identical_soldier_walk():
    def walk(seed):
        env = SoldierEnv(config=EnvConfig())
        obs, info = env.reset(seed=seed)
        positions = [info["soldier_pos"].copy()]
        for _ in range(30):
            obs, reward, done, trunc, info = env.step(np.zeros(3, dtype=np.float32))
            positions.append(info["soldier_pos"].copy())
            if done:
                break
        return positions

    a = walk(42)
    b = walk(42)
    assert len(a) == len(b)
    for pa, pb in zip(a, b):
        assert np.allclose(pa, pb)


# ---------------------------------------------------------------------------
# F/G/I. Standby invariants while undetected
# ---------------------------------------------------------------------------
def test_standby_colocation_and_zero_velocity_and_action_not_executed():
    env = SoldierEnv(config=EnvConfig())
    obs, info = env.reset(seed=123)
    action_fn = _make_random_action_fn(1)
    step = 0
    while not info["enemy_detected"] and step < 2000:
        action = action_fn(step)
        obs, reward, done, trunc, info = env.step(action)
        step += 1
        if info["enemy_detected"]:
            break  # transition step checked separately (tests J-M)
        assert np.allclose(info["defender_pos"], info["soldier_pos"], atol=_EPS)
        assert np.allclose(info["defender_vel"], 0.0, atol=_EPS)
        assert info["controller_action_executed"] is False
        if done:
            break


# ---------------------------------------------------------------------------
# H/P/Q. Fairness across arbitrarily different action sequences
# ---------------------------------------------------------------------------
def test_fairness_across_action_sequences_before_detection():
    config = EnvConfig()
    seed = 123
    reference_env, reference_hist, _ = _rollout_until_detected(config, seed, _ACTIONS["plus_x"])
    reference_detection_step = len(reference_hist) - 1

    for name in ("minus_x", "plus_y", "zero"):
        env, hist, _ = _rollout_until_detected(config, seed, _ACTIONS[name])
        assert len(hist) - 1 == reference_detection_step, f"{name}: detection step differs"
        for a, b in zip(hist, reference_hist):
            assert np.allclose(a["soldier_pos"], b["soldier_pos"])
            assert np.allclose(a["defender_pos"], b["defender_pos"])
            assert np.allclose(a["defender_vel"], b["defender_vel"])
            assert np.allclose(a["enemy_pos"], b["enemy_pos"])
            assert np.allclose(a["enemy_vel"], b["enemy_vel"])
            assert a["enemy_detected"] == b["enemy_detected"]

    env, hist, _ = _rollout_until_detected(config, seed, _make_random_action_fn(7))
    assert len(hist) - 1 == reference_detection_step
    for a, b in zip(hist, reference_hist):
        assert np.allclose(a["defender_pos"], b["defender_pos"])


def test_divergence_only_after_detection():
    """Different actions must diverge in defender state AFTER the handoff step."""
    config = EnvConfig()
    seed = 123
    env_a, hist_a, _ = _rollout_until_detected(config, seed, _ACTIONS["plus_x"])
    env_b, hist_b, _ = _rollout_until_detected(config, seed, _ACTIONS["plus_y"])

    # Pre-detection (including the just_detected transition row): identical.
    for a, b in zip(hist_a, hist_b):
        assert np.allclose(a["defender_pos"], b["defender_pos"])
        assert np.allclose(a["defender_vel"], b["defender_vel"])

    # First post-detection controlled step for each: continue rolling out one more step.
    obs_a, r, d, t, info_a = env_a.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    obs_b, r, d, t, info_b = env_b.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert info_a["controller_action_executed"] is True
    assert info_b["controller_action_executed"] is True
    assert not np.allclose(info_a["defender_vel"], info_b["defender_vel"]), \
        "differing post-detection actions should produce differing defender velocity"


# ---------------------------------------------------------------------------
# J/K/L/M. Detection-transition semantics
# ---------------------------------------------------------------------------
def test_detection_transition_semantics():
    env = SoldierEnv(config=EnvConfig())
    obs, info = env.reset(seed=123)
    prev_detected = info["enemy_detected"]
    for _ in range(2000):
        obs, reward, done, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if info["just_detected"]:
            assert info["enemy_detected"] is True
            assert not prev_detected
            assert np.allclose(info["defender_pos"], info["soldier_pos"], atol=_EPS)
            assert np.allclose(info["defender_vel"], 0.0, atol=_EPS)
            assert info["controller_action_executed"] is False
            return
        prev_detected = info["enemy_detected"]
        if done:
            break
    raise AssertionError("episode ended before a detection transition occurred")


# ---------------------------------------------------------------------------
# N/O. First post-detection controlled step
# ---------------------------------------------------------------------------
def test_first_post_detection_step_executes_action_within_limits():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=123)
    while not info["just_detected"]:
        obs, reward, done, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert not done
    obs, reward, done, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert info["controller_action_executed"] is True
    assert info["defender_accel"] <= config.defender_max_accel + 1e-3
    assert info["defender_turn_rate"] <= config.defender_max_turn_rate_deg + 1e-3
    assert info["defender_speed"] <= config.v_d + 1e-3
    vz = info["defender_vel"][2]
    assert vz <= config.defender_max_climb_rate + 1e-3
    assert -vz <= config.defender_max_descent_rate + 1e-3
    # Must have actually accelerated away from rest (not an instantaneous jump to v_d).
    assert 0.0 < info["defender_speed"] < config.v_d - 1e-6


# ---------------------------------------------------------------------------
# R. Direct vs Kalman pre-detection equality
# ---------------------------------------------------------------------------
def test_direct_kalman_predetection_equality():
    seed = 123
    action_fn = _make_random_action_fn(3)
    _, hist_direct, _ = _rollout_until_detected(EnvConfig(use_kalman_tracking=False), seed, action_fn)
    _, hist_kalman, _ = _rollout_until_detected(EnvConfig(use_kalman_tracking=True), seed, action_fn)
    assert len(hist_direct) == len(hist_kalman)
    for a, b in zip(hist_direct, hist_kalman):
        assert np.allclose(a["soldier_pos"], b["soldier_pos"])
        assert np.allclose(a["defender_pos"], b["defender_pos"])
        assert np.allclose(a["enemy_pos"], b["enemy_pos"])
        assert np.allclose(a["enemy_vel"], b["enemy_vel"])
        assert a["enemy_detected"] == b["enemy_detected"]


# ---------------------------------------------------------------------------
# S/T/U/V. Observation/estimator semantics before and at first detection
# ---------------------------------------------------------------------------
def test_hostile_observation_channels_zero_before_detection():
    env = SoldierEnv(config=EnvConfig())
    obs, info = env.reset(seed=123)
    assert np.allclose(obs[10:16], 0.0)
    for _ in range(2000):
        obs, reward, done, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        if info["enemy_detected"]:
            break
        assert np.allclose(obs[10:16], 0.0)
        if done:
            break


def test_hostile_measurement_available_at_first_detection():
    env = SoldierEnv(config=EnvConfig())
    obs, info = env.reset(seed=123)
    for _ in range(2000):
        obs, reward, done, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        if info["just_detected"]:
            assert info["enemy_measurement"] is not None
            assert info["e_hat"] is not None
            return
        if done:
            break
    raise AssertionError("detection never occurred")


def test_direct_finite_difference_velocity_invalid_on_first_measurement():
    env = SoldierEnv(config=EnvConfig(use_kalman_tracking=False))
    obs, info = env.reset(seed=123)
    for _ in range(2000):
        obs, reward, done, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        if info["just_detected"]:
            assert info["enemy_measurement_velocity_valid"] is False
            assert np.allclose(info["enemy_measurement_velocity"], 0.0)
            return
        if done:
            break
    raise AssertionError("detection never occurred")


def test_no_ground_truth_leakage_in_observation():
    """Observation must never reveal enemy_pos before detection (obs stays zero); after
    detection, hostile channels reflect the measurement/estimate, not raw enemy_pos exactly."""
    env = SoldierEnv(config=EnvConfig(measurement_var=0.5))
    obs, info = env.reset(seed=123)
    for _ in range(2000):
        obs, reward, done, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        if info["enemy_detected"]:
            hostile_pos_component = obs[10:13]
            true_enemy_norm = np.array([info["enemy_pos"][0] / env.config.L,
                                          info["enemy_pos"][1] / env.config.L,
                                          info["enemy_pos"][2] / env.config.max_altitude])
            # With nonzero measurement noise, the observed value should not be
            # bit-identical to the (never-exposed) ground truth normalization.
            assert not np.allclose(hostile_pos_component, true_enemy_norm, atol=1e-12)
            return
        if done:
            break
    raise AssertionError("detection never occurred")


# ---------------------------------------------------------------------------
# W. Hostile reactive evasion remains enabled and deterministic
# ---------------------------------------------------------------------------
def test_hostile_evasion_enabled_and_deterministic():
    config = EnvConfig(enemy_evasion_enabled=True, enemy_evasion_gain=0.75, enemy_evasion_radius=20.0)

    def run(seed):
        env = SoldierEnv(config=config)
        obs, info = env.reset(seed=seed)
        activations = [info.get("enemy_evasion_activation", 0.0)]
        for _ in range(50):
            obs, reward, done, trunc, info = env.step(np.zeros(3, dtype=np.float32))
            activations.append(info["enemy_evasion_activation"])
            if done:
                break
        return activations

    a = run(123)
    b = run(123)
    assert a == b
    assert max(a) > 0.0, "evasion should activate at least once given R_evasion=20 > R_det=15"


# ---------------------------------------------------------------------------
# X. Legacy mode (standby=False) retains prior action-driven behavior
# ---------------------------------------------------------------------------
def test_legacy_mode_action_driven_predetection():
    config = EnvConfig(defender_standby_until_detection=False)
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=123)
    assert np.allclose(info["defender_pos"], info["soldier_pos"])  # still co-located at reset
    moved_independently = False
    for _ in range(50):
        obs, reward, done, trunc, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert info["controller_action_executed"] is True
        if not np.allclose(info["defender_pos"], info["soldier_pos"], atol=1e-3):
            moved_independently = True
        if done:
            break
    assert moved_independently, "legacy mode should allow the defender to move independently of the soldier"


def test_legacy_vs_standby_differ():
    config_standby = EnvConfig(defender_standby_until_detection=True)
    config_legacy = EnvConfig(defender_standby_until_detection=False)
    env_s = SoldierEnv(config=config_standby)
    env_l = SoldierEnv(config=config_legacy)
    obs_s, info_s = env_s.reset(seed=123)
    obs_l, info_l = env_l.reset(seed=123)
    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(5):
        obs_s, r, d, t, info_s = env_s.step(action)
        obs_l, r, d, t, info_l = env_l.step(action)
    assert not np.allclose(info_s["defender_pos"], info_l["defender_pos"]), \
        "standby and legacy pre-detection defender trajectories should differ"


# ---------------------------------------------------------------------------
# Y. Physical-limit checks continue to pass under standby
# ---------------------------------------------------------------------------
def test_physical_limits_hold_under_standby():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=123)
    _check_physical_limits(info, config)
    action_fn = _make_random_action_fn(9)
    for step in range(300):
        obs, reward, done, trunc, info = env.step(action_fn(step))
        _check_physical_limits(info, config)
        if done:
            break


# ---------------------------------------------------------------------------
# Z. Existing experiment-fairness/environment tests continue to pass
# ---------------------------------------------------------------------------
def test_existing_fairness_and_leakage_tests_still_pass():
    import test_experiment_fairness as fairness_mod
    fairness_mod.test_direct_kalman_equivalence()
    fairness_mod.test_no_ground_truth_leakage_all_controllers()
    fairness_mod.test_rng_streams_independent_of_detection_radius()
    fairness_mod.test_matched_seed_reproducibility()


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
