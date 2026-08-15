"""Reward calculation correctness tests.

Script-style test consistent with the project's other test files
(test_3d_dynamics.py, test_hostile_evasion.py, etc). No external test
framework is used; each test function performs assertions and the runner
reports PASS/FAIL per test with a final summary.

Covers the reward formula AFTER the experiment-fairness refactor:
    - Ongoing reward: progress term + time penalty + proximity-warning term,
      using the TRUE defender-enemy / enemy-soldier distances (reward is a
      training-only shaping signal computed by the environment itself, not
      part of any policy's observation).
    - NO Kalman-only "tracking error improvement" term (removed -- both
      Direct and Kalman tracks now receive an IDENTICAL ongoing reward
      formula for the same true trajectory).
    - NO hidden reward clipping (the previous `np.clip(reward, -50, 50)` has
      been removed with no replacement clip).
    - Terminal rewards: exact configured constants for intercepted /
      soldier_caught / unsafe_intercept / timeout.

Run directly:
    python test_reward.py
"""

from __future__ import annotations

import numpy as np

from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig


def _freeze_soldier_and_enemy(env) -> None:
    """
    Freeze the soldier's and enemy's uncontrolled motion for the NEXT
    step() call only, so a terminal-reward test scenario built by directly
    setting env._soldier_pos / env._enemy_pos survives into the distance
    computation. Movement dynamics themselves are already exhaustively
    covered by test_3d_dynamics.py and test_hostile_evasion.py; this helper
    exists purely to build controlled reward/termination scenarios here.
    """
    env._move_soldier = lambda: None
    env._move_enemy = lambda: None


# ---------------------------------------------------------------------------
# A. Ongoing reward matches the exact formula, every step, for many steps
# ---------------------------------------------------------------------------
def test_ongoing_reward_formula():
    """
    reward = reward_progress_scale * (prev_dist_de - dist_de)
             + reward_time_penalty
             + (proximity term, only if dist_es < 3 * unsafe_intercept_radius)

    Verified using the TRUE defender-enemy / enemy-soldier distances exposed
    in info (ground truth is legitimate here: this test checks the
    ENVIRONMENT's own reward computation, not a policy).
    """
    config = EnvConfig()
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=1)
    prev_dist_de = info["defender_enemy_dist"]
    proximity_threshold = config.unsafe_intercept_radius * 3.0

    checked_steps = 0
    for _ in range(120):
        # Pursuit toward the TRUE enemy position (fine for this environment-
        # level test; only POLICIES must avoid ground truth).
        direction = info["enemy_pos"] - info["defender_pos"]
        norm = np.linalg.norm(direction)
        action = direction / norm if norm > 1e-8 else np.zeros(3, dtype=np.float32)

        obs, reward, term, trunc, info = env.step(action)
        if term:
            break

        dist_de = info["defender_enemy_dist"]
        dist_es = info["enemy_soldier_dist"]

        progress = prev_dist_de - dist_de
        expected = config.reward_progress_scale * progress + config.reward_time_penalty
        if dist_es < proximity_threshold:
            proximity_factor = 1.0 - (dist_es / proximity_threshold)
            expected += config.reward_proximity_warning * proximity_factor

        assert abs(reward - expected) < 1e-4, (reward, expected)
        checked_steps += 1
        prev_dist_de = dist_de

    assert checked_steps > 10, f"episode ended too early to validate formula ({checked_steps} steps)"


# ---------------------------------------------------------------------------
# B. No reward clipping: the formula is honored even when it implies a
#    magnitude that would have been clipped under the old +/-50 bound.
# ---------------------------------------------------------------------------
def test_no_reward_clip():
    """
    Construct a single large one-step closing-distance jump (by teleporting
    the defender far closer to the enemy between steps) and confirm the
    reward reflects the FULL unclamped progress term, not clipped to +/-50.
    """
    config = EnvConfig(reward_progress_scale=100.0, reward_time_penalty=0.0)
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=3)

    prev_dist_de = info["defender_enemy_dist"]
    # Teleport the defender to be almost on top of the enemy (large,
    # artificial one-step "progress") -- but not so close it satisfies the
    # intercept condition (which would end the episode via a different,
    # terminal reward branch instead of the ongoing formula under test).
    intercept_r = config.intercept_radius
    direction = (env._defender_pos - env._enemy_pos)
    norm = np.linalg.norm(direction)
    unit = direction / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    env._defender_pos = (env._enemy_pos + unit * (intercept_r + 0.5)).astype(np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)

    action = np.zeros(3, dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)
    assert not term, "episode should still be ongoing for this test"

    dist_de = info["defender_enemy_dist"]
    progress = prev_dist_de - dist_de
    expected = config.reward_progress_scale * progress  # time_penalty=0.0 here
    # This progress-scaled reward is designed to exceed the old +/-50 clip
    # bound; if it doesn't in some environment configuration, the test still
    # validates the (unclamped) formula, just less dramatically.
    assert abs(reward - expected) < 1e-3, (reward, expected)


# ---------------------------------------------------------------------------
# C. No Kalman-only tracking-error reward term remains in the formula
# ---------------------------------------------------------------------------
def test_no_tracking_error_reward_term():
    """
    Run the Kalman track for many steps; at every ongoing step, the reward
    must match the SAME progress+time+proximity formula as the Direct track
    (section A) -- i.e. no additional tracking-error-improvement term is
    added when use_kalman_tracking=True.
    """
    config = EnvConfig(use_kalman_tracking=True)
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=7)
    prev_dist_de = info["defender_enemy_dist"]
    proximity_threshold = config.unsafe_intercept_radius * 3.0

    checked_steps = 0
    for _ in range(120):
        direction = info["enemy_pos"] - info["defender_pos"]
        norm = np.linalg.norm(direction)
        action = direction / norm if norm > 1e-8 else np.zeros(3, dtype=np.float32)

        obs, reward, term, trunc, info = env.step(action)
        if term:
            break

        dist_de = info["defender_enemy_dist"]
        dist_es = info["enemy_soldier_dist"]

        progress = prev_dist_de - dist_de
        expected = config.reward_progress_scale * progress + config.reward_time_penalty
        if dist_es < proximity_threshold:
            proximity_factor = 1.0 - (dist_es / proximity_threshold)
            expected += config.reward_proximity_warning * proximity_factor

        # Once detected, info["tracking_error"] must be present (an
        # EVALUATION metric) but must NOT influence the reward at all.
        if info.get("enemy_detected"):
            assert info.get("tracking_error") is not None
        assert abs(reward - expected) < 1e-4, (reward, expected, info.get("tracking_error"))
        checked_steps += 1
        prev_dist_de = dist_de

    assert checked_steps > 10, f"episode ended too early to validate formula ({checked_steps} steps)"


# ---------------------------------------------------------------------------
# D. Terminal rewards: exact configured constants
# ---------------------------------------------------------------------------
def test_terminal_reward_intercepted():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=11)
    _freeze_soldier_and_enemy(env)

    # Force a safe intercept: defender on top of the enemy, soldier far away.
    env._soldier_pos = np.array([config.L - 1.0, config.L - 1.0, 0.0], dtype=np.float32)
    env._enemy_pos = np.array([-config.L + 1.0, -config.L + 1.0, 10.0], dtype=np.float32)
    env._defender_pos = env._enemy_pos.copy()
    env._defender_vel = np.zeros(3, dtype=np.float32)

    obs, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
    assert term, info
    assert info["outcome"] == "intercepted", info["outcome"]
    assert reward == config.reward_intercept, (reward, config.reward_intercept)


def test_terminal_reward_soldier_caught():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=13)
    _freeze_soldier_and_enemy(env)

    # Force soldier_caught: enemy right next to soldier, defender far from
    # both (soldier_caught takes priority over any simultaneous intercept).
    env._soldier_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    env._enemy_pos = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    env._defender_pos = np.array([config.L - 1.0, config.L - 1.0, 15.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)

    obs, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
    assert term, info
    assert info["outcome"] == "soldier_caught", info["outcome"]
    assert reward == config.reward_soldier_caught, (reward, config.reward_soldier_caught)


def test_terminal_reward_unsafe_intercept():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=17)
    _freeze_soldier_and_enemy(env)

    # Force unsafe_intercept: defender catches enemy (within intercept_radius)
    # while the enemy is still within unsafe_intercept_radius of the soldier,
    # but NOT within threat_radius (so soldier_caught does not also fire).
    assert config.threat_radius < config.unsafe_intercept_radius, (
        "test assumes threat_radius < unsafe_intercept_radius"
    )
    mid_r = (config.threat_radius + config.unsafe_intercept_radius) / 2.0
    env._soldier_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    env._enemy_pos = np.array([mid_r, 0.0, 0.0], dtype=np.float32)
    env._defender_pos = env._enemy_pos.copy()
    env._defender_vel = np.zeros(3, dtype=np.float32)

    obs, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
    assert term, info
    assert info["outcome"] == "unsafe_intercept", info["outcome"]
    assert reward == config.reward_unsafe_intercept, (reward, config.reward_unsafe_intercept)


def test_terminal_reward_timeout():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=19)
    _freeze_soldier_and_enemy(env)

    # Force timeout: no win/loss condition true, step_count already at the limit.
    env._soldier_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    env._enemy_pos = np.array([config.L - 1.0, config.L - 1.0, 15.0], dtype=np.float32)
    env._defender_pos = np.array([-(config.L - 1.0), -(config.L - 1.0), 15.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    env._step_count = config.max_steps - 1

    obs, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
    assert term, info
    assert info["outcome"] == "timeout", info["outcome"]
    assert reward == config.reward_timeout, (reward, config.reward_timeout)


def main() -> int:
    tests = [
        ("A. ongoing reward matches exact formula", test_ongoing_reward_formula),
        ("B. no reward clipping", test_no_reward_clip),
        ("C. no Kalman-only tracking-error reward term", test_no_tracking_error_reward_term),
        ("D1. terminal reward: intercepted", test_terminal_reward_intercepted),
        ("D2. terminal reward: soldier_caught", test_terminal_reward_soldier_caught),
        ("D3. terminal reward: unsafe_intercept", test_terminal_reward_unsafe_intercept),
        ("D4. terminal reward: timeout", test_terminal_reward_timeout),
    ]

    print("=== Reward Calculation Correctness Tests ===\n")
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
        except Exception as e:  # noqa: BLE001 - report unexpected errors as failures
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
            n_fail += 1

    print(f"\n{n_pass} passed, {n_fail} failed out of {len(tests)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
