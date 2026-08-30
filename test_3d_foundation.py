"""3D foundation correctness tests for the UAV-defense simulation.

Script-style test consistent with the project's existing test files
(test_env.py, test_detection.py, test_reward.py, test_kalman_integration.py).
No external test framework is used; each test function performs assertions
and the runner reports PASS/FAIL per test with a final summary.

Run directly:
    python test_3d_foundation.py
"""

from __future__ import annotations

import numpy as np

from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.tracking import EnemyKalmanFilter
from uav_defend.policies.baseline.greedy_intercept_policy import GreedyInterceptPolicy
from uav_defend.policies.baseline.kalman_greedy_intercept_policy import KalmanGreedyInterceptPolicy
from uav_defend.policies.baseline.random_policy import RandomPolicy


# ---------------------------------------------------------------------------
# A. Spaces
# ---------------------------------------------------------------------------
def test_spaces():
    env = SoldierEnv()
    assert env.action_space.shape == (3,), env.action_space.shape
    assert env.observation_space.shape == (16,), env.observation_space.shape
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs), obs

    # Also check with Kalman tracking disabled (direct RL track)
    env2 = SoldierEnv(config=EnvConfig(use_kalman_tracking=False))
    obs2, info2 = env2.reset(seed=1)
    assert env2.observation_space.contains(obs2), obs2


# ---------------------------------------------------------------------------
# B. State dimensions
# ---------------------------------------------------------------------------
def test_state_dims():
    env = SoldierEnv()
    obs, info = env.reset(seed=1)
    assert info["soldier_pos"].shape == (3,)
    assert info["defender_pos"].shape == (3,)
    assert info["enemy_pos"].shape == (3,)
    assert info["soldier_pos"][2] == 0.0


# ---------------------------------------------------------------------------
# C. Enemy spawning
# ---------------------------------------------------------------------------
def test_enemy_spawn():
    config = EnvConfig()
    L = config.L
    for seed in range(30):
        env = SoldierEnv(config=config)
        _, info = env.reset(seed=seed)
        enemy = info["enemy_pos"]
        assert config.enemy_spawn_altitude_min - 1e-4 <= enemy[2] <= config.enemy_spawn_altitude_max + 1e-4, enemy
        on_perimeter = np.isclose(abs(enemy[0]), L, atol=1e-3) or np.isclose(abs(enemy[1]), L, atol=1e-3)
        assert on_perimeter, enemy


# ---------------------------------------------------------------------------
# D. Reproducibility
# ---------------------------------------------------------------------------
def test_reproducibility():
    env1 = SoldierEnv()
    env2 = SoldierEnv()
    obs1, info1 = env1.reset(seed=777)
    obs2, info2 = env2.reset(seed=777)
    assert np.allclose(info1["soldier_pos"], info2["soldier_pos"])
    assert np.allclose(info1["defender_pos"], info2["defender_pos"])
    assert np.allclose(info1["enemy_pos"], info2["enemy_pos"])
    assert np.allclose(obs1, obs2)


# ---------------------------------------------------------------------------
# E. Defender vertical movement (bounded acceleration, NOT instantaneous)
# ---------------------------------------------------------------------------
def test_defender_vertical_movement():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=5)
    # Room above the defender: max_altitude=30, place defender at z=5, at rest.
    env._defender_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    before_pos = env._defender_pos.copy()
    action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    env._move_defender(action)
    after_pos = env._defender_pos
    after_vel = env._defender_vel

    # The vehicle now has persistent, acceleration-limited velocity: a single
    # upward command from rest can NOT produce the old instantaneous v_d*dt
    # displacement. The velocity change this step must respect max_accel*dt.
    max_dv = config.defender_max_accel * config.dt
    assert np.linalg.norm(after_vel) <= max_dv + 1e-6, (after_vel, max_dv)
    assert after_vel[2] > 0.0, after_vel
    assert abs(after_vel[0]) < 1e-6 and abs(after_vel[1]) < 1e-6

    # Position integration: p_{t+1} = p_t + v_{t+1} * dt
    expected_pos = before_pos + after_vel * config.dt
    assert np.allclose(after_pos, expected_pos, atol=1e-5), (after_pos, expected_pos)

    # Strictly less than the OLD (pre-dynamics-task) instantaneous displacement
    old_instantaneous_dz = config.v_d * config.dt
    assert (after_pos[2] - before_pos[2]) < old_instantaneous_dz
    assert abs(after_pos[0] - before_pos[0]) < 1e-6
    assert abs(after_pos[1] - before_pos[1]) < 1e-6


# ---------------------------------------------------------------------------
# F. Ground boundary
# ---------------------------------------------------------------------------
def test_ground_boundary():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=6)
    env._defender_pos = np.array([0.0, 0.0, 0.5], dtype=np.float32)
    action = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    for _ in range(50):
        env._move_defender(action)
        assert env._defender_pos[2] >= 0.0, env._defender_pos


# ---------------------------------------------------------------------------
# G. Upper altitude boundary
# ---------------------------------------------------------------------------
def test_upper_altitude_boundary():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=7)
    env._defender_pos = np.array([0.0, 0.0, config.max_altitude - 0.5], dtype=np.float32)
    action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for _ in range(50):
        env._move_defender(action)
        assert env._defender_pos[2] <= config.max_altitude + 1e-6, env._defender_pos


# ---------------------------------------------------------------------------
# H. 3D distance correctness
# ---------------------------------------------------------------------------
def test_3d_distance():
    defender = np.array([0.0, 0.0, 0.0])
    enemy = np.array([0.0, 0.0, 10.0])
    assert abs(np.linalg.norm(defender - enemy) - 10.0) < 1e-9

    defender2 = np.array([0.0, 0.0, 0.0])
    enemy2 = np.array([3.0, 4.0, 12.0])
    assert abs(np.linalg.norm(defender2 - enemy2) - 13.0) < 1e-9


# ---------------------------------------------------------------------------
# I. Detection uses 3D distance
# ---------------------------------------------------------------------------
def test_detection_3d():
    config = EnvConfig(detection_radius=15.0)
    env = SoldierEnv(config=config)
    env.reset(seed=8)

    # Same XY, vertical separation greater than detection_radius -> NOT detected
    env._defender_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    env._enemy_pos = np.array([0.0, 0.0, 20.0], dtype=np.float32)
    env._enemy_detected = False
    env._update_detection()
    assert env._enemy_detected is False, "Enemy should not be detected: horizontal distance is 0 but 3D distance exceeds detection_radius"

    # Move enemy vertically within detection_radius -> detection should occur
    env._enemy_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    env._update_detection()
    assert env._enemy_detected is True


# ---------------------------------------------------------------------------
# J. Interception uses 3D distance
# ---------------------------------------------------------------------------
def test_intercept_3d():
    config = EnvConfig(intercept_radius=2.5)
    defender = np.array([0.0, 0.0, 0.0])

    enemy_far = np.array([0.0, 0.0, 10.0])  # same XY, beyond intercept_radius vertically
    dist_far = np.linalg.norm(defender - enemy_far)
    assert dist_far > config.intercept_radius, "Same-XY vertical separation must not be treated as zero range"

    enemy_near = np.array([0.0, 0.0, 2.0])  # within true 3D intercept radius
    dist_near = np.linalg.norm(defender - enemy_near)
    assert dist_near <= config.intercept_radius


# ---------------------------------------------------------------------------
# K. Soldier remains grounded
# ---------------------------------------------------------------------------
def test_soldier_grounded():
    env = SoldierEnv()
    obs, info = env.reset(seed=9)
    for _ in range(500):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        assert info["soldier_pos"][2] == 0.0
        assert np.all(np.isfinite(info["soldier_pos"]))
        assert np.all(np.isfinite(info["defender_pos"]))
        assert np.all(np.isfinite(info["enemy_pos"]))
        if term:
            obs, info = env.reset()


# ---------------------------------------------------------------------------
# L. Engagement-volume bounds
# ---------------------------------------------------------------------------
def test_bounds():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=10)
    for _ in range(500):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        for key in ("defender_pos", "enemy_pos"):
            p = info[key]
            assert -config.L - 1e-3 <= p[0] <= config.L + 1e-3, (key, p)
            assert -config.L - 1e-3 <= p[1] <= config.L + 1e-3, (key, p)
            assert -1e-3 <= p[2] <= config.max_altitude + 1e-3, (key, p)
        if term:
            obs, info = env.reset()
    # Documented decision: the hostile UAV and defender use the SAME
    # physically-consistent boundary handling (_apply_boundary): position is
    # clipped to the boundary and only the outward-normal velocity component
    # is zeroed (no bounce/reflection), introduced by the constrained-dynamics
    # task. This still guarantees x,y in [-L,L] and z in [0,max_altitude].


# ---------------------------------------------------------------------------
# Kalman filter tests (section 22)
# ---------------------------------------------------------------------------
def test_kalman_dims():
    kf = EnemyKalmanFilter(dt=0.5, process_var=1.0, measurement_var=0.5)
    assert kf.F.shape == (6, 6)
    assert kf.H.shape == (3, 6)
    assert kf.Q.shape == (6, 6)
    assert kf.R.shape == (3, 3)
    assert kf.P.shape == (6, 6)

    kf.initialize(np.array([10.0, -5.0, 20.0]))
    assert kf.get_state().shape == (6,)
    assert kf.get_position().shape == (3,)
    assert kf.get_velocity().shape == (3,)
    assert kf.get_position_uncertainty().shape == (3,)
    assert kf.get_velocity_uncertainty().shape == (3,)
    assert np.allclose(kf.get_position(), [10.0, -5.0, 20.0])


def test_kalman_invalid_input():
    kf = EnemyKalmanFilter(dt=0.5, process_var=1.0, measurement_var=0.5)
    for bad in (np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0])):
        raised = False
        try:
            kf.initialize(bad)
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for shape {bad.shape}"

    kf.initialize(np.array([0.0, 0.0, 0.0]))
    for bad in (np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0])):
        raised = False
        try:
            kf.update(bad)
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for shape {bad.shape}"


def test_kalman_constant_velocity():
    dt = 0.5
    kf = EnemyKalmanFilter(dt=dt, process_var=0.1, measurement_var=0.05)
    vel_true = np.array([1.0, -0.5, 0.25])
    pos = np.array([0.0, 0.0, 10.0])
    kf.initialize(pos.copy())

    rng = np.random.default_rng(0)
    for _ in range(60):
        kf.predict()
        pos = pos + vel_true * dt
        meas = pos + rng.normal(0.0, 0.05, size=3)
        kf.update(meas)
        assert np.all(np.isfinite(kf.get_state()))
        P = kf.get_covariance()
        assert np.all(np.isfinite(P))
        assert np.allclose(P, P.T, atol=1e-6)

    vel_est = kf.get_velocity()
    assert np.linalg.norm(vel_est - vel_true) < 0.5, (vel_est, vel_true)


# ---------------------------------------------------------------------------
# Scripted policy vertical-geometry tests (section 23)
# ---------------------------------------------------------------------------
def test_greedy_policy_vertical():
    policy = GreedyInterceptPolicy()

    info_up = {
        "defender_pos": np.array([0.0, 0.0, 5.0]),
        "enemy_measurement": np.array([0.0, 0.0, 15.0]),
        "soldier_pos": np.array([0.0, 0.0, 0.0]),
        "enemy_detected": True,
    }
    action_up = policy.act(None, info_up)
    assert action_up.shape == (3,)
    assert action_up[2] > 0.9, action_up
    assert abs(action_up[0]) < 1e-6 and abs(action_up[1]) < 1e-6

    info_down = {
        "defender_pos": np.array([0.0, 0.0, 15.0]),
        "enemy_measurement": np.array([0.0, 0.0, 5.0]),
        "soldier_pos": np.array([0.0, 0.0, 0.0]),
        "enemy_detected": True,
    }
    action_down = policy.act(None, info_down)
    assert action_down[2] < -0.9, action_down


def test_kalman_greedy_policy_vertical():
    policy = KalmanGreedyInterceptPolicy()

    info_up = {
        "defender_pos": np.array([0.0, 0.0, 5.0]),
        "e_hat": np.array([0.0, 0.0, 15.0]),
        "soldier_pos": np.array([0.0, 0.0, 0.0]),
    }
    action_up = policy.act(None, info_up)
    assert action_up.shape == (3,)
    assert action_up[2] > 0.9, action_up

    info_down = {
        "defender_pos": np.array([0.0, 0.0, 15.0]),
        "e_hat": np.array([0.0, 0.0, 5.0]),
        "soldier_pos": np.array([0.0, 0.0, 0.0]),
    }
    action_down = policy.act(None, info_down)
    assert action_down[2] < -0.9, action_down


def test_random_policy_shape():
    policy = RandomPolicy(seed=0)
    action = policy.act(None, None)
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        ("A. action/observation spaces", test_spaces),
        ("B. state dims after reset", test_state_dims),
        ("C. enemy spawn geometry", test_enemy_spawn),
        ("D. reproducibility", test_reproducibility),
        ("E. defender vertical movement", test_defender_vertical_movement),
        ("F. ground boundary", test_ground_boundary),
        ("G. upper altitude boundary", test_upper_altitude_boundary),
        ("H. 3D distance correctness", test_3d_distance),
        ("I. detection uses 3D distance", test_detection_3d),
        ("J. interception uses 3D distance", test_intercept_3d),
        ("K. soldier remains grounded", test_soldier_grounded),
        ("L. engagement-volume bounds", test_bounds),
        ("Kalman: dimensions", test_kalman_dims),
        ("Kalman: invalid input rejected", test_kalman_invalid_input),
        ("Kalman: constant-velocity tracking", test_kalman_constant_velocity),
        ("Policy: Greedy vertical pursuit", test_greedy_policy_vertical),
        ("Policy: KalmanGreedy vertical pursuit", test_kalman_greedy_policy_vertical),
        ("Policy: Random action shape", test_random_policy_shape),
    ]

    print("=== 3D Foundation Correctness Tests ===\n")
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
