"""Constrained 3D UAV dynamics correctness tests.

Script-style test consistent with the project's existing test files
(test_env.py, test_3d_foundation.py, etc). No external test framework is
used; each test function performs assertions and the runner reports
PASS/FAIL per test with a final summary.

Covers persistent velocity, bounded acceleration, bounded horizontal turn
rate, bounded climb/descent rate, maximum speed, position integration, and
physically-consistent boundary handling introduced by the constrained 3D
point-mass dynamics upgrade (feature/3d-constrained-dynamics).

Run directly:
    python test_3d_dynamics.py
"""

from __future__ import annotations

import numpy as np

from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig


# ---------------------------------------------------------------------------
# A. Observation/action dimensions
# ---------------------------------------------------------------------------
def test_dimensions():
    env = SoldierEnv()
    assert env.action_space.shape == (3,), env.action_space.shape
    assert env.observation_space.shape == (16,), env.observation_space.shape
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs), obs

    env2 = SoldierEnv(config=EnvConfig(use_kalman_tracking=False))
    obs2, info2 = env2.reset(seed=1)
    assert env2.observation_space.contains(obs2), obs2


# ---------------------------------------------------------------------------
# B. Initial velocities
# ---------------------------------------------------------------------------
def test_initial_velocities():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=1)

    assert info["defender_vel"].shape == (3,)
    assert info["enemy_vel"].shape == (3,)
    assert np.allclose(info["defender_vel"], [0.0, 0.0, 0.0], atol=1e-6), info["defender_vel"]

    enemy_speed = np.linalg.norm(info["enemy_vel"])
    assert enemy_speed <= config.v_e + 1e-4, enemy_speed

    to_soldier = info["soldier_pos"] - info["enemy_pos"]
    to_soldier_dir = to_soldier / np.linalg.norm(to_soldier)
    enemy_dir = info["enemy_vel"] / (enemy_speed + 1e-12)
    assert np.dot(to_soldier_dir, enemy_dir) > 0.9, (to_soldier_dir, enemy_dir)


# ---------------------------------------------------------------------------
# C. Defender cannot instantly reach maximum speed
# ---------------------------------------------------------------------------
def test_defender_no_instant_max_speed():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=2)

    v0 = env._defender_vel.copy()
    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    env._move_defender(action)
    v1 = env._defender_vel.copy()

    dv = np.linalg.norm(v1 - v0)
    assert dv <= config.defender_max_accel * config.dt + 1e-6, dv

    if config.defender_max_accel * config.dt < config.v_d:
        assert np.linalg.norm(v1) < config.v_d - 1e-6, v1


# ---------------------------------------------------------------------------
# D. Acceleration bound over many steps (defender + enemy)
# ---------------------------------------------------------------------------
def test_acceleration_bound_many_steps():
    """
    Verify the DYNAMICS-limited acceleration (the "accel_used" diagnostic
    computed inside _advance_velocity, before any boundary contact) never
    exceeds the configured max_accel. This is checked via the info dict's
    diagnostic fields, which are the correct measure of the acceleration
    constraint in isolation.

    Note: the RAW velocity delta (before vs. after step()) can legitimately
    exceed max_accel*dt when the vehicle contacts a boundary that step,
    because boundary handling (_apply_boundary, per item 15 of the dynamics
    spec) instantly zeroes the outward-normal velocity component -- a
    separate, intentionally discontinuous "wall collision" rule, NOT
    governed by the acceleration constraint. This is verified explicitly
    below rather than silently ignored.
    """
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=3)

    L = config.L
    H = config.max_altitude
    boundary_tol = 1e-3

    for _ in range(200):
        prev_dv = env._defender_vel.copy()
        prev_ev = env._enemy_vel.copy()
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)

        # Primary check: the dynamics-limited acceleration diagnostic must
        # always respect the configured bound.
        assert info["defender_accel"] <= config.defender_max_accel + 1e-4, info["defender_accel"]
        assert info["enemy_accel"] <= config.enemy_max_accel + 1e-4, info["enemy_accel"]

        # Secondary check: if the RAW velocity delta exceeds the dynamics
        # bound, it must be explained by a boundary contact this step.
        d_accel_raw = np.linalg.norm(env._defender_vel - prev_dv) / config.dt
        e_accel_raw = np.linalg.norm(env._enemy_vel - prev_ev) / config.dt

        if d_accel_raw > config.defender_max_accel + 1e-3:
            p = info["defender_pos"]
            at_boundary = (
                abs(abs(p[0]) - L) < boundary_tol
                or abs(abs(p[1]) - L) < boundary_tol
                or p[2] < boundary_tol
                or abs(p[2] - H) < boundary_tol
            )
            assert at_boundary, (
                "Defender raw acceleration exceeded max_accel without a "
                "boundary contact to explain it", d_accel_raw, p
            )
        if e_accel_raw > config.enemy_max_accel + 1e-3:
            p = info["enemy_pos"]
            at_boundary = (
                abs(abs(p[0]) - L) < boundary_tol
                or abs(abs(p[1]) - L) < boundary_tol
                or p[2] < boundary_tol
                or abs(p[2] - H) < boundary_tol
            )
            assert at_boundary, (
                "Enemy raw acceleration exceeded max_accel without a "
                "boundary contact to explain it", e_accel_raw, p
            )

        if term:
            env.reset()


# ---------------------------------------------------------------------------
# E. Maximum defender speed
# ---------------------------------------------------------------------------
def test_max_defender_speed():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=4)

    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    max_speed_seen = 0.0
    for _ in range(200):
        env._move_defender(action)
        speed = np.linalg.norm(env._defender_vel)
        max_speed_seen = max(max_speed_seen, speed)
        assert speed <= config.v_d + 1e-6, speed

    assert max_speed_seen > 0.5 * config.v_d, max_speed_seen


# ---------------------------------------------------------------------------
# F. Maximum hostile speed
# ---------------------------------------------------------------------------
def test_max_enemy_speed():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=5)

    for _ in range(300):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        assert info["enemy_speed"] <= config.v_e + 1e-4, info["enemy_speed"]
        if term:
            env.reset()


# ---------------------------------------------------------------------------
# G. Zero action means bounded braking
# ---------------------------------------------------------------------------
def test_zero_action_bounded_braking():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=6)

    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # A small number of ramp-up steps: enough to reach a meaningfully
    # nonzero speed without traveling far enough to contact the horizontal
    # boundary (which would zero the velocity via _apply_boundary and
    # confound this test with boundary behavior instead of braking).
    for _ in range(3):
        env._move_defender(action)
    v_before_brake = env._defender_vel.copy()
    speed_before = np.linalg.norm(v_before_brake)
    assert speed_before > 1.0, speed_before  # meaningfully moving

    zero_action = np.zeros(3, dtype=np.float32)
    env._move_defender(zero_action)
    speed_after_first = np.linalg.norm(env._defender_vel)

    # Must NOT stop instantly, and must be strictly decelerating
    assert speed_after_first > 1e-6, speed_after_first
    assert speed_after_first < speed_before, (speed_after_first, speed_before)

    decel = np.linalg.norm(env._defender_vel - v_before_brake) / config.dt
    assert decel <= config.defender_max_accel + 1e-4, decel

    prev_speed = speed_after_first
    for _ in range(50):
        env._move_defender(zero_action)
        speed = np.linalg.norm(env._defender_vel)
        assert speed <= prev_speed + 1e-6, (speed, prev_speed)
        prev_speed = speed
    assert prev_speed < 1e-3, prev_speed


# ---------------------------------------------------------------------------
# H. Turn-rate constraint (with shortest-path wrapping)
# ---------------------------------------------------------------------------
def test_turn_rate_constraint():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=7)
    max_delta = np.radians(config.defender_max_turn_rate_deg) * config.dt

    def one_turn(v0_heading_deg, action_heading_deg):
        env._defender_vel = np.array([
            10.0 * np.cos(np.radians(v0_heading_deg)),
            10.0 * np.sin(np.radians(v0_heading_deg)),
            0.0,
        ], dtype=np.float32)
        psi0 = np.arctan2(env._defender_vel[1], env._defender_vel[0])
        action = np.array([
            np.cos(np.radians(action_heading_deg)),
            np.sin(np.radians(action_heading_deg)),
            0.0,
        ], dtype=np.float32)
        env._move_defender(action)
        psi1 = np.arctan2(env._defender_vel[1], env._defender_vel[0])
        delta = (psi1 - psi0 + np.pi) % (2 * np.pi) - np.pi
        assert abs(delta) <= max_delta + 1e-6, (np.degrees(delta), np.degrees(max_delta))
        return delta

    # Large left / right turns: heading change must respect the turn-rate
    # cap. Note: at high speed, the SIMULTANEOUS acceleration constraint can
    # further reduce (never increase) the realized heading change below the
    # turn-rate cap, because rotating a velocity vector at near-constant
    # speed itself requires a lateral velocity change that competes for the
    # same acceleration budget. The turn-rate cap is therefore an upper
    # bound on realized heading change, not a guarantee of exact saturation.
    d_left = one_turn(0.0, 90.0)
    assert 0.0 < d_left <= max_delta + 1e-6, np.degrees(d_left)
    d_right = one_turn(0.0, -90.0)
    assert -max_delta - 1e-6 <= d_right < 0.0, np.degrees(d_right)

    # Wrapping near +-179 degrees: true angular distance is small (~2 deg),
    # NOT ~358 deg. The realized change may be further reduced by the
    # acceleration constraint (see above), but must remain small in
    # magnitude and correctly signed -- a wrapping bug would instead produce
    # a large (~358 deg, wrong-signed) heading change.
    d_wrap1 = one_turn(179.0, -179.0)
    assert 0.0 < d_wrap1 < 10.0, np.degrees(d_wrap1)
    d_wrap2 = one_turn(-179.0, 179.0)
    assert -10.0 < d_wrap2 < 0.0, np.degrees(d_wrap2)


# ---------------------------------------------------------------------------
# I. Turn rate when stationary
# ---------------------------------------------------------------------------
def test_turn_rate_when_stationary():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=8)
    env._defender_vel = np.zeros(3, dtype=np.float32)

    action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    env._move_defender(action)
    v1 = env._defender_vel

    assert np.all(np.isfinite(v1)), v1
    assert v1[1] > 0.0, v1
    assert abs(v1[0]) < 1e-6, v1
    assert np.linalg.norm(v1) <= config.defender_max_accel * config.dt + 1e-6, v1


# ---------------------------------------------------------------------------
# J. Climb-rate limit
# ---------------------------------------------------------------------------
def test_climb_rate_limit():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=9)
    action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for _ in range(50):
        env._move_defender(action)
        assert env._defender_vel[2] <= config.defender_max_climb_rate + 1e-6, env._defender_vel

    env2 = SoldierEnv(config=config)
    env2.reset(seed=10)
    for _ in range(200):
        obs, r, term, trunc, info = env2.step(env2.action_space.sample())
        assert info["enemy_vel"][2] <= config.enemy_max_climb_rate + 1e-6, info["enemy_vel"]
        if term:
            env2.reset()


# ---------------------------------------------------------------------------
# K. Descent-rate limit
# ---------------------------------------------------------------------------
def test_descent_rate_limit():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=11)
    action = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    for _ in range(50):
        env._move_defender(action)
        assert env._defender_vel[2] >= -config.defender_max_descent_rate - 1e-6, env._defender_vel

    env2 = SoldierEnv(config=config)
    env2.reset(seed=12)
    for _ in range(200):
        obs, r, term, trunc, info = env2.step(env2.action_space.sample())
        assert info["enemy_vel"][2] >= -config.enemy_max_descent_rate - 1e-6, info["enemy_vel"]
        if term:
            env2.reset()


# ---------------------------------------------------------------------------
# L. Total speed plus climb constraint simultaneously
# ---------------------------------------------------------------------------
def test_total_speed_and_climb_simultaneous():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=13)
    action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    for _ in range(200):
        env._move_defender(action)
        speed = np.linalg.norm(env._defender_vel)
        assert speed <= config.v_d + 1e-6, speed
        assert env._defender_vel[2] <= config.defender_max_climb_rate + 1e-6, env._defender_vel
        assert env._defender_vel[2] >= -config.defender_max_descent_rate - 1e-6, env._defender_vel


# ---------------------------------------------------------------------------
# M. Ground collision/boundary behavior
# ---------------------------------------------------------------------------
def test_ground_boundary_zeroes_outward_velocity():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=14)
    env._defender_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    env._defender_vel = np.array([2.0, 3.0, -5.0], dtype=np.float32)
    action = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    hit_ground = False
    for _ in range(20):
        env._move_defender(action)
        assert env._defender_pos[2] >= -1e-6, env._defender_pos
        if env._defender_pos[2] <= 1e-6:
            hit_ground = True
            assert env._defender_vel[2] >= -1e-6, env._defender_vel  # outward vz zeroed
    assert hit_ground


# ---------------------------------------------------------------------------
# N. Ceiling behavior
# ---------------------------------------------------------------------------
def test_ceiling_boundary_zeroes_outward_velocity():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=15)
    env._defender_pos = np.array([0.0, 0.0, config.max_altitude - 1.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    action = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    hit_ceiling = False
    for _ in range(20):
        env._move_defender(action)
        assert env._defender_pos[2] <= config.max_altitude + 1e-6, env._defender_pos
        if env._defender_pos[2] >= config.max_altitude - 1e-6:
            hit_ceiling = True
            assert env._defender_vel[2] <= 1e-6, env._defender_vel  # outward vz zeroed
    assert hit_ceiling


# ---------------------------------------------------------------------------
# O. Horizontal boundary behavior
# ---------------------------------------------------------------------------
def test_horizontal_boundary_positive():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=16)
    env._defender_pos = np.array([config.L - 1.0, 0.0, 5.0], dtype=np.float32)
    env._defender_vel = np.array([0.0, 3.0, 0.0], dtype=np.float32)  # tangential
    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    hit_boundary = False
    for _ in range(20):
        env._move_defender(action)
        assert env._defender_pos[0] <= config.L + 1e-6, env._defender_pos
        if env._defender_pos[0] >= config.L - 1e-6:
            hit_boundary = True
            assert env._defender_vel[0] <= 1e-6, env._defender_vel  # outward vx zeroed
    assert hit_boundary


def test_horizontal_boundary_negative():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=17)
    env._defender_pos = np.array([-config.L + 1.0, 0.0, 5.0], dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    action = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    hit_boundary = False
    for _ in range(20):
        env._move_defender(action)
        assert env._defender_pos[0] >= -config.L - 1e-6, env._defender_pos
        if env._defender_pos[0] <= -config.L + 1e-6:
            hit_boundary = True
            assert env._defender_vel[0] >= -1e-6, env._defender_vel  # outward vx zeroed
    assert hit_boundary


# ---------------------------------------------------------------------------
# P. Position integration
# ---------------------------------------------------------------------------
def test_position_integration():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=18)
    # State well inside the domain: no boundary clipping should occur
    env._defender_pos = np.array([0.0, 0.0, 15.0], dtype=np.float32)
    env._defender_vel = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    before_pos = env._defender_pos.copy()

    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    env._move_defender(action)
    after_pos = env._defender_pos
    after_vel = env._defender_vel

    expected_pos = before_pos + after_vel * config.dt
    assert np.allclose(after_pos, expected_pos, atol=1e-5), (after_pos, expected_pos)


# ---------------------------------------------------------------------------
# Q. Observation velocity normalization
# ---------------------------------------------------------------------------
def test_obs_velocity_normalization():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=19)
    env._defender_vel = np.array([config.v_d, 0.0, 0.0], dtype=np.float32)
    obs = env._get_obs()
    assert obs.shape == (16,), obs.shape
    defender_vel_obs = obs[6:9]
    assert np.allclose(defender_vel_obs, [1.0, 0.0, 0.0], atol=1e-4), defender_vel_obs


# ---------------------------------------------------------------------------
# R. No NaN/Inf stress test + measured constraint maxima
# ---------------------------------------------------------------------------
def test_stress_no_nan_and_measure_maxima():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=42)

    maxima = {
        "defender_speed": 0.0, "enemy_speed": 0.0,
        "defender_accel": 0.0, "enemy_accel": 0.0,
        "defender_turn_rate": 0.0, "enemy_turn_rate": 0.0,
        "defender_climb": 0.0, "defender_descent": 0.0,
        "enemy_climb": 0.0, "enemy_descent": 0.0,
    }

    for _ in range(1000):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)

        assert np.all(np.isfinite(obs)), obs
        assert np.isfinite(r), r
        assert env.observation_space.contains(obs), obs
        assert np.all(np.isfinite(info["defender_pos"]))
        assert np.all(np.isfinite(info["enemy_pos"]))
        assert np.all(np.isfinite(info["defender_vel"]))
        assert np.all(np.isfinite(info["enemy_vel"]))

        assert info["defender_speed"] <= config.v_d + 1e-4
        assert info["enemy_speed"] <= config.v_e + 1e-4
        assert info["defender_accel"] <= config.defender_max_accel + 1e-4
        assert info["enemy_accel"] <= config.enemy_max_accel + 1e-4
        assert info["defender_turn_rate"] <= config.defender_max_turn_rate_deg + 1e-2
        assert info["enemy_turn_rate"] <= config.enemy_max_turn_rate_deg + 1e-2
        assert info["defender_vel"][2] <= config.defender_max_climb_rate + 1e-4
        assert info["defender_vel"][2] >= -config.defender_max_descent_rate - 1e-4
        assert info["enemy_vel"][2] <= config.enemy_max_climb_rate + 1e-4
        assert info["enemy_vel"][2] >= -config.enemy_max_descent_rate - 1e-4

        maxima["defender_speed"] = max(maxima["defender_speed"], info["defender_speed"])
        maxima["enemy_speed"] = max(maxima["enemy_speed"], info["enemy_speed"])
        maxima["defender_accel"] = max(maxima["defender_accel"], info["defender_accel"])
        maxima["enemy_accel"] = max(maxima["enemy_accel"], info["enemy_accel"])
        maxima["defender_turn_rate"] = max(maxima["defender_turn_rate"], info["defender_turn_rate"])
        maxima["enemy_turn_rate"] = max(maxima["enemy_turn_rate"], info["enemy_turn_rate"])
        maxima["defender_climb"] = max(maxima["defender_climb"], info["defender_vel"][2])
        maxima["defender_descent"] = max(maxima["defender_descent"], -info["defender_vel"][2])
        maxima["enemy_climb"] = max(maxima["enemy_climb"], info["enemy_vel"][2])
        maxima["enemy_descent"] = max(maxima["enemy_descent"], -info["enemy_vel"][2])

        if term:
            env.reset()

    print("    --- measured maxima over 1000 steps (stress test) ---")
    print(f"    defender speed:   {maxima['defender_speed']:.4f}  (limit v_d={config.v_d})")
    print(f"    enemy speed:      {maxima['enemy_speed']:.4f}  (limit v_e={config.v_e})")
    print(f"    defender accel:   {maxima['defender_accel']:.4f}  (limit={config.defender_max_accel})")
    print(f"    enemy accel:      {maxima['enemy_accel']:.4f}  (limit={config.enemy_max_accel})")
    print(f"    defender turn:    {maxima['defender_turn_rate']:.4f} deg/s (limit={config.defender_max_turn_rate_deg})")
    print(f"    enemy turn:       {maxima['enemy_turn_rate']:.4f} deg/s (limit={config.enemy_max_turn_rate_deg})")
    print(f"    defender climb:   {maxima['defender_climb']:.4f}  (limit={config.defender_max_climb_rate})")
    print(f"    defender descent: {maxima['defender_descent']:.4f}  (limit={config.defender_max_descent_rate})")
    print(f"    enemy climb:      {maxima['enemy_climb']:.4f}  (limit={config.enemy_max_climb_rate})")
    print(f"    enemy descent:    {maxima['enemy_descent']:.4f}  (limit={config.enemy_max_descent_rate})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        ("A. observation/action dimensions", test_dimensions),
        ("B. initial velocities", test_initial_velocities),
        ("C. defender cannot instantly reach max speed", test_defender_no_instant_max_speed),
        ("D. acceleration bound over many steps", test_acceleration_bound_many_steps),
        ("E. maximum defender speed", test_max_defender_speed),
        ("F. maximum hostile speed", test_max_enemy_speed),
        ("G. zero action means bounded braking", test_zero_action_bounded_braking),
        ("H. turn-rate constraint (shortest-path wrapping)", test_turn_rate_constraint),
        ("I. turn rate when stationary", test_turn_rate_when_stationary),
        ("J. climb-rate limit", test_climb_rate_limit),
        ("K. descent-rate limit", test_descent_rate_limit),
        ("L. total speed + climb simultaneously", test_total_speed_and_climb_simultaneous),
        ("M. ground boundary zeroes outward velocity", test_ground_boundary_zeroes_outward_velocity),
        ("N. ceiling boundary zeroes outward velocity", test_ceiling_boundary_zeroes_outward_velocity),
        ("O1. horizontal boundary (+x)", test_horizontal_boundary_positive),
        ("O2. horizontal boundary (-x)", test_horizontal_boundary_negative),
        ("P. position integration", test_position_integration),
        ("Q. observation velocity normalization", test_obs_velocity_normalization),
        ("R. stress test (no NaN/Inf) + measured maxima", test_stress_no_nan_and_measure_maxima),
    ]

    print("=== Constrained 3D Dynamics Correctness Tests ===\n")
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
