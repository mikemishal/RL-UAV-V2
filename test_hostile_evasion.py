"""Hostile-UAV reactive evasion correctness tests.

Script-style test consistent with the project's existing test files
(test_3d_dynamics.py, test_3d_foundation.py, etc). No external test
framework is used; each test function performs assertions and the runner
reports PASS/FAIL per test with a final summary.

Covers the proximity-triggered reactive evasion behavior layered onto the
hostile UAV's existing pursuit + weave guidance, introduced by
feature/hostile-evasion. Evasion must feed through the existing constrained
3D point-mass dynamics (acceleration/turn-rate/speed/climb/descent limits)
without bypassing them.

Run directly:
    python test_hostile_evasion.py
"""

from __future__ import annotations

import numpy as np

from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig


# ---------------------------------------------------------------------------
# A. Configuration validation
# ---------------------------------------------------------------------------
def test_config_validation():
    # Valid values must not raise
    EnvConfig(enemy_evasion_radius=20.0, enemy_evasion_gain=0.75)
    EnvConfig(enemy_evasion_gain=0.0)
    EnvConfig(enemy_evasion_gain=1.0)
    EnvConfig(enemy_evasion_enabled=False)

    for bad_kwargs in (
        {"enemy_evasion_radius": 0.0},
        {"enemy_evasion_radius": -5.0},
        {"enemy_evasion_gain": -0.1},
        {"enemy_evasion_gain": 1.1},
    ):
        raised = False
        try:
            EnvConfig(**bad_kwargs)
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for {bad_kwargs}"


# ---------------------------------------------------------------------------
# B. Outside-radius activation
# ---------------------------------------------------------------------------
def test_outside_radius_no_activation():
    config = EnvConfig(enemy_evasion_radius=20.0)
    env = SoldierEnv(config=config)
    env.reset(seed=1)

    enemy = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    defender = np.array([30.0, 0.0, 10.0], dtype=np.float32)
    result = env._compute_enemy_evasion(defender, enemy)

    assert abs(result["distance"] - 30.0) < 1e-5, result["distance"]
    assert result["activation"] == 0.0, result["activation"]
    assert result["effective_weight"] == 0.0, result["effective_weight"]
    assert np.allclose(result["evasion_component"], [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# C. Radius-boundary behavior
# ---------------------------------------------------------------------------
def test_radius_boundary():
    config = EnvConfig(enemy_evasion_radius=20.0)
    env = SoldierEnv(config=config)
    env.reset(seed=1)

    defender = np.array([0.0, 0.0, 10.0], dtype=np.float32)

    enemy_at_boundary = np.array([20.0, 0.0, 10.0], dtype=np.float32)
    result_at = env._compute_enemy_evasion(defender, enemy_at_boundary)
    assert abs(result_at["activation"]) < 1e-6, result_at["activation"]

    enemy_just_inside = np.array([19.9, 0.0, 10.0], dtype=np.float32)
    result_in = env._compute_enemy_evasion(defender, enemy_just_inside)
    assert 0.0 < result_in["activation"] < 0.1, result_in["activation"]  # small, no binary jump


# ---------------------------------------------------------------------------
# D. Half-radius activation
# ---------------------------------------------------------------------------
def test_half_radius_activation():
    config = EnvConfig(enemy_evasion_radius=20.0, enemy_evasion_gain=0.75)
    env = SoldierEnv(config=config)
    env.reset(seed=1)

    defender = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    enemy = np.array([10.0, 0.0, 10.0], dtype=np.float32)  # distance = R_e / 2
    result = env._compute_enemy_evasion(defender, enemy)

    assert abs(result["activation"] - 0.5) < 1e-6, result["activation"]
    assert abs(result["effective_weight"] - 0.5 * 0.75) < 1e-6, result["effective_weight"]


# ---------------------------------------------------------------------------
# E. Close-range activation
# ---------------------------------------------------------------------------
def test_close_range_activation():
    config = EnvConfig(enemy_evasion_radius=20.0)
    env = SoldierEnv(config=config)
    env.reset(seed=1)

    defender = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    enemy = np.array([0.001, 0.0, 10.0], dtype=np.float32)
    result = env._compute_enemy_evasion(defender, enemy)

    assert result["activation"] > 0.99, result["activation"]
    assert result["activation"] <= 1.0, result["activation"]


# ---------------------------------------------------------------------------
# F. Correct away direction (proves evasion is truly 3D)
# ---------------------------------------------------------------------------
def test_away_direction():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=1)

    defender = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    enemy = np.array([10.0, 0.0, 10.0], dtype=np.float32)
    result = env._compute_enemy_evasion(defender, enemy)
    assert np.allclose(result["away_direction"], [1.0, 0.0, 0.0], atol=1e-4), result["away_direction"]

    defender2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    enemy2 = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    result2 = env._compute_enemy_evasion(defender2, enemy2)
    assert np.allclose(result2["away_direction"], [0.0, 0.0, 1.0], atol=1e-4), result2["away_direction"]


# ---------------------------------------------------------------------------
# G. Evasion disabled == zero-gain regression
# ---------------------------------------------------------------------------
def test_evasion_disabled_matches_zero_gain():
    seed = 123
    config_off = EnvConfig(enemy_evasion_enabled=False)
    config_gain0 = EnvConfig(enemy_evasion_enabled=True, enemy_evasion_gain=0.0)
    env_off = SoldierEnv(config=config_off)
    env_gain0 = SoldierEnv(config=config_gain0)

    obs_off, info_off = env_off.reset(seed=seed)
    obs_g0, info_g0 = env_gain0.reset(seed=seed)
    assert np.allclose(info_off["enemy_pos"], info_g0["enemy_pos"])
    assert np.allclose(obs_off, obs_g0)

    rng = np.random.default_rng(999)
    for _ in range(300):
        action = rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
        obs_off, r_off, term_off, trunc_off, info_off = env_off.step(action)
        obs_g0, r_g0, term_g0, trunc_g0, info_g0 = env_gain0.step(action)

        assert np.allclose(info_off["enemy_pos"], info_g0["enemy_pos"], atol=1e-4), (
            info_off["enemy_pos"], info_g0["enemy_pos"]
        )
        assert np.allclose(info_off["enemy_vel"], info_g0["enemy_vel"], atol=1e-4)
        assert np.allclose(obs_off, obs_g0, atol=1e-4)

        if term_off or term_g0:
            break


# ---------------------------------------------------------------------------
# H. No-evasion behavior outside radius
# ---------------------------------------------------------------------------
def test_no_evasion_outside_radius_matches():
    seed = 55
    radius = 5.0
    config_off = EnvConfig(enemy_evasion_enabled=False, enemy_evasion_radius=radius)
    config_on = EnvConfig(enemy_evasion_enabled=True, enemy_evasion_radius=radius, enemy_evasion_gain=0.75)
    env_off = SoldierEnv(config=config_off)
    env_on = SoldierEnv(config=config_on)

    obs_off, info_off = env_off.reset(seed=seed)
    obs_on, info_on = env_on.reset(seed=seed)

    zero_action = np.zeros(3, dtype=np.float32)  # keeps defender at rest near the soldier
    iterations = 0
    for _ in range(300):
        if info_off["defender_enemy_dist"] <= radius:
            break  # stop before the premise (outside radius) is violated
        obs_off, r_off, term_off, trunc_off, info_off = env_off.step(zero_action)
        obs_on, r_on, term_on, trunc_on, info_on = env_on.step(zero_action)
        iterations += 1

        assert np.allclose(info_off["enemy_pos"], info_on["enemy_pos"], atol=1e-4)
        assert np.allclose(info_off["enemy_vel"], info_on["enemy_vel"], atol=1e-4)

        if term_off or term_on:
            break

    assert iterations > 5, "test did not exercise enough outside-radius steps"


# ---------------------------------------------------------------------------
# I. Evasion changes guidance inside radius
# ---------------------------------------------------------------------------
def test_evasion_changes_guidance_inside_radius():
    config_on = EnvConfig(enemy_evasion_enabled=True, enemy_evasion_gain=0.75, enemy_evasion_radius=20.0)
    config_off = EnvConfig(enemy_evasion_enabled=False, enemy_evasion_radius=20.0)
    env_on = SoldierEnv(config=config_on)
    env_off = SoldierEnv(config=config_off)
    env_on.reset(seed=1)
    env_off.reset(seed=1)

    defender_pos = np.array([5.0, 0.0, 10.0], dtype=np.float32)
    enemy_pos = np.array([0.0, 0.0, 10.0], dtype=np.float32)  # distance = 5 < 20

    ev_on = env_on._compute_enemy_evasion(defender_pos, enemy_pos)
    ev_off = env_off._compute_enemy_evasion(defender_pos, enemy_pos)

    assert ev_on["activation"] > 0.0, ev_on["activation"]
    assert ev_off["activation"] == 0.0, ev_off["activation"]
    assert not np.allclose(ev_on["evasion_component"], ev_off["evasion_component"])

    away = enemy_pos - defender_pos
    away_dir = away / np.linalg.norm(away)
    diff_component = ev_on["evasion_component"] - ev_off["evasion_component"]
    assert np.dot(diff_component, away_dir) > 0.0, diff_component


# ---------------------------------------------------------------------------
# J. Primary pursuit remains present
# ---------------------------------------------------------------------------
def test_pursuit_component_present():
    config = EnvConfig(enemy_evasion_enabled=True, enemy_evasion_gain=0.75, enemy_evasion_radius=20.0)
    env = SoldierEnv(config=config)
    env.reset(seed=1)

    env._soldier_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    env._enemy_pos = np.array([10.0, 0.0, 10.0], dtype=np.float32)
    env._defender_pos = np.array([8.0, 0.0, 10.0], dtype=np.float32)  # distance = 2, well inside radius

    expected_pursuit = env._soldier_pos - env._enemy_pos
    expected_pursuit = expected_pursuit / np.linalg.norm(expected_pursuit)

    env._move_enemy()

    # The pursuit direction u_p is computed and stored unconditionally,
    # proving it remains present in the guidance construction alongside
    # (not replaced by) an active evasion component.
    assert np.allclose(env._enemy_pursuit_direction, expected_pursuit, atol=1e-4), env._enemy_pursuit_direction
    assert env._enemy_evasion_info["activation"] > 0.0, env._enemy_evasion_info
    assert env._enemy_evasion_info["effective_weight"] > 0.0, env._enemy_evasion_info


# ---------------------------------------------------------------------------
# K. Evasion respects acceleration constraint
# ---------------------------------------------------------------------------
def test_evasion_respects_accel():
    config = EnvConfig(enemy_evasion_radius=30.0, enemy_evasion_gain=1.0)
    env = SoldierEnv(config=config)
    env.reset(seed=2)

    for _ in range(200):
        env._defender_pos = (env._enemy_pos + np.array([2.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
        env._move_enemy()
        assert env._enemy_dynamics_info["accel_used"] <= config.enemy_max_accel + 1e-4, (
            env._enemy_dynamics_info["accel_used"]
        )


# ---------------------------------------------------------------------------
# L. Evasion respects turn-rate constraint
# ---------------------------------------------------------------------------
def test_evasion_respects_turn_rate():
    config = EnvConfig(enemy_evasion_radius=30.0, enemy_evasion_gain=1.0)
    env = SoldierEnv(config=config)
    env.reset(seed=3)

    for i in range(200):
        # Alternate the defender's relative offset to force substantial
        # evasive heading adjustments step to step.
        offset = np.array([2.0, 0.0, 0.0]) if i % 2 == 0 else np.array([0.0, 2.0, 0.0])
        env._defender_pos = (env._enemy_pos + offset).astype(np.float32)
        env._move_enemy()
        assert env._enemy_dynamics_info["turn_rate_used_deg"] <= config.enemy_max_turn_rate_deg + 1e-2, (
            env._enemy_dynamics_info["turn_rate_used_deg"]
        )


# ---------------------------------------------------------------------------
# M. Evasion respects speed constraint
# ---------------------------------------------------------------------------
def test_evasion_respects_speed():
    config = EnvConfig(enemy_evasion_radius=30.0, enemy_evasion_gain=1.0)
    env = SoldierEnv(config=config)
    env.reset(seed=4)

    for _ in range(200):
        env._defender_pos = (env._enemy_pos + np.array([1.5, 1.5, 0.0], dtype=np.float32)).astype(np.float32)
        env._move_enemy()
        assert np.linalg.norm(env._enemy_vel) <= config.v_e + 1e-4, env._enemy_vel


# ---------------------------------------------------------------------------
# N. Evasion respects climb/descent constraints
# ---------------------------------------------------------------------------
def test_evasion_respects_climb_descent():
    config = EnvConfig(enemy_evasion_radius=30.0, enemy_evasion_gain=1.0)
    env = SoldierEnv(config=config)
    env.reset(seed=5)

    for _ in range(200):
        # Defender directly above/below induces a vertical evasive component.
        env._defender_pos = (env._enemy_pos + np.array([0.0, 0.0, 2.0], dtype=np.float32)).astype(np.float32)
        env._move_enemy()
        vz = env._enemy_vel[2]
        assert vz <= config.enemy_max_climb_rate + 1e-4, vz
        assert vz >= -config.enemy_max_descent_rate - 1e-4, vz


# ---------------------------------------------------------------------------
# O. Evasion diagnostics validity
# ---------------------------------------------------------------------------
def test_evasion_diagnostics_valid():
    config = EnvConfig(enemy_evasion_radius=20.0, enemy_evasion_gain=0.75)
    env = SoldierEnv(config=config)
    env.reset(seed=6)

    for _ in range(50):
        env._defender_pos = (env._enemy_pos + np.array([3.0, 1.0, -1.0], dtype=np.float32)).astype(np.float32)
        env._move_enemy()
        info = env._enemy_evasion_info

        assert 0.0 <= info["activation"] <= 1.0, info["activation"]
        assert 0.0 <= info["effective_weight"] <= config.enemy_evasion_gain + 1e-9, info["effective_weight"]
        assert info["evasion_component"].shape == (3,)
        assert np.all(np.isfinite(info["evasion_component"]))
        assert np.isfinite(info["activation"])
        assert np.isfinite(info["effective_weight"])


# ---------------------------------------------------------------------------
# P. Reproducibility
# ---------------------------------------------------------------------------
def test_reproducibility():
    config = EnvConfig(enemy_evasion_radius=20.0, enemy_evasion_gain=0.75)
    env1 = SoldierEnv(config=config)
    env2 = SoldierEnv(config=config)

    obs1, info1 = env1.reset(seed=777)
    obs2, info2 = env2.reset(seed=777)
    assert np.allclose(obs1, obs2)

    rng = np.random.default_rng(42)
    for _ in range(200):
        action = rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
        obs1, r1, term1, trunc1, info1 = env1.step(action)
        obs2, r2, term2, trunc2, info2 = env2.step(action)

        assert np.allclose(info1["enemy_pos"], info2["enemy_pos"], atol=1e-6)
        assert np.allclose(info1["enemy_vel"], info2["enemy_vel"], atol=1e-6)
        assert abs(info1["enemy_evasion_activation"] - info2["enemy_evasion_activation"]) < 1e-6
        assert abs(info1["enemy_evasion_weight"] - info2["enemy_evasion_weight"]) < 1e-6

        if term1 or term2:
            break


# ---------------------------------------------------------------------------
# Q. Stress test (no NaN/Inf, all constraints + diagnostics valid)
# ---------------------------------------------------------------------------
def test_stress_no_nan():
    config = EnvConfig(enemy_evasion_radius=20.0, enemy_evasion_gain=0.75)
    env = SoldierEnv(config=config)
    env.reset(seed=99)

    for _ in range(1000):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)

        assert np.all(np.isfinite(obs)), obs
        assert np.isfinite(r), r
        assert env.observation_space.contains(obs), obs

        assert np.isfinite(info["enemy_evasion_activation"])
        assert np.isfinite(info["enemy_evasion_weight"])
        assert np.all(np.isfinite(info["enemy_evasion_vector"]))
        assert 0.0 <= info["enemy_evasion_activation"] <= 1.0, info["enemy_evasion_activation"]
        assert 0.0 <= info["enemy_evasion_weight"] <= config.enemy_evasion_gain + 1e-6, info["enemy_evasion_weight"]

        assert info["enemy_speed"] <= config.v_e + 1e-4, info["enemy_speed"]
        assert info["enemy_accel"] <= config.enemy_max_accel + 1e-4, info["enemy_accel"]
        assert info["enemy_turn_rate"] <= config.enemy_max_turn_rate_deg + 1e-2, info["enemy_turn_rate"]
        assert info["enemy_vel"][2] <= config.enemy_max_climb_rate + 1e-4, info["enemy_vel"]
        assert info["enemy_vel"][2] >= -config.enemy_max_descent_rate - 1e-4, info["enemy_vel"]

        if term:
            env.reset()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        ("A. configuration validation", test_config_validation),
        ("B. outside-radius activation", test_outside_radius_no_activation),
        ("C. radius-boundary behavior", test_radius_boundary),
        ("D. half-radius activation", test_half_radius_activation),
        ("E. close-range activation", test_close_range_activation),
        ("F. correct away direction (3D)", test_away_direction),
        ("G. evasion disabled == zero-gain regression", test_evasion_disabled_matches_zero_gain),
        ("H. no-evasion behavior outside radius", test_no_evasion_outside_radius_matches),
        ("I. evasion changes guidance inside radius", test_evasion_changes_guidance_inside_radius),
        ("J. primary pursuit remains present", test_pursuit_component_present),
        ("K. evasion respects acceleration constraint", test_evasion_respects_accel),
        ("L. evasion respects turn-rate constraint", test_evasion_respects_turn_rate),
        ("M. evasion respects speed constraint", test_evasion_respects_speed),
        ("N. evasion respects climb/descent constraints", test_evasion_respects_climb_descent),
        ("O. evasion diagnostics validity", test_evasion_diagnostics_valid),
        ("P. reproducibility", test_reproducibility),
        ("Q. stress test (no NaN/Inf)", test_stress_no_nan),
    ]

    print("=== Hostile-UAV Reactive Evasion Correctness Tests ===\n")
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
