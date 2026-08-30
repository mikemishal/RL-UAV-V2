"""
Regression tests proving the factored pure `advance_velocity` helper
(uav_defend/dynamics/constrained_point_mass.py) reproduces the exact
numerical behavior of the prior in-place SoldierEnv._advance_velocity
implementation, across a broad deterministic set of states.

Run directly:
    python test_constrained_point_mass.py
"""

from __future__ import annotations

import numpy as np

from uav_defend.dynamics.constrained_point_mass import advance_velocity

TOL = 1e-9


def _reference_advance_velocity(current_velocity, desired_velocity, max_speed, max_accel,
                                 max_turn_rate_rad, max_climb_rate, max_descent_rate, dt, eps):
    """Independent re-derivation of the ORIGINAL SoldierEnv._advance_velocity
    equations (prior to factoring), used ONLY as a regression oracle -- kept
    intentionally separate from the implementation under test so this test
    cannot pass merely by both sides sharing a bug."""
    current_velocity = np.asarray(current_velocity, dtype=np.float64)
    desired_velocity = np.asarray(desired_velocity, dtype=np.float64)

    desired_vz = desired_velocity[2]
    desired_vz_clipped = float(np.clip(desired_vz, -max_descent_rate, max_climb_rate))

    cur_h = current_velocity[0:2]
    des_h = desired_velocity[0:2]
    ch = float(np.linalg.norm(cur_h))
    dh = float(np.linalg.norm(des_h))

    if dh <= eps:
        target_heading = np.arctan2(cur_h[1], cur_h[0]) if ch > eps else 0.0
        target_h_mag = 0.0
    elif ch <= eps:
        target_heading = np.arctan2(des_h[1], des_h[0])
        target_h_mag = dh
    else:
        psi_current = np.arctan2(cur_h[1], cur_h[0])
        psi_desired = np.arctan2(des_h[1], des_h[0])
        delta_psi = psi_desired - psi_current
        delta_psi = (delta_psi + np.pi) % (2 * np.pi) - np.pi
        max_delta = max_turn_rate_rad * dt
        if abs(delta_psi) > max_delta:
            delta_psi = np.sign(delta_psi) * max_delta
        target_heading = psi_current + delta_psi
        target_h_mag = dh

    target_velocity = np.array([
        target_h_mag * np.cos(target_heading),
        target_h_mag * np.sin(target_heading),
        desired_vz_clipped,
    ])

    delta_v = target_velocity - current_velocity
    delta_v_norm = float(np.linalg.norm(delta_v))
    max_delta_v = max_accel * dt
    if delta_v_norm > max_delta_v and delta_v_norm > eps:
        delta_v = delta_v / delta_v_norm * max_delta_v
    next_velocity = current_velocity + delta_v

    speed = float(np.linalg.norm(next_velocity))
    if speed > max_speed and speed > eps:
        next_velocity = next_velocity / speed * max_speed
    next_velocity[2] = np.clip(next_velocity[2], -max_descent_rate, max_climb_rate)
    return next_velocity.astype(np.float32)


DEFAULT_PARAMS = dict(max_speed=18.0, max_accel=8.0, max_turn_rate_rad=np.radians(90.0),
                       max_climb_rate=6.0, max_descent_rate=6.0, dt=0.5, eps=1e-8)


def _check(current_velocity, desired_velocity, **overrides):
    params = {**DEFAULT_PARAMS, **overrides}
    expected = _reference_advance_velocity(current_velocity, desired_velocity, **params)
    actual, diagnostics = advance_velocity(current_velocity, desired_velocity, **params)
    assert np.allclose(actual, expected, atol=TOL), f"mismatch: {actual} vs {expected}"
    assert actual.dtype == np.float32
    assert np.all(np.isfinite(actual))
    return actual, diagnostics


def test_rest():
    _check([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_accelerating_from_rest():
    actual, diag = _check([0.0, 0.0, 0.0], [18.0, 0.0, 0.0])
    assert diag["accel_saturated"]
    assert np.linalg.norm(actual) <= 18.0 + TOL


def test_deceleration():
    _check([18.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_heading_change_small():
    _check([10.0, 0.0, 0.0], [0.0, 10.0, 0.0])


def test_turn_rate_saturation():
    actual, diag = _check([18.0, 0.0, 0.0], [-18.0, 0.0, 0.0])
    assert diag["turn_saturated"] or diag["accel_saturated"]


def test_acceleration_saturation():
    actual, diag = _check([0.0, 0.0, 0.0], [18.0, 0.0, 0.0], max_accel=1.0)
    assert diag["accel_saturated"]


def test_climb_saturation():
    actual, diag = _check([5.0, 0.0, 0.0], [5.0, 0.0, 100.0], max_climb_rate=6.0)
    assert diag["climb_saturated"]
    assert actual[2] <= 6.0 + TOL


def test_descent_saturation():
    actual, diag = _check([5.0, 0.0, 0.0], [5.0, 0.0, -100.0], max_descent_rate=6.0)
    assert diag["climb_saturated"]
    assert actual[2] >= -6.0 - TOL


def test_combined_horizontal_vertical_motion():
    _check([3.0, 4.0, 2.0], [-2.0, 6.0, -3.0])


def test_near_zero_velocity():
    _check([1e-10, -1e-10, 1e-10], [1e-9, 1e-9, 0.0])


def test_max_speed_clipping():
    actual, diag = _check([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], max_accel=1000.0, max_turn_rate_rad=np.radians(1000.0))
    assert np.linalg.norm(actual) <= DEFAULT_PARAMS["max_speed"] + TOL


def test_zero_desired_command_holds_heading_and_brakes():
    _check([10.0, 5.0, 1.0], [0.0, 0.0, 0.0])


def test_randomized_deterministic_samples():
    rng = np.random.default_rng(42)
    for _ in range(200):
        cur = rng.uniform(-20, 20, size=3)
        des = rng.uniform(-20, 20, size=3)
        max_speed = float(rng.uniform(5, 25))
        max_accel = float(rng.uniform(1, 15))
        max_turn = np.radians(float(rng.uniform(10, 180)))
        max_climb = float(rng.uniform(1, 10))
        max_descent = float(rng.uniform(1, 10))
        dt = float(rng.choice([0.1, 0.25, 0.5, 1.0]))
        _check(cur, des, max_speed=max_speed, max_accel=max_accel, max_turn_rate_rad=max_turn,
               max_climb_rate=max_climb, max_descent_rate=max_descent, dt=dt)


def test_does_not_mutate_input_arrays():
    cur = np.array([1.0, 2.0, 3.0])
    des = np.array([4.0, 5.0, 6.0])
    cur_copy, des_copy = cur.copy(), des.copy()
    advance_velocity(cur, des, **DEFAULT_PARAMS)
    assert np.array_equal(cur, cur_copy)
    assert np.array_equal(des, des_copy)


def test_diagnostics_keys_present():
    _, diag = advance_velocity([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], **DEFAULT_PARAMS)
    for key in ("accel_used", "turn_rate_used_deg", "accel_saturated", "turn_saturated", "climb_saturated"):
        assert key in diag


def test_environment_wrapper_matches_pure_helper():
    """SoldierEnv._advance_velocity must delegate to the pure helper and
    produce identical output for the same inputs/config."""
    from uav_defend.envs import SoldierEnv
    from uav_defend.config import EnvConfig

    config = EnvConfig()
    env = SoldierEnv(config=config)
    env.reset(seed=0)

    cur = np.array([3.0, -2.0, 1.0])
    des = np.array([-1.0, 4.0, -2.0])
    via_wrapper, diag_wrapper = env._advance_velocity(
        cur, des, max_speed=config.v_d, max_accel=config.defender_max_accel,
        max_turn_rate_rad=np.radians(config.defender_max_turn_rate_deg),
        max_climb_rate=config.defender_max_climb_rate, max_descent_rate=config.defender_max_descent_rate,
    )
    via_pure, diag_pure = advance_velocity(
        cur, des, max_speed=config.v_d, max_accel=config.defender_max_accel,
        max_turn_rate_rad=np.radians(config.defender_max_turn_rate_deg),
        max_climb_rate=config.defender_max_climb_rate, max_descent_rate=config.defender_max_descent_rate,
        dt=config.dt, eps=config.eps,
    )
    assert np.array_equal(via_wrapper, via_pure)
    assert diag_wrapper == diag_pure
    env.close()


if __name__ == "__main__":
    import sys
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
