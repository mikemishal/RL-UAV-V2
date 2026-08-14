"""Proportional-navigation (PN) guidance correctness tests.

Script-style test consistent with the project's existing test files
(test_hostile_evasion.py, test_3d_dynamics.py, etc). No external test
framework is used; each test function performs assertions and the runner
reports PASS/FAIL per test with a final summary.

Covers the 3-D vector PN guidance law (ProportionalNavigationPolicy),
including the ground-truth-independence guarantee, both sensing modes
(measurement / kalman), fallback behavior, and integration with the
environment's existing constrained 3-D point-mass dynamics and reactive
hostile evasion.

Run directly:
    python test_proportional_navigation.py
"""

from __future__ import annotations

import numpy as np

from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.baseline.proportional_navigation_policy import (
    ProportionalNavigationPolicy,
)


# ---------------------------------------------------------------------------
# Reference (independent) PN math, used to cross-check the policy's
# internal computation against the exact formulas from the specification.
# ---------------------------------------------------------------------------
def _expected_pn(defender_pos, defender_vel, target_pos, target_vel, N):
    r = np.asarray(target_pos, dtype=np.float64) - np.asarray(defender_pos, dtype=np.float64)
    R = float(np.linalg.norm(r))
    r_hat = r / R
    v_rel = np.asarray(target_vel, dtype=np.float64) - np.asarray(defender_vel, dtype=np.float64)
    Vc = -float(np.dot(r, v_rel)) / R
    omega_los = np.cross(r, v_rel) / (R * R)
    a_pn = N * Vc * np.cross(omega_los, r_hat)
    return {"R": R, "r_hat": r_hat, "Vc": Vc, "omega_los": omega_los, "a_pn": a_pn}


def _base_info(defender_pos, defender_vel, soldier_pos=None, enemy_detected=True):
    return {
        "defender_pos": np.asarray(defender_pos, dtype=np.float64),
        "defender_vel": np.asarray(defender_vel, dtype=np.float64),
        "soldier_pos": np.zeros(3) if soldier_pos is None else np.asarray(soldier_pos, dtype=np.float64),
        "enemy_detected": enemy_detected,
    }


# ---------------------------------------------------------------------------
# A. Policy construction
# ---------------------------------------------------------------------------
def test_construction():
    ProportionalNavigationPolicy(state_source="measurement")
    ProportionalNavigationPolicy(state_source="kalman")

    for bad in ("invalid", "", "Measurement", "PN"):
        raised = False
        try:
            ProportionalNavigationPolicy(state_source=bad)
        except ValueError:
            raised = True
        assert raised, bad

    for bad_nc in (0.0, -1.0, -3.0):
        raised = False
        try:
            ProportionalNavigationPolicy(navigation_constant=bad_nc)
        except ValueError:
            raised = True
        assert raised, bad_nc

    for bad_frac in (-0.1, 1.0, 1.5):
        raised = False
        try:
            ProportionalNavigationPolicy(min_pn_speed_fraction=bad_frac)
        except ValueError:
            raised = True
        assert raised, bad_frac

    for bad_dt in (0.0, -0.5):
        raised = False
        try:
            ProportionalNavigationPolicy(dt=bad_dt)
        except ValueError:
            raised = True
        assert raised, bad_dt

    # Valid boundary values must not raise
    ProportionalNavigationPolicy(min_pn_speed_fraction=0.0)
    ProportionalNavigationPolicy(navigation_constant=0.01)


# ---------------------------------------------------------------------------
# B. Action dimensions (through the real environment)
# ---------------------------------------------------------------------------
def test_action_dimensions():
    for state_source in ("measurement", "kalman"):
        policy = ProportionalNavigationPolicy(state_source=state_source)
        env_cfg = EnvConfig(use_kalman_tracking=(state_source == "kalman"))
        env = SoldierEnv(config=env_cfg)
        obs, info = env.reset(seed=1)
        policy.reset()
        for _ in range(80):
            action = policy.act(obs, info)
            assert action.shape == (3,), action.shape
            assert np.all(np.isfinite(action)), action
            assert np.all(action >= -1.0 - 1e-6) and np.all(action <= 1.0 + 1e-6), action
            obs, r, term, trunc, info = env.step(action)
            if term:
                obs, info = env.reset()
                policy.reset()


# ---------------------------------------------------------------------------
# C. Pre-detection escort
# ---------------------------------------------------------------------------
def test_pre_detection_escort():
    policy = ProportionalNavigationPolicy()
    defender_pos = np.array([0.0, 0.0, 0.0])
    soldier_pos = np.array([5.0, 0.0, 0.0])
    info = _base_info(defender_pos, np.zeros(3), soldier_pos=soldier_pos, enemy_detected=False)

    action = policy.act(None, info)
    assert np.allclose(action, [1.0, 0.0, 0.0], atol=1e-5), action
    assert policy.last_guidance_mode == "escort"

    info2 = dict(info)
    info2["soldier_pos"] = defender_pos.copy()
    action2 = policy.act(None, info2)
    assert np.allclose(action2, [0.0, 0.0, 0.0]), action2
    assert policy.last_guidance_mode == "escort"


# ---------------------------------------------------------------------------
# D. Measurement velocity finite difference
# ---------------------------------------------------------------------------
def test_measurement_finite_difference():
    policy = ProportionalNavigationPolicy(state_source="measurement", dt=0.5)
    defender_pos = np.zeros(3)
    z0 = np.array([10.0, 5.0, 20.0])
    z1 = np.array([11.0, 4.5, 20.25])

    info0 = _base_info(defender_pos, np.zeros(3))
    info0["enemy_measurement"] = z0
    policy.act(None, info0)
    assert policy.last_guidance_mode == "first_measurement_fallback"
    assert policy.last_target_velocity_estimate is None

    info1 = _base_info(defender_pos, np.zeros(3))
    info1["enemy_measurement"] = z1
    policy.act(None, info1)

    expected_vel = (z1 - z0) / 0.5
    assert np.allclose(policy.last_target_velocity_estimate, expected_vel, atol=1e-4), (
        policy.last_target_velocity_estimate, expected_vel
    )


# ---------------------------------------------------------------------------
# E. Reset removes measurement history
# ---------------------------------------------------------------------------
def test_reset_clears_history():
    policy = ProportionalNavigationPolicy(state_source="measurement", dt=0.5)
    defender_pos = np.zeros(3)
    z0 = np.array([10.0, 5.0, 20.0])

    info0 = _base_info(defender_pos, np.zeros(3))
    info0["enemy_measurement"] = z0
    policy.act(None, info0)

    policy.reset()

    z1 = np.array([11.0, 4.5, 20.25])
    info1 = _base_info(defender_pos, np.zeros(3))
    info1["enemy_measurement"] = z1
    policy.act(None, info1)

    assert policy.last_guidance_mode == "first_measurement_fallback", (
        "cross-episode velocity estimate leaked after reset()"
    )
    assert policy.last_target_velocity_estimate is None


# ---------------------------------------------------------------------------
# F. Kalman mode uses v_hat (not raw measurement)
# ---------------------------------------------------------------------------
def test_kalman_uses_v_hat():
    policy = ProportionalNavigationPolicy(state_source="kalman")
    defender_pos = np.zeros(3)
    defender_vel = np.array([5.0, 0.0, 0.0])
    e_hat = np.array([10.0, 0.0, 0.0])
    v_hat = np.array([1.0, 2.0, 3.0])

    info = _base_info(defender_pos, defender_vel)
    info["e_hat"] = e_hat
    info["v_hat"] = v_hat
    info["enemy_measurement"] = np.array([999.0, 999.0, 999.0])  # must be ignored

    action1 = policy.act(None, info)
    assert np.allclose(policy.last_target_velocity_estimate, v_hat, atol=1e-5)

    info2 = dict(info)
    info2["enemy_measurement"] = np.array([-999.0, -999.0, -999.0])
    action2 = policy.act(None, info2)
    assert np.allclose(action1, action2, atol=1e-6), (action1, action2)


# ---------------------------------------------------------------------------
# G. Ground-truth leakage test (mandatory)
# ---------------------------------------------------------------------------
def test_ground_truth_leakage():
    for state_source in ("measurement", "kalman"):
        policy = ProportionalNavigationPolicy(state_source=state_source)
        defender_pos = np.zeros(3)
        defender_vel = np.array([5.0, 0.0, 0.0])

        step1 = _base_info(defender_pos, defender_vel)
        step1["enemy_measurement"] = np.array([10.0, 0.0, 0.0])
        step1["e_hat"] = np.array([10.0, 0.0, 0.0])
        step1["v_hat"] = np.array([-1.0, 0.0, 0.0])
        step1["enemy_pos"] = np.array([10.0, 0.0, 0.0])
        step1["enemy_vel"] = np.array([-1.0, 0.0, 0.0])

        step2 = _base_info(defender_pos, defender_vel)
        step2["enemy_measurement"] = np.array([10.5, 0.0, 0.0])
        step2["e_hat"] = np.array([10.5, 0.0, 0.0])
        step2["v_hat"] = np.array([-1.0, 0.0, 0.0])
        step2["enemy_pos"] = np.array([10.5, 0.0, 0.0])
        step2["enemy_vel"] = np.array([-1.0, 0.0, 0.0])

        policy.reset()
        policy.act(None, step1)
        action_a = policy.act(None, step2)

        step2_leaked = dict(step2)
        step2_leaked["enemy_pos"] = np.array([-500.0, 300.0, 700.0])
        step2_leaked["enemy_vel"] = np.array([50.0, -80.0, 20.0])

        policy.reset()
        policy.act(None, step1)
        action_b = policy.act(None, step2_leaked)

        assert np.allclose(action_a, action_b, atol=1e-6), (
            state_source, action_a, action_b
        )


# ---------------------------------------------------------------------------
# H. LOS geometry
# ---------------------------------------------------------------------------
def test_los_geometry():
    policy = ProportionalNavigationPolicy(state_source="kalman")
    defender_pos = np.zeros(3)
    defender_vel = np.array([5.0, 0.0, 0.0])  # above low-speed threshold (0.1 * 18 = 1.8)

    info = _base_info(defender_pos, defender_vel)
    info["e_hat"] = np.array([10.0, 0.0, 0.0])
    info["v_hat"] = np.zeros(3)

    policy.act(None, info)
    assert abs(policy.last_range - 10.0) < 1e-6, policy.last_range
    assert np.allclose(policy.last_los_unit, [1.0, 0.0, 0.0], atol=1e-5), policy.last_los_unit


# ---------------------------------------------------------------------------
# I. Closing speed
# ---------------------------------------------------------------------------
def test_closing_speed():
    policy = ProportionalNavigationPolicy(state_source="kalman")
    defender_pos = np.zeros(3)
    defender_vel = np.array([5.0, 0.0, 0.0])
    target_pos = np.array([10.0, 0.0, 0.0])
    target_vel = np.zeros(3)  # v_rel = target_vel - defender_vel = [-5,0,0]

    info = _base_info(defender_pos, defender_vel)
    info["e_hat"] = target_pos
    info["v_hat"] = target_vel

    policy.act(None, info)
    assert abs(policy.last_closing_speed - 5.0) < 1e-5, policy.last_closing_speed


# ---------------------------------------------------------------------------
# J. Zero LOS-rate geometry (pure radial closing)
# ---------------------------------------------------------------------------
def test_zero_los_rate():
    policy = ProportionalNavigationPolicy(state_source="kalman", navigation_constant=3.0)
    defender_pos = np.zeros(3)
    defender_vel = np.array([5.0, 0.0, 0.0])
    target_pos = np.array([10.0, 0.0, 0.0])
    target_vel = np.zeros(3)  # v_rel = [-5, 0, 0], purely radial

    info = _base_info(defender_pos, defender_vel)
    info["e_hat"] = target_pos
    info["v_hat"] = target_vel

    policy.act(None, info)
    assert policy.last_guidance_mode == "pn", policy.last_guidance_mode
    assert np.allclose(policy.last_los_rate, [0.0, 0.0, 0.0], atol=1e-6), policy.last_los_rate
    assert np.allclose(policy.last_pn_acceleration, [0.0, 0.0, 0.0], atol=1e-6), policy.last_pn_acceleration


# ---------------------------------------------------------------------------
# K. Nonzero LOS-rate geometry (check sign/orientation, not just nonzero)
# ---------------------------------------------------------------------------
def test_nonzero_los_rate():
    N = 3.0
    policy = ProportionalNavigationPolicy(state_source="kalman", navigation_constant=N)
    defender_pos = np.zeros(3)
    defender_vel = np.array([5.0, -2.0, 0.0])
    target_pos = np.array([10.0, 0.0, 0.0])
    target_vel = np.zeros(3)  # v_rel = target_vel - defender_vel = [-5, 2, 0]

    info = _base_info(defender_pos, defender_vel)
    info["e_hat"] = target_pos
    info["v_hat"] = target_vel

    policy.act(None, info)
    expected = _expected_pn(defender_pos, defender_vel, target_pos, target_vel, N)

    assert policy.last_guidance_mode == "pn"
    assert np.allclose(policy.last_los_rate, expected["omega_los"], atol=1e-6), (
        policy.last_los_rate, expected["omega_los"]
    )
    # Expected omega_LOS = [0, 0, 0.2] (nonzero z component, sign as derived)
    assert abs(policy.last_los_rate[2] - 0.2) < 1e-6, policy.last_los_rate
    assert np.allclose(policy.last_pn_acceleration, expected["a_pn"], atol=1e-5), (
        policy.last_pn_acceleration, expected["a_pn"]
    )
    # Expected a_PN = [0, 3, 0]: positive-y (matches target drifting toward +y
    # relative to defender -- PN corrects toward that drift direction).
    assert policy.last_pn_acceleration[1] > 0, policy.last_pn_acceleration


# ---------------------------------------------------------------------------
# L. Truly 3-D PN (non-coplanar geometry)
# ---------------------------------------------------------------------------
def test_3d_pn_noncoplanar():
    N = 3.0
    policy = ProportionalNavigationPolicy(state_source="kalman", navigation_constant=N)
    defender_pos = np.zeros(3)
    defender_vel = np.array([5.0, -2.0, 1.0])
    target_pos = np.array([10.0, 0.0, 10.0])
    target_vel = np.zeros(3)  # v_rel = [-5, 2, -1]

    info = _base_info(defender_pos, defender_vel)
    info["e_hat"] = target_pos
    info["v_hat"] = target_vel

    policy.act(None, info)
    expected = _expected_pn(defender_pos, defender_vel, target_pos, target_vel, N)

    assert policy.last_guidance_mode == "pn"
    assert policy.last_los_rate.shape == (3,)
    assert np.all(np.isfinite(policy.last_los_rate))
    assert np.all(np.isfinite(policy.last_pn_acceleration))
    assert np.allclose(policy.last_los_rate, expected["omega_los"], atol=1e-6)
    assert np.allclose(policy.last_pn_acceleration, expected["a_pn"], atol=1e-5)
    # The vertical (z) component must be free to be nonzero (not collapsed to XY-only).
    assert abs(policy.last_pn_acceleration[2]) > 1e-6, policy.last_pn_acceleration


# ---------------------------------------------------------------------------
# M. PN acceleration is lateral (perpendicular to LOS)
# ---------------------------------------------------------------------------
def test_pn_acceleration_lateral():
    policy = ProportionalNavigationPolicy(state_source="kalman")
    defender_pos = np.zeros(3)
    defender_vel = np.array([5.0, -2.0, 1.0])
    target_pos = np.array([10.0, 0.0, 10.0])
    target_vel = np.zeros(3)

    info = _base_info(defender_pos, defender_vel)
    info["e_hat"] = target_pos
    info["v_hat"] = target_vel

    policy.act(None, info)
    assert policy.last_guidance_mode == "pn"
    dot = float(np.dot(policy.last_pn_acceleration, policy.last_los_unit))
    assert abs(dot) < 1e-4, dot


# ---------------------------------------------------------------------------
# N. Navigation-constant scaling
# ---------------------------------------------------------------------------
def test_navigation_constant_scaling():
    defender_pos = np.zeros(3)
    defender_vel = np.array([5.0, -2.0, 0.0])
    target_pos = np.array([10.0, 0.0, 0.0])
    target_vel = np.zeros(3)

    def run(N):
        policy = ProportionalNavigationPolicy(state_source="kalman", navigation_constant=N)
        info = _base_info(defender_pos, defender_vel)
        info["e_hat"] = target_pos
        info["v_hat"] = target_vel
        policy.act(None, info)
        assert policy.last_guidance_mode == "pn"
        return np.linalg.norm(policy.last_pn_acceleration)

    mag_n2 = run(2.0)
    mag_n4 = run(4.0)
    assert mag_n2 > 1e-8
    assert abs(mag_n4 / mag_n2 - 2.0) < 1e-4, (mag_n2, mag_n4)


# ---------------------------------------------------------------------------
# O. Non-closing fallback
# ---------------------------------------------------------------------------
def test_nonclosing_fallback():
    policy = ProportionalNavigationPolicy(state_source="kalman")
    defender_pos = np.zeros(3)
    defender_vel = np.array([5.0, 0.0, 0.0])  # above low-speed threshold
    target_pos = np.array([10.0, 0.0, 0.0])
    target_vel = np.array([20.0, 0.0, 0.0])  # target moving away faster than closing

    info = _base_info(defender_pos, defender_vel)
    info["e_hat"] = target_pos
    info["v_hat"] = target_vel

    action = policy.act(None, info)
    assert policy.last_guidance_mode == "nonclosing_fallback"
    assert policy.last_closing_speed <= 0.0, policy.last_closing_speed
    assert np.allclose(action, [1.0, 0.0, 0.0], atol=1e-5), action  # pure pursuit toward target


# ---------------------------------------------------------------------------
# P. Low-speed startup fallback
# ---------------------------------------------------------------------------
def test_low_speed_fallback_and_activation():
    policy = ProportionalNavigationPolicy(state_source="kalman", min_pn_speed_fraction=0.10)
    defender_pos = np.zeros(3)
    target_pos = np.array([10.0, 0.0, 0.0])
    target_vel = np.zeros(3)

    # Below threshold: 0.10 * v_d(18.0) = 1.8
    low_vel = np.array([1.0, 0.0, 0.0])
    info_low = _base_info(defender_pos, low_vel)
    info_low["e_hat"] = target_pos
    info_low["v_hat"] = target_vel
    policy.act(None, info_low)
    assert policy.last_guidance_mode == "low_speed_fallback"

    # Above threshold, otherwise identical well-defined PN geometry
    high_vel = np.array([5.0, 0.0, 0.0])
    info_high = _base_info(defender_pos, high_vel)
    info_high["e_hat"] = target_pos
    info_high["v_hat"] = target_vel
    policy.act(None, info_high)
    assert policy.last_guidance_mode == "pn"


# ---------------------------------------------------------------------------
# Q. Degenerate range
# ---------------------------------------------------------------------------
def test_degenerate_range():
    policy = ProportionalNavigationPolicy(state_source="kalman")
    defender_pos = np.array([5.0, 5.0, 5.0])
    defender_vel = np.array([5.0, 0.0, 0.0])
    target_pos = defender_pos + 1e-10  # numerically co-located

    info = _base_info(defender_pos, defender_vel)
    info["e_hat"] = target_pos
    info["v_hat"] = np.zeros(3)

    action = policy.act(None, info)
    assert policy.last_guidance_mode == "degenerate_fallback"
    assert np.all(np.isfinite(action))
    assert np.allclose(action, [0.0, 0.0, 0.0]), action


# ---------------------------------------------------------------------------
# Section 26: PN through actual constrained dynamics
# ---------------------------------------------------------------------------
def test_pn_respects_constrained_dynamics():
    config = EnvConfig(use_kalman_tracking=True)
    env = SoldierEnv(config=config)
    policy = ProportionalNavigationPolicy(state_source="kalman")
    obs, info = env.reset(seed=21)
    policy.reset()

    for _ in range(500):
        action = policy.act(obs, info)
        obs, r, term, trunc, info = env.step(action)

        assert info["defender_speed"] <= config.v_d + 1e-4, info["defender_speed"]
        assert info["defender_accel"] <= config.defender_max_accel + 1e-4, info["defender_accel"]
        assert info["defender_turn_rate"] <= config.defender_max_turn_rate_deg + 1e-2, info["defender_turn_rate"]
        assert info["defender_vel"][2] <= config.defender_max_climb_rate + 1e-4, info["defender_vel"]
        assert info["defender_vel"][2] >= -config.defender_max_descent_rate - 1e-4, info["defender_vel"]

        if term:
            obs, info = env.reset()
            policy.reset()


# ---------------------------------------------------------------------------
# Section 27: PN with hostile evasion enabled (default config)
# ---------------------------------------------------------------------------
def test_pn_with_hostile_evasion():
    for state_source, use_kalman in (("measurement", False), ("kalman", True)):
        config = EnvConfig(use_kalman_tracking=use_kalman, enemy_evasion_enabled=True)
        env = SoldierEnv(config=config)
        policy = ProportionalNavigationPolicy(state_source=state_source)
        obs, info = env.reset(seed=7)
        policy.reset()

        saw_evasion = False
        for _ in range(600):
            action = policy.act(obs, info)
            assert np.all(np.isfinite(action)), action
            obs, r, term, trunc, info = env.step(action)

            assert np.all(np.isfinite(obs))
            assert np.isfinite(r)
            if info.get("enemy_evasion_active"):
                saw_evasion = True

            assert info["defender_speed"] <= config.v_d + 1e-4
            assert info["defender_accel"] <= config.defender_max_accel + 1e-4
            assert info["defender_turn_rate"] <= config.defender_max_turn_rate_deg + 1e-2

            if state_source == "measurement" and policy.last_target_velocity_estimate is not None:
                assert np.all(np.isfinite(policy.last_target_velocity_estimate))

            if term:
                obs, info = env.reset()
                policy.reset()

        # Not a hard requirement, but report-worthy; do not fail the suite on this.
        _ = saw_evasion


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        ("A. policy construction", test_construction),
        ("B. action dimensions", test_action_dimensions),
        ("C. pre-detection escort", test_pre_detection_escort),
        ("D. measurement finite-difference velocity", test_measurement_finite_difference),
        ("E. reset clears measurement history", test_reset_clears_history),
        ("F. kalman mode uses v_hat", test_kalman_uses_v_hat),
        ("G. ground-truth leakage test (mandatory)", test_ground_truth_leakage),
        ("H. LOS geometry", test_los_geometry),
        ("I. closing speed", test_closing_speed),
        ("J. zero LOS-rate geometry", test_zero_los_rate),
        ("K. nonzero LOS-rate geometry (sign/orientation)", test_nonzero_los_rate),
        ("L. truly 3D PN (non-coplanar)", test_3d_pn_noncoplanar),
        ("M. PN acceleration is lateral", test_pn_acceleration_lateral),
        ("N. navigation-constant scaling", test_navigation_constant_scaling),
        ("O. non-closing fallback", test_nonclosing_fallback),
        ("P. low-speed startup fallback / activation", test_low_speed_fallback_and_activation),
        ("Q. degenerate range", test_degenerate_range),
        ("PN respects constrained dynamics", test_pn_respects_constrained_dynamics),
        ("PN stable with hostile evasion enabled", test_pn_with_hostile_evasion),
    ]

    print("=== Proportional Navigation (PN) Correctness Tests ===\n")
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
