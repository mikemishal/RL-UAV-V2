"""Lead-intercept (constant-velocity predictive interception) correctness tests.

Script-style test consistent with the project's existing test files
(test_proportional_navigation.py, test_hostile_evasion.py, etc). No external
test framework is used; each test function performs assertions and the
runner reports PASS/FAIL per test with a final summary.

Covers the 3-D constant-velocity lead-intercept guidance law
(LeadInterceptPolicy): the quadratic intercept-time solver (normal,
near-linear, and roundoff-degenerate branches), ground-truth independence,
both sensing modes (measurement/kalman), fallback behavior, and integration
with the environment's existing constrained dynamics and reactive hostile
evasion.

Run directly:
    python test_lead_intercept.py
"""

from __future__ import annotations

import numpy as np

from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy


def _base_info(defender_pos, defender_vel=None, soldier_pos=None, enemy_detected=True):
    return {
        "defender_pos": np.asarray(defender_pos, dtype=np.float64),
        "defender_vel": np.zeros(3) if defender_vel is None else np.asarray(defender_vel, dtype=np.float64),
        "soldier_pos": np.zeros(3) if soldier_pos is None else np.asarray(soldier_pos, dtype=np.float64),
        "enemy_detected": enemy_detected,
    }


def _residual(r, v, t, s):
    """|| r + v t || - s t, the interception-equation residual."""
    return float(np.linalg.norm(np.asarray(r) + np.asarray(v) * t) - s * t)


# ---------------------------------------------------------------------------
# A. Construction
# ---------------------------------------------------------------------------
def test_construction():
    LeadInterceptPolicy(state_source="measurement")
    LeadInterceptPolicy(state_source="kalman")

    for bad in ("invalid", "", "Measurement"):
        raised = False
        try:
            LeadInterceptPolicy(state_source=bad)
        except ValueError:
            raised = True
        assert raised, bad

    for bad_dt in (0.0, -0.5):
        raised = False
        try:
            LeadInterceptPolicy(dt=bad_dt)
        except ValueError:
            raised = True
        assert raised, bad_dt

    for bad_vd in (0.0, -18.0):
        raised = False
        try:
            LeadInterceptPolicy(v_d=bad_vd)
        except ValueError:
            raised = True
        assert raised, bad_vd

    for bad_eps in (0.0, -1e-8):
        raised = False
        try:
            LeadInterceptPolicy(eps=bad_eps)
        except ValueError:
            raised = True
        assert raised, bad_eps


# ---------------------------------------------------------------------------
# B. Action validity (through real environment rollouts)
# ---------------------------------------------------------------------------
def test_action_validity():
    for state_source in ("measurement", "kalman"):
        policy = LeadInterceptPolicy(state_source=state_source)
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
    policy = LeadInterceptPolicy()
    info = _base_info([0.0, 0.0, 0.0], soldier_pos=[5.0, 0.0, 0.0], enemy_detected=False)
    action = policy.act(None, info)
    assert np.allclose(action, [1.0, 0.0, 0.0], atol=1e-5), action
    assert policy.last_guidance_mode == "escort"

    info2 = dict(info)
    info2["soldier_pos"] = np.array([0.0, 0.0, 0.0])
    action2 = policy.act(None, info2)
    assert np.allclose(action2, [0.0, 0.0, 0.0]), action2


# ---------------------------------------------------------------------------
# D. Finite-difference velocity
# ---------------------------------------------------------------------------
def test_finite_difference_velocity():
    policy = LeadInterceptPolicy(state_source="measurement", dt=0.5)
    defender_pos = np.zeros(3)
    z0 = np.array([10.0, 5.0, 20.0])
    z1 = np.array([11.0, 4.5, 20.25])

    info0 = _base_info(defender_pos)
    info0["enemy_measurement"] = z0
    policy.act(None, info0)
    assert policy.last_guidance_mode == "first_measurement_fallback"
    assert policy.last_target_velocity_estimate is None

    info1 = _base_info(defender_pos)
    info1["enemy_measurement"] = z1
    policy.act(None, info1)

    expected_vel = (z1 - z0) / 0.5
    assert np.allclose(policy.last_target_velocity_estimate, expected_vel, atol=1e-4), (
        policy.last_target_velocity_estimate, expected_vel
    )


# ---------------------------------------------------------------------------
# E. Reset clears measurement history
# ---------------------------------------------------------------------------
def test_reset_clears_history():
    policy = LeadInterceptPolicy(state_source="measurement", dt=0.5)
    defender_pos = np.zeros(3)
    z0 = np.array([10.0, 5.0, 20.0])

    info0 = _base_info(defender_pos)
    info0["enemy_measurement"] = z0
    policy.act(None, info0)

    policy.reset()

    z1 = np.array([11.0, 4.5, 20.25])
    info1 = _base_info(defender_pos)
    info1["enemy_measurement"] = z1
    policy.act(None, info1)

    assert policy.last_guidance_mode == "first_measurement_fallback"
    assert policy.last_target_velocity_estimate is None


# ---------------------------------------------------------------------------
# F. Kalman information separation
# ---------------------------------------------------------------------------
def test_kalman_information_separation():
    policy = LeadInterceptPolicy(state_source="kalman")
    defender_pos = np.zeros(3)
    e_hat = np.array([10.0, 0.0, 0.0])
    v_hat = np.array([-1.0, 0.0, 0.0])

    info = _base_info(defender_pos)
    info["e_hat"] = e_hat
    info["v_hat"] = v_hat
    info["enemy_measurement"] = np.array([999.0, 999.0, 999.0])

    action1 = policy.act(None, info)
    t1 = policy.last_intercept_time

    info2 = dict(info)
    info2["enemy_measurement"] = np.array([-999.0, -999.0, -999.0])
    action2 = policy.act(None, info2)
    t2 = policy.last_intercept_time

    assert np.allclose(action1, action2, atol=1e-6)
    assert abs(t1 - t2) < 1e-9


# ---------------------------------------------------------------------------
# G. Ground-truth leakage test (mandatory)
# ---------------------------------------------------------------------------
def test_ground_truth_leakage():
    for state_source in ("measurement", "kalman"):
        policy = LeadInterceptPolicy(state_source=state_source)
        defender_pos = np.zeros(3)

        step1 = _base_info(defender_pos)
        step1["enemy_measurement"] = np.array([10.0, 0.0, 0.0])
        step1["e_hat"] = np.array([10.0, 0.0, 0.0])
        step1["v_hat"] = np.array([-1.0, 0.0, 0.0])
        step1["enemy_pos"] = np.array([10.0, 0.0, 0.0])
        step1["enemy_vel"] = np.array([-1.0, 0.0, 0.0])

        step2 = _base_info(defender_pos)
        step2["enemy_measurement"] = np.array([10.5, 0.0, 0.0])
        step2["e_hat"] = np.array([10.5, 0.0, 0.0])
        step2["v_hat"] = np.array([-1.0, 0.0, 0.0])
        step2["enemy_pos"] = np.array([10.5, 0.0, 0.0])
        step2["enemy_vel"] = np.array([-1.0, 0.0, 0.0])

        policy.reset()
        policy.act(None, step1)
        action_a = policy.act(None, step2)
        t_a = policy.last_intercept_time
        p_a = policy.last_intercept_point

        step2_leaked = dict(step2)
        step2_leaked["enemy_pos"] = np.array([-500.0, 300.0, 700.0])
        step2_leaked["enemy_vel"] = np.array([50.0, -80.0, 20.0])

        policy.reset()
        policy.act(None, step1)
        action_b = policy.act(None, step2_leaked)
        t_b = policy.last_intercept_time
        p_b = policy.last_intercept_point

        assert np.allclose(action_a, action_b, atol=1e-6), (state_source, action_a, action_b)
        assert abs(t_a - t_b) < 1e-9, (state_source, t_a, t_b)
        assert np.allclose(p_a, p_b, atol=1e-4), (state_source, p_a, p_b)


# ---------------------------------------------------------------------------
# H. Stationary target
# ---------------------------------------------------------------------------
def test_stationary_target():
    policy = LeadInterceptPolicy(state_source="kalman", v_d=18.0)
    info = _base_info([0.0, 0.0, 0.0])
    info["e_hat"] = np.array([18.0, 0.0, 0.0])
    info["v_hat"] = np.array([0.0, 0.0, 0.0])

    action = policy.act(None, info)
    assert abs(policy.last_quadratic_a - (-324.0)) < 1e-6, policy.last_quadratic_a
    assert abs(policy.last_quadratic_b - 0.0) < 1e-6, policy.last_quadratic_b
    assert abs(policy.last_quadratic_c - 324.0) < 1e-6, policy.last_quadratic_c
    assert policy.last_guidance_mode == "lead"
    assert abs(policy.last_intercept_time - 1.0) < 1e-6, policy.last_intercept_time
    assert np.allclose(policy.last_intercept_point, [18.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(action, [1.0, 0.0, 0.0], atol=1e-5), action


# ---------------------------------------------------------------------------
# I. Target moving directly away
# ---------------------------------------------------------------------------
def test_target_moving_away():
    policy = LeadInterceptPolicy(state_source="kalman", v_d=10.0)
    info = _base_info([0.0, 0.0, 0.0])
    info["e_hat"] = np.array([10.0, 0.0, 0.0])
    info["v_hat"] = np.array([5.0, 0.0, 0.0])

    action = policy.act(None, info)
    assert policy.last_guidance_mode == "lead"
    assert abs(policy.last_intercept_time - 2.0) < 1e-6, policy.last_intercept_time
    assert np.allclose(policy.last_intercept_point, [20.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(action, [1.0, 0.0, 0.0], atol=1e-5), action


# ---------------------------------------------------------------------------
# J. Target moving toward interceptor
# ---------------------------------------------------------------------------
def test_target_moving_toward():
    policy = LeadInterceptPolicy(state_source="kalman", v_d=10.0)
    info = _base_info([0.0, 0.0, 0.0])
    info["e_hat"] = np.array([10.0, 0.0, 0.0])
    info["v_hat"] = np.array([-5.0, 0.0, 0.0])

    action = policy.act(None, info)
    assert policy.last_guidance_mode == "lead"
    assert abs(policy.last_intercept_time - (2.0 / 3.0)) < 1e-5, policy.last_intercept_time
    expected_point = np.array([10.0, 0.0, 0.0]) + np.array([-5.0, 0.0, 0.0]) * (2.0 / 3.0)
    assert np.allclose(policy.last_intercept_point, expected_point, atol=1e-4)


# ---------------------------------------------------------------------------
# K. Lateral lead
# ---------------------------------------------------------------------------
def test_lateral_lead():
    policy = LeadInterceptPolicy(state_source="kalman", v_d=10.0)
    info = _base_info([0.0, 0.0, 0.0])
    info["e_hat"] = np.array([10.0, 0.0, 0.0])
    info["v_hat"] = np.array([0.0, 5.0, 0.0])

    action = policy.act(None, info)
    expected_t = np.sqrt(100.0 / 75.0)
    assert abs(policy.last_intercept_time - expected_t) < 1e-6, policy.last_intercept_time
    expected_point = np.array([10.0, 5.0 * expected_t, 0.0])
    assert np.allclose(policy.last_intercept_point, expected_point, atol=1e-4), (
        policy.last_intercept_point, expected_point
    )
    assert action[0] > 0.0 and action[1] > 0.0, action
    # Must not simply point at the current (unlead) target location [10,0,0]:
    current_dir = np.array([1.0, 0.0, 0.0])
    assert not np.allclose(action, current_dir, atol=0.05), action


# ---------------------------------------------------------------------------
# L. True 3-D lead
# ---------------------------------------------------------------------------
def test_3d_lead():
    policy = LeadInterceptPolicy(state_source="kalman", v_d=18.0)
    info = _base_info([0.0, 0.0, 0.0])
    info["e_hat"] = np.array([10.0, 0.0, 10.0])
    info["v_hat"] = np.array([0.0, 2.0, -1.0])

    action = policy.act(None, info)
    assert policy.last_guidance_mode == "lead", policy.last_guidance_mode
    assert policy.last_intercept_time > 0 and np.isfinite(policy.last_intercept_time)
    assert policy.last_intercept_point.shape == (3,)
    assert np.all(np.isfinite(policy.last_intercept_point))
    assert abs(action[2]) > 1e-6, action  # meaningful vertical component, no XY-only collapse


# ---------------------------------------------------------------------------
# M. Intercept-equation residual (critical correctness check)
# ---------------------------------------------------------------------------
def test_intercept_equation_residual():
    cases = [
        ([0.0, 0.0, 0.0], [18.0, 0.0, 0.0], [0.0, 0.0, 0.0], 18.0),
        ([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.0, 0.0], 10.0),
        ([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [-5.0, 0.0, 0.0], 10.0),
        ([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 5.0, 0.0], 10.0),
        ([0.0, 0.0, 0.0], [10.0, 0.0, 10.0], [0.0, 2.0, -1.0], 18.0),
        ([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [-20.0, 0.0, 0.0], 10.0),  # faster, approaching
    ]
    for pd, pt, vt, s in cases:
        policy = LeadInterceptPolicy(state_source="kalman", v_d=s)
        info = _base_info(pd)
        info["e_hat"] = np.array(pt)
        info["v_hat"] = np.array(vt)
        policy.act(None, info)
        assert policy.last_guidance_mode == "lead", (pd, pt, vt, s, policy.last_guidance_mode)
        r = np.array(pt) - np.array(pd)
        residual = _residual(r, vt, policy.last_intercept_time, s)
        assert abs(residual) < 1e-4, (pd, pt, vt, s, residual)


# ---------------------------------------------------------------------------
# N. Target too fast moving away -> no solution
# ---------------------------------------------------------------------------
def test_target_too_fast_moving_away():
    policy = LeadInterceptPolicy(state_source="kalman", v_d=10.0)
    info = _base_info([0.0, 0.0, 0.0])
    info["e_hat"] = np.array([10.0, 0.0, 0.0])
    info["v_hat"] = np.array([20.0, 0.0, 0.0])

    action = policy.act(None, info)
    assert policy.last_guidance_mode in (
        "no_real_solution_fallback", "no_positive_root_fallback"
    ), policy.last_guidance_mode
    # Pure-pursues the current legitimate target estimate (e_hat), not the intercept point.
    assert np.allclose(action, [1.0, 0.0, 0.0], atol=1e-5), action


# ---------------------------------------------------------------------------
# O. Faster target moving toward defender -> solution DOES exist
# ---------------------------------------------------------------------------
def test_faster_target_approaching_has_solution():
    policy = LeadInterceptPolicy(state_source="kalman", v_d=10.0)
    info = _base_info([0.0, 0.0, 0.0])
    info["e_hat"] = np.array([10.0, 0.0, 0.0])
    info["v_hat"] = np.array([-20.0, 0.0, 0.0])  # target speed 20 > defender speed 10

    action = policy.act(None, info)
    assert policy.last_guidance_mode == "lead", policy.last_guidance_mode
    assert policy.last_intercept_time > 0 and np.isfinite(policy.last_intercept_time)
    r = np.array([10.0, 0.0, 0.0])
    residual = _residual(r, [-20.0, 0.0, 0.0], policy.last_intercept_time, 10.0)
    assert abs(residual) < 1e-4, residual


# ---------------------------------------------------------------------------
# P. Near-linear case (|v| ~ s, a ~ 0)
# ---------------------------------------------------------------------------
def test_near_linear_case():
    s = 10.0
    policy = LeadInterceptPolicy(state_source="kalman", v_d=s)
    info = _base_info([0.0, 0.0, 0.0])
    info["e_hat"] = np.array([10.0, 0.0, 0.0])
    info["v_hat"] = np.array([9.999999, 0.0, 0.0])  # ||v|| ~ s -> a ~ 0

    action = policy.act(None, info)
    assert np.all(np.isfinite(action)), action
    # a should indeed be nearly zero for this constructed geometry
    assert abs(policy.last_quadratic_a) < 1e-3, policy.last_quadratic_a
    if policy.last_guidance_mode == "lead":
        assert np.isfinite(policy.last_intercept_time)
        assert policy.last_intercept_time > 0

    # Also directly unit-test the static solver's near-linear branch with a == 0 exactly.
    t, disc, reason = LeadInterceptPolicy._solve_intercept_time(a=0.0, b=-2.0, c=10.0, eps=1e-8)
    assert reason == "ok"
    assert abs(t - 5.0) < 1e-9, t  # b t + c = 0 -> t = -c/b = -10/-2 = 5


# ---------------------------------------------------------------------------
# Q. Negative-root filtering
# ---------------------------------------------------------------------------
def test_negative_root_filtering():
    # a=1, roots at t=-3 and t=2 (t^2 + t - 6 = 0)
    t, disc, reason = LeadInterceptPolicy._solve_intercept_time(a=1.0, b=1.0, c=-6.0, eps=1e-8)
    assert reason == "ok"
    assert abs(t - 2.0) < 1e-9, t


# ---------------------------------------------------------------------------
# R. No positive root
# ---------------------------------------------------------------------------
def test_no_positive_root():
    # a=1, roots at t=-2 and t=-3 (both negative): t^2+5t+6=0
    t, disc, reason = LeadInterceptPolicy._solve_intercept_time(a=1.0, b=5.0, c=6.0, eps=1e-8)
    assert reason == "no_positive_root"
    assert t is None


# ---------------------------------------------------------------------------
# S. Discriminant round-off tolerance
# ---------------------------------------------------------------------------
def test_discriminant_roundoff():
    eps = 1e-6
    # Construct a,b,c with discriminant slightly negative due to roundoff only.
    a, b, c = 1.0, 2.0, 1.0 + 1e-9  # true discriminant = 4 - 4*(1+1e-9) = -4e-9 (tiny negative)
    t, disc, reason = LeadInterceptPolicy._solve_intercept_time(a, b, c, eps)
    assert reason != "no_real_solution", (disc, reason)  # tiny negative must be clamped, not rejected

    # A materially negative discriminant must still be rejected.
    a2, b2, c2 = 1.0, 0.0, 100.0  # discriminant = -400
    t2, disc2, reason2 = LeadInterceptPolicy._solve_intercept_time(a2, b2, c2, eps)
    assert reason2 == "no_real_solution", (disc2, reason2)
    assert t2 is None


# ---------------------------------------------------------------------------
# T. Large intercept time (no arbitrary horizon truncation)
# ---------------------------------------------------------------------------
def test_large_intercept_time():
    s = 10.0
    policy = LeadInterceptPolicy(state_source="kalman", v_d=s)
    info = _base_info([0.0, 0.0, 0.0])
    info["e_hat"] = np.array([10.0, 0.0, 0.0])
    info["v_hat"] = np.array([9.9999, 0.0, 0.0])  # near-speed-matched pursuit-style chase

    action = policy.act(None, info)
    if policy.last_guidance_mode == "lead":
        assert np.isfinite(policy.last_intercept_time)
        assert policy.last_intercept_time > 10.0, policy.last_intercept_time  # large, not truncated
        assert np.all(np.isfinite(action))


# ---------------------------------------------------------------------------
# U. Predicted point is not boundary-clipped
# ---------------------------------------------------------------------------
def test_predicted_point_not_clipped():
    config = EnvConfig()
    policy = LeadInterceptPolicy(state_source="kalman", v_d=100.0)  # large s to force point outside domain
    info = _base_info([0.0, 0.0, 0.0])
    # Target far outside the default engagement volume (L=50) with velocity
    # chosen so the mathematical intercept point remains outside the box.
    info["e_hat"] = np.array([200.0, 0.0, 10.0])
    info["v_hat"] = np.array([50.0, 0.0, 0.0])

    policy.act(None, info)
    assert policy.last_guidance_mode == "lead", policy.last_guidance_mode
    p = policy.last_intercept_point
    assert p[0] > config.L, p  # outside the horizontal domain -- NOT clipped by the policy


# ---------------------------------------------------------------------------
# V. Constrained-dynamics integration
# ---------------------------------------------------------------------------
def test_lead_respects_constrained_dynamics():
    config = EnvConfig(use_kalman_tracking=True)
    env = SoldierEnv(config=config)
    policy = LeadInterceptPolicy(state_source="kalman")
    obs, info = env.reset(seed=11)
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
# W. Hostile evasion integration
# ---------------------------------------------------------------------------
def test_lead_with_hostile_evasion():
    for state_source, use_kalman in (("measurement", False), ("kalman", True)):
        config = EnvConfig(use_kalman_tracking=use_kalman, enemy_evasion_enabled=True)
        env = SoldierEnv(config=config)
        policy = LeadInterceptPolicy(state_source=state_source)
        obs, info = env.reset(seed=13)
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

            if policy.last_target_velocity_estimate is not None:
                assert np.all(np.isfinite(policy.last_target_velocity_estimate))
            if policy.last_intercept_time is not None:
                assert np.isfinite(policy.last_intercept_time)

            if term:
                obs, info = env.reset()
                policy.reset()

        _ = saw_evasion  # report-worthy, not a hard requirement


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        ("A. construction", test_construction),
        ("B. action validity", test_action_validity),
        ("C. pre-detection escort", test_pre_detection_escort),
        ("D. finite-difference velocity", test_finite_difference_velocity),
        ("E. reset clears measurement history", test_reset_clears_history),
        ("F. kalman information separation", test_kalman_information_separation),
        ("G. ground-truth leakage (mandatory)", test_ground_truth_leakage),
        ("H. stationary target", test_stationary_target),
        ("I. target moving away", test_target_moving_away),
        ("J. target moving toward interceptor", test_target_moving_toward),
        ("K. lateral lead", test_lateral_lead),
        ("L. true 3D lead", test_3d_lead),
        ("M. intercept-equation residual", test_intercept_equation_residual),
        ("N. target too fast moving away -> no solution", test_target_too_fast_moving_away),
        ("O. faster target approaching -> solution exists", test_faster_target_approaching_has_solution),
        ("P. near-linear case", test_near_linear_case),
        ("Q. negative-root filtering", test_negative_root_filtering),
        ("R. no positive root", test_no_positive_root),
        ("S. discriminant round-off", test_discriminant_roundoff),
        ("T. large intercept time (no horizon truncation)", test_large_intercept_time),
        ("U. predicted point not boundary-clipped", test_predicted_point_not_clipped),
        ("V. constrained-dynamics integration", test_lead_respects_constrained_dynamics),
        ("W. hostile evasion integration", test_lead_with_hostile_evasion),
    ]

    print("=== Lead-Intercept (Constant-Velocity Predictive) Correctness Tests ===\n")
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
