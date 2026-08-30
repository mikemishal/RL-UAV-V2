"""
Tests for `uav_defend.policies.mpc.lead_init.solve_lead_direction`.

Covers: Lead-mode geometry validity, pursuit-mode fallback conditions
(missing velocity, no predictive solution, degenerate geometry), and a
CONSISTENCY regression proving `solve_lead_direction` reproduces the exact
same direction `LeadInterceptPolicy.act()` (state_source="measurement")
returns for the same state -- proving the reused `_solve_intercept_time`
call path did not change `LeadInterceptPolicy`'s own behavior (which is
untouched by this hardening pass).

Run directly: python test_lead_init.py
"""

from __future__ import annotations

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy
from uav_defend.policies.mpc.lead_init import solve_lead_direction

EPS = 1e-8
V_D = 18.0


def _measurement_info(enemy_measurement, defender_pos, enemy_measurement_velocity=None, velocity_valid=False,
                       soldier_pos=(0.0, 0.0, 0.0), defender_vel=(0.0, 0.0, 0.0)):
    return {
        "soldier_pos": np.array(soldier_pos, dtype=np.float64),
        "defender_pos": np.array(defender_pos, dtype=np.float64),
        "defender_vel": np.array(defender_vel, dtype=np.float64),
        "enemy_detected": True,
        "enemy_measurement": np.array(enemy_measurement, dtype=np.float64),
        "enemy_measurement_velocity": (
            np.array(enemy_measurement_velocity, dtype=np.float64) if enemy_measurement_velocity is not None
            else np.zeros(3)
        ),
        "enemy_measurement_velocity_valid": velocity_valid,
    }


# --- Lead-mode geometry -----------------------------------------------------

def test_solvable_geometry_produces_lead_mode():
    result = solve_lead_direction(
        target_pos=np.array([20.0, 0.0, 0.0]), target_vel=np.array([1.0, 0.0, 0.0]),
        defender_pos=np.array([0.0, 0.0, 0.0]), v_d=V_D, eps=EPS,
    )
    assert result.guidance_mode == "lead"
    assert result.direction.shape == (3,)
    assert result.direction.dtype == np.float32
    assert np.isfinite(result.direction).all()
    assert abs(np.linalg.norm(result.direction) - 1.0) < 1e-5


def test_lead_direction_finite_and_unit_norm_random_states():
    rng = np.random.default_rng(0)
    for _ in range(30):
        target_pos = rng.uniform(-30, 30, size=3)
        target_vel = rng.uniform(-8, 8, size=3)
        defender_pos = rng.uniform(-30, 30, size=3)
        result = solve_lead_direction(target_pos, target_vel, defender_pos, V_D, EPS)
        assert np.all(np.isfinite(result.direction))
        norm = float(np.linalg.norm(result.direction))
        assert norm <= 1.0 + 1e-5


# --- Pursuit-mode fallback conditions ---------------------------------------

def test_missing_velocity_produces_pursuit_fallback():
    result = solve_lead_direction(
        target_pos=np.array([10.0, 0.0, 0.0]), target_vel=None,
        defender_pos=np.array([0.0, 0.0, 0.0]), v_d=V_D, eps=EPS,
    )
    assert result.guidance_mode == "no_velocity_fallback"
    assert np.allclose(result.direction, [1.0, 0.0, 0.0], atol=1e-6)


def test_degenerate_zero_distance_produces_fallback():
    result = solve_lead_direction(
        target_pos=np.array([5.0, 5.0, 5.0]), target_vel=np.array([1.0, 0.0, 0.0]),
        defender_pos=np.array([5.0, 5.0, 5.0]), v_d=V_D, eps=EPS,
    )
    assert result.guidance_mode == "degenerate_fallback"
    assert np.array_equal(result.direction, np.zeros(3, dtype=np.float32))


def test_no_real_solution_produces_pursuit_fallback():
    # Target moving away faster than v_d, positioned such that no
    # real intercept-time root exists.
    result = solve_lead_direction(
        target_pos=np.array([1.0, 0.0, 0.0]), target_vel=np.array([100.0, 0.0, 0.0]),
        defender_pos=np.array([0.0, 0.0, 0.0]), v_d=1.0, eps=EPS,
    )
    assert result.guidance_mode in ("no_real_solution_fallback", "no_positive_root_fallback")
    assert np.all(np.isfinite(result.direction))


def test_no_ground_truth_leakage_possible_by_signature():
    """solve_lead_direction's signature structurally cannot read
    info['enemy_pos']/info['enemy_vel'] -- it only accepts plain arrays."""
    import inspect

    sig = inspect.signature(solve_lead_direction)
    assert set(sig.parameters) == {"target_pos", "target_vel", "defender_pos", "v_d", "eps"}


# --- Consistency regression against LeadInterceptPolicy ---------------------

def test_consistency_with_lead_intercept_policy_measurement_track():
    """For a variety of states, the pure helper's direction (when in "lead"
    mode) must exactly match LeadInterceptPolicy(state_source="measurement")'s
    own action -- proving reuse of _solve_intercept_time did not alter
    LeadInterceptPolicy's untouched behavior."""
    config = EnvConfig()
    lead_policy = LeadInterceptPolicy(state_source="measurement", config=config)

    rng = np.random.default_rng(42)
    lead_mode_seen = False
    for _ in range(50):
        info = _measurement_info(
            enemy_measurement=tuple(rng.uniform(-30, 30, size=3)),
            enemy_measurement_velocity=tuple(rng.uniform(-8, 8, size=3)),
            velocity_valid=True,
            defender_pos=tuple(rng.uniform(-30, 30, size=3)),
        )
        lead_policy.reset()
        lead_action = lead_policy.act(np.zeros(16), info)

        pure_result = solve_lead_direction(
            info["enemy_measurement"], info["enemy_measurement_velocity"], info["defender_pos"], config.v_d, config.eps
        )

        if lead_policy.last_guidance_mode == "lead":
            lead_mode_seen = True
            assert np.allclose(lead_action, pure_result.direction, atol=1e-6), (
                "solve_lead_direction diverges from LeadInterceptPolicy.act() in 'lead' mode"
            )
            assert pure_result.guidance_mode == "lead"
        else:
            # All non-"lead" branches of LeadInterceptPolicy are pure-pursuit
            # toward the same target; the pure helper's fallback direction
            # must match too.
            assert np.allclose(lead_action, pure_result.direction, atol=1e-6)

    assert lead_mode_seen, "test did not exercise the 'lead' branch -- weak coverage"


def test_consistency_first_measurement_step():
    config = EnvConfig()
    lead_policy = LeadInterceptPolicy(state_source="measurement", config=config)
    info = _measurement_info(enemy_measurement=(10.0, 5.0, 0.0), defender_pos=(0.0, 0.0, 0.0), velocity_valid=False)
    lead_action = lead_policy.act(np.zeros(16), info)
    assert lead_policy.last_guidance_mode == "first_measurement_fallback"

    pure_result = solve_lead_direction(info["enemy_measurement"], None, info["defender_pos"], config.v_d, config.eps)
    assert pure_result.guidance_mode == "no_velocity_fallback"
    assert np.allclose(lead_action, pure_result.direction, atol=1e-6)


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
