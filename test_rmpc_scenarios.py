"""
Tests for the six-scenario hostile-motion uncertainty model
(`uav_defend.policies.mpc.hostile_scenarios`).

Covers: physical-limit satisfaction for all six scenarios (speed, implicit
accel/turn/climb/descent bounds via `advance_velocity`), open-loop
independence of S1-S5 from the candidate defender trajectory, closed-loop
dependence of S6 on the candidate defender trajectory, and -- critically
for publication fairness -- independence of ALL SIX scenarios from
`enemy_evasion_gain` / `enemy_evasion_radius` (these must have zero effect
on any scenario's predicted trajectory).

Run directly: python test_rmpc_scenarios.py
"""

from __future__ import annotations

import inspect

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.mpc import hostile_scenarios as hs

EPS = 1e-8


def _max_step_horizontal_speed(config: EnvConfig) -> float:
    return config.v_e


# --- Physical validity ----------------------------------------------------

def test_all_open_loop_scenarios_respect_speed_limit():
    config = EnvConfig()
    hostile_pos0 = np.array([0.0, 0.0, 10.0])
    hostile_vel0 = np.array([2.0, -1.0, 0.5])
    soldier_pos = np.array([50.0, 50.0, 0.0])
    horizon = 8

    for scenario_id in ("S1", "S2", "S3", "S4", "S5"):
        positions = hs.simulate_open_loop_scenario(
            scenario_id, hostile_pos0, hostile_vel0, soldier_pos, horizon, config
        )
        assert positions.shape == (horizon + 1, 3)
        # Implied per-step displacement speed must not exceed v_e (dynamics
        # rate-limits velocity itself, so this is an indirect but valid
        # check since position deltas equal velocity * dt).
        deltas = np.diff(positions, axis=0)
        speeds = np.linalg.norm(deltas, axis=1) / config.dt
        assert np.all(speeds <= config.v_e + 1e-6), f"{scenario_id} exceeded v_e: {speeds}"


def test_s4_climbs_and_s5_descends():
    config = EnvConfig()
    hostile_pos0 = np.array([0.0, 0.0, 10.0])
    hostile_vel0 = np.array([3.0, 0.0, 0.0])
    soldier_pos = np.array([50.0, 0.0, 0.0])
    horizon = 6

    s4 = hs.simulate_open_loop_scenario("S4", hostile_pos0, hostile_vel0, soldier_pos, horizon, config)
    s5 = hs.simulate_open_loop_scenario("S5", hostile_pos0, hostile_vel0, soldier_pos, horizon, config)

    assert s4[-1, 2] > hostile_pos0[2]
    assert s5[-1, 2] < hostile_pos0[2]

    # Vertical rate must never exceed the hostile's own climb/descent limits.
    vz_s4 = np.diff(s4[:, 2]) / config.dt
    vz_s5 = np.diff(s5[:, 2]) / config.dt
    assert np.all(vz_s4 <= config.enemy_max_climb_rate + 1e-6)
    assert np.all(vz_s5 >= -config.enemy_max_descent_rate - 1e-6)


def test_s2_s3_diverge_laterally_from_s1():
    config = EnvConfig()
    hostile_pos0 = np.array([0.0, 0.0, 0.0])
    hostile_vel0 = np.array([5.0, 0.0, 0.0])
    soldier_pos = np.array([100.0, 0.0, 0.0])
    horizon = 6

    s1 = hs.simulate_open_loop_scenario("S1", hostile_pos0, hostile_vel0, soldier_pos, horizon, config)
    s2 = hs.simulate_open_loop_scenario("S2", hostile_pos0, hostile_vel0, soldier_pos, horizon, config)
    s3 = hs.simulate_open_loop_scenario("S3", hostile_pos0, hostile_vel0, soldier_pos, horizon, config)

    # S2 (left) and S3 (right) must diverge in OPPOSITE lateral (y) directions.
    assert s2[-1, 1] > 0.5
    assert s3[-1, 1] < -0.5
    assert not np.allclose(s1[-1], s2[-1])
    assert not np.allclose(s1[-1], s3[-1])


def test_s6_respects_hostile_physical_limits():
    config = EnvConfig()
    hostile_pos0 = np.array([10.0, 10.0, 10.0])
    hostile_vel0 = np.array([1.0, 0.0, 0.0])
    horizon = 6
    # A defender trajectory chasing the hostile from behind.
    defender_positions = np.tile(np.array([0.0, 0.0, 10.0]), (horizon + 1, 1))
    defender_positions += np.arange(horizon + 1).reshape(-1, 1) * np.array([1.0, 1.0, 0.0])

    positions = hs.simulate_max_away_scenario(hostile_pos0, hostile_vel0, defender_positions, config)
    assert positions.shape == (horizon + 1, 3)
    deltas = np.diff(positions, axis=0)
    speeds = np.linalg.norm(deltas, axis=1) / config.dt
    assert np.all(speeds <= config.v_e + 1e-6)


def test_s6_moves_away_from_defender():
    config = EnvConfig()
    hostile_pos0 = np.array([0.0, 0.0, 0.0])
    hostile_vel0 = np.array([0.0, 0.0, 0.0])
    horizon = 5
    # Defender stationed close behind the hostile in -x.
    defender_positions = np.tile(np.array([-2.0, 0.0, 0.0]), (horizon + 1, 1))

    positions = hs.simulate_max_away_scenario(hostile_pos0, hostile_vel0, defender_positions, config)
    # Hostile should move in +x (away from the defender at -x).
    assert positions[-1, 0] > positions[0, 0]


# --- Open-loop (S1-S5) independence from candidate defender trajectory ----

def test_open_loop_scenarios_independent_of_defender_trajectory():
    """S1-S5 must be identical regardless of which candidate defender
    trajectory is being evaluated (they are open-loop by design)."""
    config = EnvConfig()
    hostile_pos0 = np.array([5.0, 5.0, 5.0])
    hostile_vel0 = np.array([1.0, 2.0, -0.5])
    soldier_pos = np.array([20.0, 20.0, 0.0])
    horizon = 5

    for scenario_id in ("S1", "S2", "S3", "S4", "S5"):
        positions_a = hs.simulate_open_loop_scenario(
            scenario_id, hostile_pos0, hostile_vel0, soldier_pos, horizon, config
        )
        positions_b = hs.simulate_open_loop_scenario(
            scenario_id, hostile_pos0, hostile_vel0, soldier_pos, horizon, config
        )
        assert np.array_equal(positions_a, positions_b)

    # simulate_all_scenarios must also produce identical S1-S5 regardless
    # of the (only-S6-relevant) defender_positions argument.
    defender_traj_1 = np.zeros((horizon + 1, 3))
    defender_traj_2 = np.ones((horizon + 1, 3)) * 100.0

    all_1 = hs.simulate_all_scenarios(hostile_pos0, hostile_vel0, soldier_pos, defender_traj_1, config)
    all_2 = hs.simulate_all_scenarios(hostile_pos0, hostile_vel0, soldier_pos, defender_traj_2, config)

    for scenario_id in ("S1", "S2", "S3", "S4", "S5"):
        assert np.array_equal(all_1[scenario_id], all_2[scenario_id])
    # S6 (closed-loop) MUST differ given very different defender trajectories.
    assert not np.array_equal(all_1["S6"], all_2["S6"])


# --- CRITICAL publication-fairness test: evasion-parameter independence ---

def test_scenarios_independent_of_enemy_evasion_gain_and_radius():
    """No scenario's predicted trajectory may depend on
    `enemy_evasion_gain` / `enemy_evasion_radius` -- these describe the
    simulator hostile's DECISION POLICY, not airframe physics, and must be
    excluded entirely from RMPC's scenario model (design doc Section 8.1)."""
    hostile_pos0 = np.array([3.0, -2.0, 4.0])
    hostile_vel0 = np.array([2.0, 1.0, 0.2])
    soldier_pos = np.array([30.0, 30.0, 0.0])
    horizon = 6
    defender_positions = np.tile(np.array([0.0, 0.0, 0.0]), (horizon + 1, 1))

    config_low = EnvConfig(enemy_evasion_gain=0.0, enemy_evasion_radius=1.0)
    config_high = EnvConfig(enemy_evasion_gain=1.0, enemy_evasion_radius=200.0)

    all_low = hs.simulate_all_scenarios(hostile_pos0, hostile_vel0, soldier_pos, defender_positions, config_low)
    all_high = hs.simulate_all_scenarios(hostile_pos0, hostile_vel0, soldier_pos, defender_positions, config_high)

    for scenario_id in hs.SCENARIO_IDS:
        assert np.array_equal(all_low[scenario_id], all_high[scenario_id]), (
            f"scenario {scenario_id} depends on enemy_evasion_gain/radius -- information leakage"
        )


def _forbidden_identifier_usages(module, forbidden_names):
    """AST-based static audit: return a list of forbidden attribute
    accesses / name usages / calls in `module`'s actual CODE (docstrings
    and comments are never part of the AST, so mentioning a forbidden name
    in prose documentation -- as this module's own docstrings legitimately
    do, to explain what is excluded -- cannot cause a false positive)."""
    import ast

    tree = ast.parse(inspect.getsource(module))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden_names:
            hits.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in forbidden_names:
            hits.append(node.id)
    return hits


def test_hostile_scenarios_module_never_references_evasion_or_ground_truth():
    """Static AST audit: the module's actual CODE (not docstrings) must
    never access the forbidden identifiers."""
    forbidden = ("enemy_evasion_gain", "enemy_evasion_radius", "_compute_enemy_evasion", "enemy_pos", "enemy_vel")
    hits = _forbidden_identifier_usages(hs, forbidden)
    assert hits == [], f"forbidden identifiers used in hostile_scenarios.py code: {hits}"


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
