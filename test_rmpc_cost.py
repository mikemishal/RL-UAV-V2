"""
Tests for `uav_defend.policies.mpc.rmpc_cost` -- in particular the
corrected nonterminal (no-terminal-event-within-horizon) objective, which
must equal exactly `reward_progress_scale * (dist_de_0 - dist_de_H)` with
NO additional `-dist_de_H` term (that term was an undeclared extra weight,
removed in this hardening pass).

Run directly: python test_rmpc_cost.py
"""

from __future__ import annotations

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.mpc.rmpc_cost import score_trajectory

EPS = 1e-8


def test_nonterminal_score_matches_progress_only_formula():
    config = EnvConfig()
    horizon = 4
    # Straight-line trajectories, far from all safety/intercept radii, so no
    # terminal event fires at any step.
    defender_positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                                    [6.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    hostile_positions = np.array([[50.0, 0.0, 0.0], [48.0, 0.0, 0.0], [46.0, 0.0, 0.0],
                                   [44.0, 0.0, 0.0], [42.0, 0.0, 0.0]])
    soldier_pos_fixed = np.array([200.0, 200.0, 0.0])

    outcome = score_trajectory(defender_positions, hostile_positions, soldier_pos_fixed, config)
    assert outcome.terminal_step is None
    assert outcome.terminal_type is None

    dist_de_0 = float(np.linalg.norm(defender_positions[0] - hostile_positions[0]))
    dist_de_h = float(np.linalg.norm(defender_positions[horizon] - hostile_positions[horizon]))
    expected = config.reward_progress_scale * (dist_de_0 - dist_de_h)

    assert np.isclose(outcome.score, expected, atol=1e-9)
    # Explicitly confirm the OLD (removed) `-dist_de_h` term is NOT present:
    old_formula_value = expected - dist_de_h
    assert not np.isclose(outcome.score, old_formula_value, atol=1e-6) or np.isclose(dist_de_h, 0.0)


def test_nonterminal_score_zero_progress_is_zero():
    """If the defender-hostile distance is unchanged over the horizon
    (zero progress), the nonterminal score must be exactly zero (no
    residual `-dist_de_h` bias remains)."""
    config = EnvConfig()
    defender_positions = np.tile(np.array([0.0, 0.0, 0.0]), (5, 1))
    hostile_positions = np.tile(np.array([50.0, 0.0, 0.0]), (5, 1))
    soldier_pos_fixed = np.array([200.0, 200.0, 0.0])

    outcome = score_trajectory(defender_positions, hostile_positions, soldier_pos_fixed, config)
    assert outcome.terminal_step is None
    assert outcome.score == 0.0


def test_terminal_events_still_use_unchanged_reward_constants():
    config = EnvConfig()
    # Soldier caught: hostile reaches within threat_radius of soldier.
    defender_positions = np.tile(np.array([100.0, 100.0, 100.0]), (3, 1))
    hostile_positions = np.array([[10.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    soldier_pos_fixed = np.array([0.0, 0.0, 0.0])
    outcome = score_trajectory(defender_positions, hostile_positions, soldier_pos_fixed, config)
    assert outcome.terminal_type == "soldier_caught"
    assert outcome.score == config.reward_soldier_caught

    # Safe intercept: defender reaches hostile, far from soldier.
    defender_positions = np.array([[50.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    hostile_positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    soldier_pos_fixed = np.array([-500.0, -500.0, 0.0])
    outcome = score_trajectory(defender_positions, hostile_positions, soldier_pos_fixed, config)
    assert outcome.terminal_type == "intercepted"
    assert outcome.score == config.reward_intercept

    # Unsafe intercept: defender reaches hostile at a point that is beyond
    # threat_radius but within unsafe_intercept_radius of the soldier
    # (threat_radius=2.0 < 3.0 <= unsafe_intercept_radius=3.5 by default).
    defender_positions = np.array([[50.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    hostile_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    soldier_pos_fixed = np.array([1.0, 3.0, 0.0])
    outcome = score_trajectory(defender_positions, hostile_positions, soldier_pos_fixed, config)
    assert outcome.terminal_type == "unsafe_intercept"
    assert outcome.score == config.reward_unsafe_intercept


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
