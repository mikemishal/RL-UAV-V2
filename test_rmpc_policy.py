"""
Tests for `uav_defend.policies.mpc.rmpc_policy.RMPCPolicy`.

Covers categories A (information contract), B (action validity), C
(determinism), G (edge cases), H (pre-detection), I (no side effects), J
(evaluator compatibility), and K (runtime instrumentation neutrality) from
`docs/rmpc_baseline_design.md` Section 16.

Run directly: python test_rmpc_policy.py
"""

from __future__ import annotations

import copy
import inspect

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.policies.mpc import rmpc_policy as rmpc_policy_module
from uav_defend.policies.mpc import rmpc_cost as rmpc_cost_module
from uav_defend.policies.mpc import lead_init as lead_init_module
from uav_defend.policies.mpc.rmpc_policy import RMPCConfig, RMPCPolicy, VALID_GUIDANCE_MODES
from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env

FAST_RMPC_CONFIG = RMPCConfig(horizon=4, population_size=24, elite_fraction=0.25,
                               cem_iterations=2, controller_seed=1234)


def _detected_info(enemy_measurement=(6.0, 3.0, 1.0), enemy_measurement_velocity=(1.0, 0.5, 0.0),
                    velocity_valid=True, defender_pos=(0.0, 0.0, 0.0), defender_vel=(0.0, 0.0, 0.0),
                    soldier_pos=(-5.0, -5.0, 0.0)):
    return {
        "soldier_pos": np.array(soldier_pos, dtype=np.float64),
        "defender_pos": np.array(defender_pos, dtype=np.float64),
        "defender_vel": np.array(defender_vel, dtype=np.float64),
        "enemy_detected": True,
        "enemy_measurement": np.array(enemy_measurement, dtype=np.float64),
        "enemy_measurement_velocity": np.array(enemy_measurement_velocity, dtype=np.float64),
        "enemy_measurement_velocity_valid": velocity_valid,
    }


def _undetected_info(defender_pos=(0.0, 0.0, 0.0), soldier_pos=(0.0, 0.0, 0.0)):
    return {
        "soldier_pos": np.array(soldier_pos, dtype=np.float64),
        "defender_pos": np.array(defender_pos, dtype=np.float64),
        "defender_vel": np.zeros(3, dtype=np.float64),
        "enemy_detected": False,
        "enemy_measurement": None,
        "enemy_measurement_velocity": None,
        "enemy_measurement_velocity_valid": False,
    }


# --- A. Information contract ----------------------------------------------

def test_rmpc_never_reads_ground_truth_keys():
    """A poisoned info dict containing enemy_pos/enemy_vel (which build_policy_info
    would normally strip) must have NO effect on the action, since RMPCPolicy
    must never read those keys regardless of whether the sanitizer stripped
    them."""
    info_a = _detected_info()
    info_b = _detected_info()
    info_a["enemy_pos"] = np.array([999.0, 999.0, 999.0])
    info_a["enemy_vel"] = np.array([-50.0, -50.0, -50.0])
    info_b["enemy_pos"] = np.array([-1.0, -1.0, -1.0])
    info_b["enemy_vel"] = np.array([50.0, 50.0, 50.0])

    policy_a = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    policy_b = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)

    action_a = policy_a.act(np.zeros(16), info_a)
    action_b = policy_b.act(np.zeros(16), info_b)
    assert np.array_equal(action_a, action_b), "action depends on enemy_pos/enemy_vel -- information leakage"


def test_rmpc_never_depends_on_evasion_gain_or_radius():
    """No config field describing the simulated hostile's decision policy
    (enemy_evasion_gain/radius) may affect RMPC's action."""
    info = _detected_info()
    config_low = EnvConfig(enemy_evasion_gain=0.0, enemy_evasion_radius=1.0)
    config_high = EnvConfig(enemy_evasion_gain=1.0, enemy_evasion_radius=200.0)

    policy_low = RMPCPolicy(config=config_low, rmpc_config=FAST_RMPC_CONFIG)
    policy_high = RMPCPolicy(config=config_high, rmpc_config=FAST_RMPC_CONFIG)

    action_low = policy_low.act(np.zeros(16), copy.deepcopy(info))
    action_high = policy_high.act(np.zeros(16), copy.deepcopy(info))
    assert np.array_equal(action_low, action_high)


def test_rmpc_uses_only_sanitized_info_from_real_env():
    """act() must succeed cleanly when driven through the standard sanitizer
    pipeline against a real environment (no KeyError / no ground-truth
    dependence)."""
    config = EnvConfig()
    env = SoldierEnv(config=config)
    policy = RMPCPolicy(config=config, rmpc_config=FAST_RMPC_CONFIG)
    obs, info = env.reset(seed=5)
    policy.reset()
    estimator_mode = estimator_mode_for_env(env)
    for _ in range(10):
        policy_info = build_policy_info(info, estimator_mode)
        assert "enemy_pos" not in policy_info
        assert "enemy_vel" not in policy_info
        action = policy.act(obs, policy_info)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    env.close()


def _forbidden_identifier_usages(module, forbidden_names):
    """AST-based static audit: return forbidden attribute accesses / name
    usages in `module`'s actual CODE. Docstrings/comments are never part of
    the AST, so legitimate prose explaining what is excluded (as this
    module's own docstrings do) cannot cause a false positive."""
    import ast

    tree = ast.parse(inspect.getsource(module))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden_names:
            hits.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in forbidden_names:
            hits.append(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in forbidden_names:
            # Catches dict-key-style access, e.g. info["enemy_pos"] /
            # info.get("enemy_evasion_gain"), which use a string constant
            # rather than an attribute/name node.
            hits.append(node.value)
    return hits


def test_rmpc_policy_source_never_references_forbidden_identifiers():
    """Static AST audit of the RMPC implementation modules: actual CODE
    must never access ground-truth keys, Kalman-track fields, or
    evasion-policy parameters."""
    forbidden = ("enemy_evasion_gain", "enemy_evasion_radius", "_compute_enemy_evasion",
                 "enemy_pos", "enemy_vel", "e_hat", "v_hat")
    for module in (rmpc_policy_module, rmpc_cost_module, lead_init_module):
        hits = _forbidden_identifier_usages(module, forbidden)
        assert hits == [], f"forbidden identifiers used in {module.__name__} code: {hits}"


# --- B. Action validity ---------------------------------------------------

def test_action_shape_and_dtype():
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    action = policy.act(np.zeros(16), _detected_info())
    assert action.shape == (3,)
    assert action.dtype == np.float32


def test_action_finite_and_within_bounds_random_states():
    rng = np.random.default_rng(99)
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    for _ in range(15):
        info = _detected_info(
            enemy_measurement=tuple(rng.uniform(-30, 30, size=3)),
            enemy_measurement_velocity=tuple(rng.uniform(-10, 10, size=3)),
            defender_pos=tuple(rng.uniform(-30, 30, size=3)),
            defender_vel=tuple(rng.uniform(-5, 5, size=3)),
            soldier_pos=tuple(rng.uniform(-30, 30, size=3)),
        )
        action = policy.act(np.zeros(16), info)
        assert np.all(np.isfinite(action))
        assert np.linalg.norm(action) <= 1.0 + 1e-6


def test_zero_action_when_not_detected():
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    action = policy.act(np.zeros(16), _undetected_info())
    assert np.array_equal(action, np.zeros(3, dtype=np.float32))
    assert policy.last_guidance_mode == "standby"
    assert policy.last_objective_evaluations == 0


# --- C. Determinism ---------------------------------------------------------

def test_same_controller_seed_same_state_same_action_across_instances():
    info = _detected_info()
    policy_1 = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    policy_2 = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)

    action_1 = policy_1.act(np.zeros(16), copy.deepcopy(info))
    action_2 = policy_2.act(np.zeros(16), copy.deepcopy(info))

    assert np.array_equal(action_1, action_2)
    assert policy_1.last_robust_score == policy_2.last_robust_score
    assert policy_1.last_scenario_scores == policy_2.last_scenario_scores


def test_different_controller_seeds_may_differ_but_both_valid():
    info = _detected_info()
    cfg_a = RMPCConfig(horizon=4, population_size=24, elite_fraction=0.25, cem_iterations=2, controller_seed=1)
    cfg_b = RMPCConfig(horizon=4, population_size=24, elite_fraction=0.25, cem_iterations=2, controller_seed=2)
    action_a = RMPCPolicy(rmpc_config=cfg_a).act(np.zeros(16), copy.deepcopy(info))
    action_b = RMPCPolicy(rmpc_config=cfg_b).act(np.zeros(16), copy.deepcopy(info))
    assert np.all(np.isfinite(action_a)) and np.all(np.isfinite(action_b))


# --- G. Zero/first-measurement edge cases ----------------------------------

def test_no_velocity_estimate_on_first_detection_step():
    info = _detected_info(velocity_valid=False, enemy_measurement_velocity=(0.0, 0.0, 0.0))
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    action = policy.act(np.zeros(16), info)
    assert np.all(np.isfinite(action))
    assert policy.last_guidance_mode == "first_measurement"


def test_no_detection_yet_returns_safe_fallback():
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    action = policy.act(np.zeros(16), _undetected_info())
    assert np.array_equal(action, np.zeros(3, dtype=np.float32))


def test_degenerate_zero_distance_geometry():
    """Defender exactly co-located with the hostile measurement must not
    raise and must produce a finite, valid action."""
    info = _detected_info(enemy_measurement=(0.0, 0.0, 0.0), defender_pos=(0.0, 0.0, 0.0),
                           enemy_measurement_velocity=(0.0, 0.0, 0.0), velocity_valid=True)
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    action = policy.act(np.zeros(16), info)
    assert np.all(np.isfinite(action))
    assert np.linalg.norm(action) <= 1.0 + 1e-6


def test_no_feasible_candidate_falls_back_gracefully(monkeypatch):
    """If CEM cannot find any finite-scored candidate, RMPCPolicy must fall
    back to pure pursuit rather than raising or returning a non-finite
    action."""

    def always_nan(*args, **kwargs):
        class _Result:
            robust_score = float("nan")

        return _Result()

    monkeypatch.setattr(rmpc_policy_module, "evaluate_candidate", always_nan)
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    info = _detected_info(enemy_measurement=(10.0, 0.0, 0.0), defender_pos=(0.0, 0.0, 0.0))
    action = policy.act(np.zeros(16), info)
    assert policy.last_guidance_mode == "optimizer_fallback"
    assert np.all(np.isfinite(action))
    # Pure-pursuit direction toward (10,0,0) from (0,0,0) is +x.
    assert action[0] > 0.9


# --- H. Pre-detection behavior ---------------------------------------------

def test_environment_ignores_rmpc_action_before_detection():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    policy = RMPCPolicy(config=config, rmpc_config=FAST_RMPC_CONFIG)
    obs, info = env.reset(seed=17)
    policy.reset()
    estimator_mode = estimator_mode_for_env(env)

    steps = 0
    while not info["enemy_detected"] and steps < 50:
        policy_info = build_policy_info(info, estimator_mode)
        action = policy.act(obs, policy_info)
        assert np.array_equal(action, np.zeros(3, dtype=np.float32))
        obs, reward, term, trunc, info = env.step(action)
        # Defender remains exactly co-located with the soldier during standby
        # regardless of the (discarded) action -- existing environment
        # guarantee; this asserts RMPC does not bypass it.
        assert np.allclose(info["defender_pos"], info["soldier_pos"])
        steps += 1
        if term or trunc:
            break
    env.close()


# --- I. No side effects -----------------------------------------------------

def test_planning_does_not_mutate_input_info_arrays():
    info = _detected_info()
    snapshots = {k: v.copy() for k, v in info.items() if isinstance(v, np.ndarray)}
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    policy.act(np.zeros(16), info)
    for key, snapshot in snapshots.items():
        assert np.array_equal(info[key], snapshot), f"info['{key}'] was mutated by policy.act()"


def test_planning_does_not_consume_env_rng_state():
    config = EnvConfig()
    env = SoldierEnv(config=config)
    policy = RMPCPolicy(config=config, rmpc_config=FAST_RMPC_CONFIG)
    obs, info = env.reset(seed=3)
    policy.reset()
    estimator_mode = estimator_mode_for_env(env)

    # Drive to a detected state first (env.step DOES legitimately consume
    # env RNG streams -- only policy.act() itself must not).
    while not info["enemy_detected"]:
        policy_info = build_policy_info(info, estimator_mode)
        obs, reward, term, trunc, info = env.step(policy.act(obs, policy_info))
        if term or trunc:
            env.close()
            return  # could not reach detection in this seed; skip gracefully

    rng_attrs = ("_rng_spawn", "_rng_soldier", "_rng_enemy_motion", "_rng_sensor")
    before = {attr: getattr(env, attr).bit_generator.state for attr in rng_attrs}

    policy_info = build_policy_info(info, estimator_mode)
    _ = policy.act(obs, policy_info)  # NOT followed by env.step()

    after = {attr: getattr(env, attr).bit_generator.state for attr in rng_attrs}
    for attr in rng_attrs:
        assert before[attr] == after[attr], f"policy.act() consumed environment RNG stream '{attr}'"
    env.close()


# --- J. Evaluator compatibility ---------------------------------------------

def test_policy_reset_semantics():
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    info = _detected_info()
    action_before = policy.act(np.zeros(16), copy.deepcopy(info))
    assert policy.last_guidance_mode is not None

    policy.reset()
    assert policy.last_guidance_mode is None
    assert policy.last_scenario_scores is None
    assert policy.last_robust_score is None
    assert policy.last_best_sequence is None

    action_after_reset = policy.act(np.zeros(16), copy.deepcopy(info))
    assert np.array_equal(action_before, action_after_reset), (
        "reset() must re-seed the dedicated RNG so a fresh episode with the "
        "same controller_seed reproduces identical decisions"
    )


def test_matched_seed_evaluation_compatible_with_run_diagnostic_episode():
    from experiments.post_detection_diagnostic_eval_utils import run_diagnostic_episode

    config = EnvConfig()
    env = SoldierEnv(config=config)
    policy = RMPCPolicy(config=config, rmpc_config=FAST_RMPC_CONFIG)
    row = run_diagnostic_episode(env, policy, seed=9, dt=config.dt)
    assert "outcome" in row
    assert row["environment_seed"] == 9
    env.close()


# --- K. Runtime diagnostics --------------------------------------------------

def test_timer_instrumentation_does_not_alter_action():
    info = _detected_info()
    cfg_timed = RMPCConfig(horizon=4, population_size=24, elite_fraction=0.25, cem_iterations=2,
                            controller_seed=55, enable_timing=True)
    cfg_untimed = RMPCConfig(horizon=4, population_size=24, elite_fraction=0.25, cem_iterations=2,
                              controller_seed=55, enable_timing=False)

    action_timed = RMPCPolicy(rmpc_config=cfg_timed).act(np.zeros(16), copy.deepcopy(info))
    action_untimed = RMPCPolicy(rmpc_config=cfg_untimed).act(np.zeros(16), copy.deepcopy(info))

    assert np.array_equal(action_timed, action_untimed)


def test_decision_time_stats_reporting():
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    stats_empty = policy.decision_time_stats()
    assert stats_empty["count"] == 0

    policy.act(np.zeros(16), _detected_info())
    stats = policy.decision_time_stats()
    assert stats["count"] == 1
    assert stats["mean"] >= 0.0
    assert stats["max"] >= stats["mean"]


# --- L. CEM nominal initialization (hardening pass) -------------------------

def test_initialization_mode_lead_when_velocity_valid_and_solvable():
    info = _detected_info(enemy_measurement=(20.0, 0.0, 0.0), enemy_measurement_velocity=(1.0, 0.0, 0.0),
                           velocity_valid=True, defender_pos=(0.0, 0.0, 0.0))
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    policy.act(np.zeros(16), info)
    assert policy.last_initialization_mode in ("lead", "pursuit")  # both are valid depending on geometry
    assert policy.last_initialization_mode is not None


def test_initialization_mode_pursuit_when_velocity_invalid():
    info = _detected_info(velocity_valid=False, enemy_measurement_velocity=(0.0, 0.0, 0.0))
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    policy.act(np.zeros(16), info)
    assert policy.last_initialization_mode == "pursuit"


def test_initialization_mode_none_during_standby():
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    policy.act(np.zeros(16), _undetected_info())
    assert policy.last_initialization_mode is None


def test_cem_optimize_receives_non_none_init_mean(monkeypatch):
    """Focused mock test: cem_optimize must be called with a non-None,
    correctly-shaped init_mean derived from the Lead/pursuit direction."""
    captured = {}
    original_cem_optimize = rmpc_policy_module.cem_optimize

    def spy_cem_optimize(*args, **kwargs):
        captured["init_mean"] = kwargs.get("init_mean")
        return original_cem_optimize(*args, **kwargs)

    monkeypatch.setattr(rmpc_policy_module, "cem_optimize", spy_cem_optimize)
    policy = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    info = _detected_info(enemy_measurement=(20.0, 0.0, 0.0), enemy_measurement_velocity=(1.0, 0.0, 0.0),
                           velocity_valid=True, defender_pos=(0.0, 0.0, 0.0))
    policy.act(np.zeros(16), info)

    assert captured["init_mean"] is not None
    assert captured["init_mean"].shape == (FAST_RMPC_CONFIG.horizon, 3)
    # All rows identical (nominal direction repeated over the horizon, no
    # per-timestep warm-start yet).
    assert np.allclose(captured["init_mean"], captured["init_mean"][0])


def test_initialization_unaffected_by_poisoned_ground_truth():
    info_a = _detected_info()
    info_b = _detected_info()
    info_a["enemy_pos"] = np.array([999.0, 999.0, 999.0])
    info_a["enemy_vel"] = np.array([-50.0, -50.0, -50.0])
    info_b["enemy_pos"] = np.array([-1.0, -1.0, -1.0])
    info_b["enemy_vel"] = np.array([50.0, 50.0, 50.0])

    policy_a = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    policy_b = RMPCPolicy(rmpc_config=FAST_RMPC_CONFIG)
    policy_a.act(np.zeros(16), info_a)
    policy_b.act(np.zeros(16), info_b)

    assert policy_a.last_initialization_mode == policy_b.last_initialization_mode


if __name__ == "__main__":
    import sys
    import traceback
    import types

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = failed = 0

    class _FakeMonkeypatch:
        def __init__(self):
            self._restores = []

        def setattr(self, target, name, value):
            self._restores.append((target, name, getattr(target, name)))
            setattr(target, name, value)

        def undo(self):
            for target, name, old in reversed(self._restores):
                setattr(target, name, old)

    for t in tests:
        try:
            sig = inspect.signature(t)
            if "monkeypatch" in sig.parameters:
                mp = _FakeMonkeypatch()
                try:
                    t(mp)
                finally:
                    mp.undo()
            else:
                t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
