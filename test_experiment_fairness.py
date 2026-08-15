"""Experiment-fairness regression tests -- publication-integrity guardrails.

Script-style test consistent with the project's other test files
(test_3d_dynamics.py, test_proportional_navigation.py, etc). No external
test framework is used; each test function performs assertions and the
runner reports PASS/FAIL per test with a final summary.

This suite exists to STRUCTURALLY guarantee that the Direct/measurement and
Kalman experiment tracks are scientifically comparable, and to catch any
future regression that would silently reintroduce an unfair advantage for
one track over the other, or a ground-truth leakage path into a policy.
Treat failures here as publication-blocking.

Covers:
    A. Independent RNG streams: sensor-RNG consumption (which depends on
       detection timing, itself controller/config-dependent) must NEVER
       perturb the exogenous randomness (soldier walk, hostile weave/noise,
       spawn) shared by every method.
    B. Direct/Kalman TRUE-trajectory equivalence: for the same seed and the
       same fixed action sequence, the true soldier/defender/enemy
       trajectory must be bit-identical regardless of use_kalman_tracking.
    C. Common sensor measurements: the same noisy position measurement is
       drawn for both tracks (single shared sensor realization).
    D. Identical ongoing reward for the same true trajectory, regardless of
       estimator track (no Kalman-only reward term).
    E. Observation semantic parity: obs[10:13]/obs[13:16] denormalize back
       to the SAME legitimate estimator fields (enemy_measurement/e_hat,
       enemy_measurement_velocity/v_hat) for both tracks.
    F. No ground-truth leakage for EVERY scripted controller (Greedy,
       KalmanGreedy, PN, PN-Kalman, Lead, Lead-Kalman): action must be
       unchanged when info["enemy_pos"]/info["enemy_vel"] are extreme, and
       controllers must work with those keys entirely ABSENT (sanitized).
    G. PPO policy wrappers (Direct and Kalman) ignore `info` entirely.
    H. Method-to-estimator mapping (experiments/method_specs.py) is
       consistent with the policy registry.
    I. Matched-seed reproducibility holds for both tracks.
    J. Unified result schema: experiments.eval_utils.run_episode() returns
       the same metric KEY SET regardless of estimator track.

Run directly:
    python test_experiment_fairness.py
"""

from __future__ import annotations

import numpy as np

from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies import (
    GreedyInterceptPolicy,
    KalmanGreedyInterceptPolicy,
    LeadInterceptPolicy,
    ProportionalNavigationPolicy,
)
from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env
from uav_defend.policies.registry import get_policy, get_policy_name

from experiments.method_specs import METHOD_SPECS, get_method_spec
from experiments.eval_utils import run_episode


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _matched_rollout(config_a: EnvConfig, config_b: EnvConfig, seed: int, n_steps: int, action_fn):
    """
    Run two freshly-constructed environments with the SAME seed and a
    SHARED, purely-exogenous action sequence (never a function of any
    Kalman/Direct-specific field), returning parallel per-step traces.
    """
    env_a = SoldierEnv(config=config_a)
    env_b = SoldierEnv(config=config_b)
    obs_a, info_a = env_a.reset(seed=seed)
    obs_b, info_b = env_b.reset(seed=seed)

    trace_a = []
    trace_b = []
    for i in range(n_steps):
        action = action_fn(i)
        obs_a, r_a, term_a, trunc_a, info_a = env_a.step(action)
        obs_b, r_b, term_b, trunc_b, info_b = env_b.step(action)
        trace_a.append((obs_a.copy(), r_a, term_a, dict(info_a)))
        trace_b.append((obs_b.copy(), r_b, term_b, dict(info_b)))
        if term_a or term_b:
            break
    return trace_a, trace_b


def _fixed_action(i: int) -> np.ndarray:
    """A constant, purely exogenous action (never derived from obs/info)."""
    return np.array([0.3, 0.4, 0.15], dtype=np.float32)


# ---------------------------------------------------------------------------
# A. RNG stream independence
# ---------------------------------------------------------------------------
def test_rng_streams_independent_of_detection_radius():
    """
    Changing detection_radius changes WHEN (and whether) the sensor RNG
    stream is consumed. The TRUE soldier/defender/enemy trajectory must be
    completely unaffected, proving the sensor stream is independent from
    the spawn/soldier/enemy-motion streams.
    """
    config_a = EnvConfig(detection_radius=1.0)
    config_b = EnvConfig(detection_radius=49.0)
    trace_a, trace_b = _matched_rollout(config_a, config_b, seed=123, n_steps=80, action_fn=_fixed_action)

    assert len(trace_a) == len(trace_b) and len(trace_a) > 20
    for (_, _, _, info_a), (_, _, _, info_b) in zip(trace_a, trace_b):
        assert np.allclose(info_a["soldier_pos"], info_b["soldier_pos"], atol=1e-6)
        assert np.allclose(info_a["defender_pos"], info_b["defender_pos"], atol=1e-6)
        assert np.allclose(info_a["enemy_pos"], info_b["enemy_pos"], atol=1e-6)

    # Sanity: the two configs must actually have produced DIFFERENT
    # detection timing, otherwise this test isn't exercising anything.
    detected_a = [info["enemy_detected"] for _, _, _, info in trace_a]
    detected_b = [info["enemy_detected"] for _, _, _, info in trace_b]
    assert detected_a != detected_b, "test scenario failed to vary detection timing"


# ---------------------------------------------------------------------------
# B, C, D. Direct/Kalman equivalence: true trajectory, sensor measurement,
#          and reward
# ---------------------------------------------------------------------------
def test_direct_kalman_equivalence():
    config_direct = EnvConfig(use_kalman_tracking=False, detection_radius=20.0)
    config_kalman = EnvConfig(use_kalman_tracking=True, detection_radius=20.0)
    trace_direct, trace_kalman = _matched_rollout(
        config_direct, config_kalman, seed=321, n_steps=100, action_fn=_fixed_action
    )

    assert len(trace_direct) == len(trace_kalman) and len(trace_direct) > 20
    saw_detection = False
    for (_, r_d, _, info_d), (_, r_k, _, info_k) in zip(trace_direct, trace_kalman):
        # B. True trajectory equivalence
        assert np.allclose(info_d["soldier_pos"], info_k["soldier_pos"], atol=1e-6)
        assert np.allclose(info_d["defender_pos"], info_k["defender_pos"], atol=1e-6)
        assert np.allclose(info_d["enemy_pos"], info_k["enemy_pos"], atol=1e-6)

        # D. Identical ongoing reward (no Kalman-only reward term)
        assert abs(r_d - r_k) < 1e-5, (r_d, r_k)

        # C. Common sensor measurement (same detection timing, same draw)
        assert info_d["enemy_detected"] == info_k["enemy_detected"]
        if info_d["enemy_detected"]:
            saw_detection = True
            assert np.allclose(info_d["enemy_measurement"], info_k["enemy_measurement"], atol=1e-6)

    assert saw_detection, "scenario never triggered detection; test is not exercising the sensor path"


# ---------------------------------------------------------------------------
# E. Observation semantic parity
# ---------------------------------------------------------------------------
def test_observation_semantic_parity():
    """
    Both Direct and Kalman observations share IDENTICAL layout semantics:
    obs[10:13] = hostile position estimate (denormalized by [L, L, H]);
    obs[13:16] = hostile velocity estimate (denormalized by v_e). Verified
    by recovering info["enemy_measurement"]/["e_hat"] and
    info["enemy_measurement_velocity"]/["v_hat"] from obs.

    Searches across several seeds/actions and a longer horizon so the
    hostile UAV has time to move away from its edge-spawn position (where
    the horizontal normalized coordinate is near +/-1 and therefore
    frequently saturated by the observation's [-1, 1] clip).
    """
    for use_kalman in (False, True):
        config = EnvConfig(use_kalman_tracking=use_kalman, detection_radius=25.0)
        L, H, v_e = config.L, config.max_altitude, config.v_e

        checked_pos = 0
        checked_vel = 0
        for seed in range(20):
            env = SoldierEnv(config=config)
            obs, info = env.reset(seed=seed)
            for _ in range(300):
                action = np.array([0.25, -0.2, 0.1], dtype=np.float32)
                obs, reward, term, trunc, info = env.step(action)
                if info["enemy_detected"]:
                    pos_field = info["e_hat"] if use_kalman else info["enemy_measurement"]
                    vel_field = info["v_hat"] if use_kalman else info["enemy_measurement_velocity"]

                    pos_norm = obs[10:13]
                    vel_norm = obs[13:16]
                    if np.all(np.abs(pos_norm) < 0.999):
                        pos_from_obs = pos_norm * np.array([L, L, H], dtype=np.float32)
                        assert np.allclose(pos_from_obs, pos_field, atol=5e-3), (use_kalman, pos_from_obs, pos_field)
                        checked_pos += 1
                    if np.all(np.abs(vel_norm) < 0.999):
                        vel_from_obs = vel_norm * v_e
                        assert np.allclose(vel_from_obs, vel_field, atol=5e-3), (use_kalman, vel_from_obs, vel_field)
                        checked_vel += 1
                if term:
                    break
            if checked_pos > 0 and checked_vel > 0:
                break

        assert checked_pos > 0, f"no unclipped position sample for use_kalman_tracking={use_kalman}"
        assert checked_vel > 0, f"no unclipped velocity sample for use_kalman_tracking={use_kalman}"


# ---------------------------------------------------------------------------
# F. No ground-truth leakage, for EVERY scripted controller
# ---------------------------------------------------------------------------
def _make_scripted_info(
    defender_pos, defender_vel, soldier_pos, *,
    measurement=None, meas_vel=None, meas_vel_valid=False,
    e_hat=None, v_hat=None, enemy_pos=None, enemy_vel=None,
):
    info = {
        "defender_pos": np.asarray(defender_pos, dtype=np.float64),
        "defender_vel": np.asarray(defender_vel, dtype=np.float64),
        "soldier_pos": np.asarray(soldier_pos, dtype=np.float64),
        "enemy_detected": True,
    }
    if measurement is not None:
        info["enemy_measurement"] = np.asarray(measurement, dtype=np.float64)
    info["enemy_measurement_velocity"] = (
        np.asarray(meas_vel, dtype=np.float64) if meas_vel is not None else np.zeros(3)
    )
    info["enemy_measurement_velocity_valid"] = meas_vel_valid
    if e_hat is not None:
        info["e_hat"] = np.asarray(e_hat, dtype=np.float64)
    if v_hat is not None:
        info["v_hat"] = np.asarray(v_hat, dtype=np.float64)
    if enemy_pos is not None:
        info["enemy_pos"] = np.asarray(enemy_pos, dtype=np.float64)
    if enemy_vel is not None:
        info["enemy_vel"] = np.asarray(enemy_vel, dtype=np.float64)
    return info


def test_no_ground_truth_leakage_all_controllers():
    defender_pos = np.array([0.0, 0.0, 5.0])
    defender_vel = np.array([5.0, 0.0, 0.0])
    soldier_pos = np.array([-20.0, 0.0, 0.0])

    scenarios = [
        ("greedy", GreedyInterceptPolicy(), "measurement"),
        ("kalman_greedy", KalmanGreedyInterceptPolicy(), "kalman"),
        ("pn", ProportionalNavigationPolicy(state_source="measurement"), "measurement"),
        ("pn_kalman", ProportionalNavigationPolicy(state_source="kalman"), "kalman"),
        ("lead", LeadInterceptPolicy(state_source="measurement"), "measurement"),
        ("lead_kalman", LeadInterceptPolicy(state_source="kalman"), "kalman"),
    ]

    for name, policy, mode in scenarios:
        if mode == "measurement":
            base_info = _make_scripted_info(
                defender_pos, defender_vel, soldier_pos,
                measurement=[10.0, 0.0, 0.0], meas_vel=[-1.0, 0.5, 0.0], meas_vel_valid=True,
                enemy_pos=[10.0, 0.0, 0.0], enemy_vel=[-1.0, 0.5, 0.0],
            )
        else:
            base_info = _make_scripted_info(
                defender_pos, defender_vel, soldier_pos,
                e_hat=[10.0, 0.0, 0.0], v_hat=[-1.0, 0.5, 0.0],
                enemy_pos=[10.0, 0.0, 0.0], enemy_vel=[-1.0, 0.5, 0.0],
            )

        policy.reset()
        action_truth = policy.act(None, base_info)

        # (1) Action must be unchanged when ground truth is set to extreme,
        # blatantly-different values.
        leaked_info = dict(base_info)
        leaked_info["enemy_pos"] = np.array([9999.0, -8888.0, 7777.0])
        leaked_info["enemy_vel"] = np.array([-500.0, 300.0, 200.0])
        policy.reset()
        action_leaked = policy.act(None, leaked_info)
        assert np.allclose(action_truth, action_leaked, atol=1e-9), (name, "leaked", action_truth, action_leaked)

        # (2) Action must be unchanged (and must not crash) when ground
        # truth keys are entirely ABSENT (the sanitized info path used by
        # experiments/eval_utils.py).
        sanitized_info = build_policy_info(base_info, mode)
        assert "enemy_pos" not in sanitized_info
        assert "enemy_vel" not in sanitized_info
        policy.reset()
        action_sanitized = policy.act(None, sanitized_info)
        assert np.allclose(action_truth, action_sanitized, atol=1e-9), (
            name, "sanitized", action_truth, action_sanitized
        )


# ---------------------------------------------------------------------------
# G. PPO policy wrappers ignore `info` entirely
# ---------------------------------------------------------------------------
def test_ppo_wrappers_ignore_info():
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("  (stable-baselines3 not installed; skipping)")
        return

    from uav_defend.policies.rl.ppo_policy_wrapper import PPOPolicyWrapper
    from uav_defend.policies.rl_kalman.ppo_kalman_policy_wrapper import PPOKalmanPolicyWrapper

    env = SoldierEnv(config=EnvConfig(use_kalman_tracking=False))
    obs, info = env.reset(seed=1)
    # A tiny, untrained model is sufficient: we are only checking that
    # act()'s OUTPUT does not depend on the `info` argument at all, which is
    # true regardless of training state (see PPOPolicyWrapper.act()).
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)

    for wrapper_cls in (PPOPolicyWrapper, PPOKalmanPolicyWrapper):
        policy = wrapper_cls(model, deterministic=True)
        info_a = dict(info)
        info_b = {"enemy_pos": np.array([1.0, 2.0, 3.0]), "totally_unrelated_key": True}
        action_a = policy.act(obs, info_a)
        action_b = policy.act(obs, info_b)
        assert np.allclose(action_a, action_b), (wrapper_cls.__name__, action_a, action_b)


# ---------------------------------------------------------------------------
# H. Method-to-estimator mapping matches the registry
# ---------------------------------------------------------------------------
def test_method_specs_match_registry():
    for spec in METHOD_SPECS:
        if spec.requires_model:
            # PPO / PPO-Kalman require a model_path; verify the registry
            # rejects a call without one (structural check only).
            try:
                get_policy(spec.policy_name)
                raised = False
            except ValueError:
                raised = True
            assert raised, f"'{spec.policy_name}' should require model_path"
            continue

        policy = get_policy(spec.policy_name)
        canonical = get_policy_name(policy)
        assert canonical == spec.key, (spec.key, spec.policy_name, canonical)
        assert spec.estimator in ("measurement", "kalman", "none"), spec


# ---------------------------------------------------------------------------
# I. Matched-seed reproducibility (both tracks)
# ---------------------------------------------------------------------------
def test_matched_seed_reproducibility():
    for use_kalman in (False, True):
        config = EnvConfig(use_kalman_tracking=use_kalman)
        env1 = SoldierEnv(config=config)
        env2 = SoldierEnv(config=config)
        obs1, info1 = env1.reset(seed=999)
        obs2, info2 = env2.reset(seed=999)
        assert np.allclose(info1["soldier_pos"], info2["soldier_pos"])
        assert np.allclose(info1["defender_pos"], info2["defender_pos"])
        assert np.allclose(info1["enemy_pos"], info2["enemy_pos"])
        assert np.allclose(obs1, obs2)


# ---------------------------------------------------------------------------
# J. Unified result schema across estimator tracks
# ---------------------------------------------------------------------------
def test_unified_result_schema():
    env_direct = SoldierEnv(config=EnvConfig(use_kalman_tracking=False))
    env_kalman = SoldierEnv(config=EnvConfig(use_kalman_tracking=True))
    metrics_direct = run_episode(env_direct, GreedyInterceptPolicy(), seed=42)
    metrics_kalman = run_episode(env_kalman, KalmanGreedyInterceptPolicy(), seed=42)
    assert set(metrics_direct.keys()) == set(metrics_kalman.keys()), (
        set(metrics_direct.keys()) ^ set(metrics_kalman.keys())
    )


# ---------------------------------------------------------------------------
# Bonus: estimator_mode_for_env matches config
# ---------------------------------------------------------------------------
def test_estimator_mode_for_env():
    env_direct = SoldierEnv(config=EnvConfig(use_kalman_tracking=False))
    env_kalman = SoldierEnv(config=EnvConfig(use_kalman_tracking=True))
    assert estimator_mode_for_env(env_direct) == "measurement"
    assert estimator_mode_for_env(env_kalman) == "kalman"


def main() -> int:
    tests = [
        ("A. RNG streams independent of detection_radius", test_rng_streams_independent_of_detection_radius),
        ("B/C/D. Direct/Kalman true-trajectory, sensor, reward equivalence", test_direct_kalman_equivalence),
        ("E. Observation semantic parity", test_observation_semantic_parity),
        ("F. No ground-truth leakage (all scripted controllers)", test_no_ground_truth_leakage_all_controllers),
        ("G. PPO wrappers ignore info", test_ppo_wrappers_ignore_info),
        ("H. Method specs match registry", test_method_specs_match_registry),
        ("I. Matched-seed reproducibility", test_matched_seed_reproducibility),
        ("J. Unified result schema", test_unified_result_schema),
        ("K. estimator_mode_for_env helper", test_estimator_mode_for_env),
    ]

    print("=== Experiment-Fairness Regression Tests ===\n")
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
