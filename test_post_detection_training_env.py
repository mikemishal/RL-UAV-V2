"""Post-detection PPO training wrapper correctness tests -- locked guardrails.

Script-style test consistent with the project's other test files (no pytest).

Covers (per the post-detection-ppo-training task spec):
    A. reset returns detected_flag=1 (obs[9]).
    B. reset returns info["enemy_detected"]=True.
    C. reset state has defender_pos == soldier_pos.
    D. reset state has defender_vel == zero.
    E. reset corresponds to info["just_detected"]=True.
    F. no PPO-controlled transition occurs before returned reset state.
    G. first wrapper.step(action) executes controller action.
    H. post-detection physical limits enforced.
    I. standby rewards are excluded from wrapper episode reward.
    J. terminal outcome equals base canonical environment outcome when
       continued with identical post-detection actions.
    K. same root seed produces identical episode-seed sequence.
    L. different root seeds produce different episode-seed sequence.
    M. Direct/Kalman paired root produces same environment episode seeds.
    N. Direct/Kalman same environment seed -> same detection state pre-estimator.
    O. observation shape 16.
    P. action shape 3.
    Q. Direct no truth leakage.
    R. Kalman no truth leakage.
    S. first Direct hostile velocity estimate zero at first measurement.
    T. Kalman initializes correctly at first measurement.
    U. PPO timesteps do not count standby transitions.
    V. smoke training compatible with SB3 Monitor/VecEnv.

Run directly:
    python test_post_detection_training_env.py
"""

from __future__ import annotations

import numpy as np

from uav_defend.config import EnvConfig
from uav_defend.envs import SoldierEnv
from experiments.final_eval_utils import _check_physical_limits
from experiments.post_detection_training_env import PostDetectionTrainingEnv
from experiments.post_detection_training_protocol import (
    build_episode_seed_sequence,
    build_env_config,
    TRAINING_SEEDS,
    VALIDATION_SEED_OFFSET,
    DIAGNOSTIC_SEED_OFFSET,
    FINAL_TEST_SEED_START,
    FINAL_TEST_SEED_END,
    FUTURE_ROBUSTNESS_SEED_START,
)

_DEV_ROOT = 5000  # non-publication development root: < 40000, not in {45,46,47}, not >= 46000


# ---------------------------------------------------------------------------
# A/B/C/D/E/F. Reset semantics
# ---------------------------------------------------------------------------
def test_reset_detected_flag_and_state():
    config = build_env_config("direct")
    env = PostDetectionTrainingEnv(config=config, root_seed=_DEV_ROOT)
    obs, info = env.reset()
    assert obs[9] == 1.0, "obs[9] (detected_flag) must be 1.0 at reset"
    assert info["enemy_detected"] is True
    assert np.allclose(info["defender_pos"], info["soldier_pos"])
    assert np.allclose(info["defender_vel"], 0.0)
    assert info["just_detected"] is True
    assert info["controller_action_executed"] is False


# ---------------------------------------------------------------------------
# G. First step executes the action
# ---------------------------------------------------------------------------
def test_first_step_executes_action():
    config = build_env_config("direct")
    env = PostDetectionTrainingEnv(config=config, root_seed=_DEV_ROOT)
    obs, info = env.reset()
    obs2, reward, term, trunc, info2 = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert info2["controller_action_executed"] is True
    assert info2["defender_accel"] > 0.0


# ---------------------------------------------------------------------------
# H. Physical limits enforced post-detection
# ---------------------------------------------------------------------------
def test_physical_limits_post_detection():
    config = build_env_config("direct")
    env = PostDetectionTrainingEnv(config=config, root_seed=_DEV_ROOT + 1)
    obs, info = env.reset()
    _check_physical_limits(info, config)
    rng = np.random.default_rng(0)
    for _ in range(200):
        action = rng.uniform(-1, 1, size=3).astype(np.float32)
        obs, reward, term, trunc, info = env.step(action)
        _check_physical_limits(info, config)
        if term or trunc:
            break


# ---------------------------------------------------------------------------
# I. Standby rewards excluded
# ---------------------------------------------------------------------------
def test_standby_rewards_excluded():
    root = _DEV_ROOT + 2
    config = build_env_config("direct")
    env = PostDetectionTrainingEnv(config=config, root_seed=root)
    obs, info = env.reset()
    episode_seed = env.last_episode_seed

    action_fn = lambda s: np.array([0.5, -0.3, 0.1], dtype=np.float32)
    wrapper_total_reward = 0.0
    actions_taken = []
    step = 0
    done = False
    while not done:
        action = action_fn(step)
        actions_taken.append(action.copy())
        obs, reward, term, trunc, info = env.step(action)
        wrapper_total_reward += reward
        step += 1
        done = term or trunc

    # Reconstruct the full canonical mission with the SAME episode seed and
    # SAME post-detection action sequence; sum only the post-detection reward.
    ref_env = SoldierEnv(config=config)
    obs, info = ref_env.reset(seed=episode_seed)
    zero_action = np.zeros(3, dtype=np.float32)
    standby_reward_sum = 0.0
    while not info["just_detected"]:
        obs, reward, term, trunc, info = ref_env.step(zero_action)
        standby_reward_sum += reward
    post_detection_reward_sum = 0.0
    for a in actions_taken:
        obs, reward, term, trunc, info = ref_env.step(a)
        post_detection_reward_sum += reward
        if term or trunc:
            break

    assert abs(wrapper_total_reward - post_detection_reward_sum) < 1e-4
    # Sanity: standby accrued SOME (nonzero) reward, confirming it really was
    # excluded from the wrapper's episode reward rather than trivially zero.
    assert abs(standby_reward_sum) > 0.0


# ---------------------------------------------------------------------------
# J. Wrapper matches full canonical continuation with identical actions
# ---------------------------------------------------------------------------
def test_wrapper_matches_canonical_continuation():
    root = _DEV_ROOT + 3
    config = build_env_config("direct")
    env = PostDetectionTrainingEnv(config=config, root_seed=root)
    obs, info = env.reset()
    episode_seed = env.last_episode_seed

    action_fn = lambda s: np.array([1.0, 0.0, 0.2], dtype=np.float32)
    step = 0
    done = False
    wrapper_outcome = None
    while not done:
        obs, reward, term, trunc, info = env.step(action_fn(step))
        step += 1
        done = term or trunc
        if done:
            wrapper_outcome = info["outcome"]

    ref_env = SoldierEnv(config=config)
    obs, info = ref_env.reset(seed=episode_seed)
    zero_action = np.zeros(3, dtype=np.float32)
    while not info["just_detected"]:
        obs, reward, term, trunc, info = ref_env.step(zero_action)
    ref_step = 0
    ref_outcome = None
    ref_done = False
    while not ref_done:
        obs, reward, term, trunc, info = ref_env.step(action_fn(ref_step))
        ref_step += 1
        ref_done = term or trunc
        if ref_done:
            ref_outcome = info["outcome"]

    assert wrapper_outcome == ref_outcome
    assert step == ref_step


# ---------------------------------------------------------------------------
# K/L. Episode-seed reproducibility and root-dependence
# ---------------------------------------------------------------------------
def test_episode_seed_sequence_reproducible():
    a = build_episode_seed_sequence(45, 10)
    b = build_episode_seed_sequence(45, 10)
    assert a == b


def test_episode_seed_sequence_differs_across_roots():
    a = build_episode_seed_sequence(45, 10)
    b = build_episode_seed_sequence(46, 10)
    c = build_episode_seed_sequence(47, 10)
    assert a != b
    assert b != c
    assert a != c
    # Disjointness over a larger sample.
    n = 1000
    sa = set(build_episode_seed_sequence(45, n))
    sb = set(build_episode_seed_sequence(46, n))
    sc = set(build_episode_seed_sequence(47, n))
    assert not (sa & sb)
    assert not (sa & sc)
    assert not (sb & sc)


# ---------------------------------------------------------------------------
# M. Direct/Kalman paired root -> same environment episode seeds
# ---------------------------------------------------------------------------
def test_direct_kalman_paired_root_same_episode_seeds():
    root = 45
    env_d = PostDetectionTrainingEnv(config=build_env_config("direct"), root_seed=root, episode_seed_pool_size=20)
    env_k = PostDetectionTrainingEnv(config=build_env_config("kalman"), root_seed=root, episode_seed_pool_size=20)
    assert env_d._episode_seeds == env_k._episode_seeds
    env_d.reset()
    env_k.reset()
    assert env_d.last_episode_seed == env_k.last_episode_seed


# ---------------------------------------------------------------------------
# N. Direct/Kalman same environment seed -> same pre-estimator detection state
# ---------------------------------------------------------------------------
def test_direct_kalman_same_detection_state():
    root = _DEV_ROOT + 4
    env_d = PostDetectionTrainingEnv(config=build_env_config("direct"), root_seed=root)
    env_k = PostDetectionTrainingEnv(config=build_env_config("kalman"), root_seed=root)
    obs_d, info_d = env_d.reset()
    obs_k, info_k = env_k.reset()
    assert env_d.last_episode_seed == env_k.last_episode_seed
    assert np.allclose(info_d["soldier_pos"], info_k["soldier_pos"])
    assert np.allclose(info_d["defender_pos"], info_k["defender_pos"])
    assert np.allclose(info_d["enemy_pos"], info_k["enemy_pos"])
    assert np.allclose(info_d["enemy_vel"], info_k["enemy_vel"])


# ---------------------------------------------------------------------------
# O/P. Spaces
# ---------------------------------------------------------------------------
def test_observation_and_action_shapes():
    env = PostDetectionTrainingEnv(config=build_env_config("direct"), root_seed=_DEV_ROOT + 5)
    assert env.observation_space.shape == (16,)
    assert env.action_space.shape == (3,)
    obs, info = env.reset()
    assert obs.shape == (16,)


# ---------------------------------------------------------------------------
# Q/R. No ground-truth leakage
# ---------------------------------------------------------------------------
def test_direct_no_truth_leakage():
    config = build_env_config("direct")
    env = PostDetectionTrainingEnv(config=config, root_seed=_DEV_ROOT + 6)
    obs, info = env.reset()
    true_enemy_norm = np.array([info["enemy_pos"][0] / config.L, info["enemy_pos"][1] / config.L,
                                  info["enemy_pos"][2] / config.max_altitude])
    assert not np.allclose(obs[10:13], true_enemy_norm, atol=1e-12)


def test_kalman_no_truth_leakage():
    config = build_env_config("kalman")
    env = PostDetectionTrainingEnv(config=config, root_seed=_DEV_ROOT + 6)
    obs, info = env.reset()
    true_enemy_norm = np.array([info["enemy_pos"][0] / config.L, info["enemy_pos"][1] / config.L,
                                  info["enemy_pos"][2] / config.max_altitude])
    assert not np.allclose(obs[10:13], true_enemy_norm, atol=1e-12)


# ---------------------------------------------------------------------------
# S. Direct hostile finite-difference velocity invalid at first measurement
# ---------------------------------------------------------------------------
def test_direct_velocity_invalid_at_first_measurement():
    env = PostDetectionTrainingEnv(config=build_env_config("direct"), root_seed=_DEV_ROOT + 7)
    obs, info = env.reset()
    assert info["enemy_measurement_velocity_valid"] is False
    assert np.allclose(info["enemy_measurement_velocity"], 0.0)
    assert np.allclose(obs[13:16], 0.0)


# ---------------------------------------------------------------------------
# T. Kalman initializes at first measurement
# ---------------------------------------------------------------------------
def test_kalman_initializes_at_first_measurement():
    env = PostDetectionTrainingEnv(config=build_env_config("kalman"), root_seed=_DEV_ROOT + 8)
    obs, info = env.reset()
    assert info["e_hat"] is not None
    assert info["v_hat"] is not None
    assert info["enemy_measurement"] is not None


# ---------------------------------------------------------------------------
# U. PPO timesteps exclude standby transitions
# ---------------------------------------------------------------------------
def test_ppo_num_timesteps_excludes_standby():
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        return Monitor(PostDetectionTrainingEnv(config=build_env_config("direct"), root_seed=_DEV_ROOT + 9))

    vec_env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", vec_env, n_steps=64, batch_size=32, n_epochs=1, verbose=0, seed=0)
    model.learn(total_timesteps=64)
    assert model.num_timesteps == 64, (
        "num_timesteps must equal exactly the requested PPO-controlled step budget; "
        "internal standby fast-forward steps performed during reset() must not be counted."
    )
    vec_env.close()


# ---------------------------------------------------------------------------
# V. SB3 Monitor/VecEnv compatibility (smoke)
# ---------------------------------------------------------------------------
def test_sb3_monitor_vecenv_compatibility():
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        return Monitor(PostDetectionTrainingEnv(config=build_env_config("kalman"), root_seed=_DEV_ROOT + 10))

    vec_env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", vec_env, n_steps=64, batch_size=32, n_epochs=1, verbose=0, seed=0)
    model.learn(total_timesteps=64)
    import torch
    for p in model.policy.parameters():
        assert torch.isfinite(p).all(), "NaN/Inf found in policy parameters after smoke training"
    vec_env.close()


# ---------------------------------------------------------------------------
# Bonus: reserved seed ranges sanity (protocol constants)
# ---------------------------------------------------------------------------
def test_reserved_seed_ranges_disjoint_from_dev_root():
    assert _DEV_ROOT not in TRAINING_SEEDS
    assert not (VALIDATION_SEED_OFFSET <= _DEV_ROOT <= DIAGNOSTIC_SEED_OFFSET + 300)
    assert not (FINAL_TEST_SEED_START <= _DEV_ROOT <= FINAL_TEST_SEED_END)
    assert _DEV_ROOT < FUTURE_ROBUSTNESS_SEED_START


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    tests.sort(key=lambda kv: kv[0])
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
