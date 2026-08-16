"""
Extended per-episode rollout for the measurement-noise robustness sweep.

Identical field set to experiments/robustness_eval_utils.run_robustness_episode
(detection geometry, post-detection metrics, whole-episode evasion, full
operational metrics, estimator metrics) PLUS n_position_estimation_samples
(item 24's "also retain the number of valid estimation samples"). Reuses the
SAME physical-limit/numerical-health checks (1e-3 tolerance, unchanged) via
experiments.final_eval_utils.
"""

from __future__ import annotations

import numpy as np

from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env

from experiments.final_eval_utils import (  # noqa: F401 -- re-exported for convenience
    sha256_of_file,
    PhysicalLimitViolation,
    NumericalHealthViolation,
    _check_physical_limits,
    _check_numerical_health,
)


def run_measurement_noise_episode(env, policy, seed: int, dt: float) -> dict:
    """Run one episode for the measurement-noise sweep and return the full metrics dict."""
    obs, info = env.reset(seed=seed)
    policy.reset()
    estimator_mode = estimator_mode_for_env(env)
    config = env.config

    _check_physical_limits(info, config)
    if not np.all(np.isfinite(obs)):
        raise NumericalHealthViolation(f"non-finite initial observation: {obs}")

    total_reward = 0.0
    min_enemy_soldier_dist = info["enemy_soldier_dist"]
    min_defender_enemy_dist = info["defender_enemy_dist"]
    min_enemy_soldier_dist_post = None
    min_defender_enemy_dist_post = None

    detection_step = None
    intercept_step = None
    defender_pos_at_detection = None
    enemy_pos_at_detection = None
    defender_enemy_dist_at_detection = None
    enemy_soldier_dist_at_detection = None

    defender_path_length = 0.0
    prev_defender_pos = info["defender_pos"].copy()
    defender_speeds = []
    defender_accels = []
    defender_turn_rates = []
    accel_saturated_count = 0
    turn_saturated_count = 0
    climb_saturated_count = 0

    evasion_active_before_detection = False
    first_evasion_step = None
    evasion_active_count = 0
    evasion_activations = []

    estimation_errors = []

    step = 0
    done = False

    while not done:
        policy_info = build_policy_info(info, estimator_mode)
        action = policy.act(obs, policy_info)
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        total_reward += reward

        _check_physical_limits(info, config)
        _check_numerical_health(np.asarray(action, dtype=np.float64), obs, reward, info)

        was_detected_before_this_step = detection_step is not None
        min_enemy_soldier_dist = min(min_enemy_soldier_dist, info["enemy_soldier_dist"])
        min_defender_enemy_dist = min(min_defender_enemy_dist, info["defender_enemy_dist"])

        if info["enemy_detected"] and detection_step is None:
            detection_step = step
            defender_pos_at_detection = info["defender_pos"].copy()
            enemy_pos_at_detection = info["enemy_pos"].copy()
            defender_enemy_dist_at_detection = float(info["defender_enemy_dist"])
            enemy_soldier_dist_at_detection = float(info["enemy_soldier_dist"])

        if detection_step is not None:
            if min_defender_enemy_dist_post is None:
                min_defender_enemy_dist_post = info["defender_enemy_dist"]
                min_enemy_soldier_dist_post = info["enemy_soldier_dist"]
            else:
                min_defender_enemy_dist_post = min(min_defender_enemy_dist_post, info["defender_enemy_dist"])
                min_enemy_soldier_dist_post = min(min_enemy_soldier_dist_post, info["enemy_soldier_dist"])

        cur_pos = info["defender_pos"]
        defender_path_length += float(np.linalg.norm(cur_pos - prev_defender_pos))
        prev_defender_pos = cur_pos.copy()
        defender_speeds.append(info["defender_speed"])
        defender_accels.append(info["defender_accel"])
        defender_turn_rates.append(info["defender_turn_rate"])
        accel_saturated_count += int(info["defender_accel_saturated"])
        turn_saturated_count += int(info["defender_turn_saturated"])
        climb_saturated_count += int(info["defender_climb_saturated"])

        activation = float(info["enemy_evasion_activation"])
        evasion_activations.append(activation)
        if info["enemy_evasion_active"]:
            evasion_active_count += 1
            if not was_detected_before_this_step and detection_step is None:
                evasion_active_before_detection = True
                if first_evasion_step is None:
                    first_evasion_step = step

        if info["enemy_detected"]:
            estimate = info["e_hat"] if estimator_mode == "kalman" else info["enemy_measurement"]
            if estimate is not None:
                estimation_errors.append(float(np.linalg.norm(info["enemy_pos"] - estimate)))

        if done or truncated:
            break

    if info["outcome"] == "intercepted":
        intercept_step = step

    outcome = info["outcome"]
    n_steps = step

    def _s2sec(s):
        return float(s) * dt if s is not None else float("nan")

    post_detection_intercept_steps = (
        (intercept_step - detection_step)
        if (intercept_step is not None and detection_step is not None)
        else None
    )

    row = {
        "eval_seed": seed,
        "outcome": outcome,
        "success": 1 if outcome == "intercepted" else 0,
        "soldier_caught": 1 if outcome == "soldier_caught" else 0,
        "unsafe_intercept": 1 if outcome == "unsafe_intercept" else 0,
        "timeout": 1 if outcome == "timeout" else 0,

        "episode_length_steps": n_steps,
        "episode_length_seconds": n_steps * dt,
        "total_reward": float(total_reward),

        "detected": 1 if detection_step is not None else 0,
        "detection_step": detection_step if detection_step is not None else float("nan"),
        "detection_time_seconds": _s2sec(detection_step),
        "defender_enemy_distance_at_detection": (
            defender_enemy_dist_at_detection if defender_enemy_dist_at_detection is not None else float("nan")
        ),
        "enemy_soldier_distance_at_detection": (
            enemy_soldier_dist_at_detection if enemy_soldier_dist_at_detection is not None else float("nan")
        ),
        "defender_pos_at_detection_x": float(defender_pos_at_detection[0]) if defender_pos_at_detection is not None else float("nan"),
        "defender_pos_at_detection_y": float(defender_pos_at_detection[1]) if defender_pos_at_detection is not None else float("nan"),
        "defender_pos_at_detection_z": float(defender_pos_at_detection[2]) if defender_pos_at_detection is not None else float("nan"),
        "enemy_pos_at_detection_x": float(enemy_pos_at_detection[0]) if enemy_pos_at_detection is not None else float("nan"),
        "enemy_pos_at_detection_y": float(enemy_pos_at_detection[1]) if enemy_pos_at_detection is not None else float("nan"),
        "enemy_pos_at_detection_z": float(enemy_pos_at_detection[2]) if enemy_pos_at_detection is not None else float("nan"),
        "evasion_active_before_detection": 1 if evasion_active_before_detection else 0,
        "first_evasion_step": first_evasion_step if first_evasion_step is not None else float("nan"),

        "intercept_step": intercept_step if intercept_step is not None else float("nan"),
        "intercept_time_seconds": _s2sec(intercept_step),
        "post_detection_intercept_steps": post_detection_intercept_steps if post_detection_intercept_steps is not None else float("nan"),
        "post_detection_intercept_seconds": _s2sec(post_detection_intercept_steps),
        "min_defender_enemy_dist": float(min_defender_enemy_dist),
        "min_enemy_soldier_dist": float(min_enemy_soldier_dist),
        "min_defender_enemy_dist_post_detection": (
            float(min_defender_enemy_dist_post) if min_defender_enemy_dist_post is not None else float("nan")
        ),
        "min_enemy_soldier_dist_post_detection": (
            float(min_enemy_soldier_dist_post) if min_enemy_soldier_dist_post is not None else float("nan")
        ),
        "final_enemy_soldier_dist": float(info["enemy_soldier_dist"]),
        "final_defender_enemy_dist": float(info["defender_enemy_dist"]),

        "defender_path_length": float(defender_path_length),
        "mean_defender_speed": float(np.mean(defender_speeds)) if defender_speeds else float("nan"),
        "max_defender_speed": float(np.max(defender_speeds)) if defender_speeds else float("nan"),
        "mean_defender_accel": float(np.mean(defender_accels)) if defender_accels else float("nan"),
        "max_defender_accel": float(np.max(defender_accels)) if defender_accels else float("nan"),
        "mean_defender_turn_rate": float(np.mean(defender_turn_rates)) if defender_turn_rates else float("nan"),
        "max_defender_turn_rate": float(np.max(defender_turn_rates)) if defender_turn_rates else float("nan"),
        "fraction_defender_accel_saturated": accel_saturated_count / n_steps if n_steps else float("nan"),
        "fraction_defender_turn_saturated": turn_saturated_count / n_steps if n_steps else float("nan"),
        "fraction_defender_climb_saturated": climb_saturated_count / n_steps if n_steps else float("nan"),
        "evasion_activated_episode": 1 if evasion_active_count > 0 else 0,
        "fraction_steps_evasion_active": evasion_active_count / n_steps if n_steps else float("nan"),
        "max_evasion_activation": float(np.max(evasion_activations)) if evasion_activations else float("nan"),

        "mean_position_estimation_error": float(np.mean(estimation_errors)) if estimation_errors else float("nan"),
        "position_estimation_rmse": (
            float(np.sqrt(np.mean(np.square(estimation_errors)))) if estimation_errors else float("nan")
        ),
        "final_position_estimation_error": float(estimation_errors[-1]) if estimation_errors else float("nan"),
        "n_position_estimation_samples": len(estimation_errors),
    }
    return row
