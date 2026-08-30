"""
Extended per-episode rollout for the hostile-evasion-strength robustness sweep.

Superset of experiments/mobility_grid_eval_utils.py's field set (detection
geometry, post-detection metrics, defender + hostile operational/dynamics
diagnostics, estimator metrics with n_position_estimation_samples) PLUS
detailed evasion metrics required by this sweep (items 23/24):
  - mean/max evasion activation (info["enemy_evasion_activation"])
  - mean/max EFFECTIVE evasion weight = gain * activation
    (info["enemy_evasion_weight"], already computed by the environment --
    NOT re-derived independently, see SoldierEnv._compute_enemy_evasion)
  - first evasion activation time, and whether it precedes detection
  - pre-detection evasion timing (time between first evasion activation and
    detection, when both occur)

Reuses the SAME physical-limit/numerical-health checks (1e-3 tolerance,
unchanged) via experiments.final_eval_utils. Does NOT change environment
dynamics -- only observes existing info dict fields.
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


def run_hostile_evasion_episode(env, policy, seed: int, dt: float, evasion_gain: float) -> dict:
    """Run one episode for the hostile-evasion-strength sweep and return the full metrics dict."""
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
    defender_speeds, defender_accels, defender_turn_rates = [], [], []
    accel_sat_count = turn_sat_count = climb_sat_count = 0

    enemy_path_length = 0.0
    prev_enemy_pos = info["enemy_pos"].copy()
    enemy_speeds, enemy_horiz_speeds, enemy_vert_speeds = [], [], []
    enemy_accel_sat_count = enemy_turn_sat_count = enemy_vert_sat_count = 0
    ground_hits = ceiling_hits = wall_hits = 0

    evasion_active_before_detection = False
    first_evasion_step = None
    evasion_active_count = 0
    evasion_activations = []
    effective_evasion_weights = []

    estimation_errors = []

    step = 0
    done = False

    L = config.L
    H = config.max_altitude

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

        cur_def_pos = info["defender_pos"]
        defender_path_length += float(np.linalg.norm(cur_def_pos - prev_defender_pos))
        prev_defender_pos = cur_def_pos.copy()
        defender_speeds.append(info["defender_speed"])
        defender_accels.append(info["defender_accel"])
        defender_turn_rates.append(info["defender_turn_rate"])
        accel_sat_count += int(info["defender_accel_saturated"])
        turn_sat_count += int(info["defender_turn_saturated"])
        climb_sat_count += int(info["defender_climb_saturated"])

        cur_enemy_pos = info["enemy_pos"]
        enemy_path_length += float(np.linalg.norm(cur_enemy_pos - prev_enemy_pos))
        prev_enemy_pos = cur_enemy_pos.copy()
        enemy_vel = info["enemy_vel"]
        enemy_speed = float(np.linalg.norm(enemy_vel))
        enemy_horiz_speed = float(np.hypot(enemy_vel[0], enemy_vel[1]))
        enemy_vert_speed = float(abs(enemy_vel[2]))
        enemy_speeds.append(enemy_speed)
        enemy_horiz_speeds.append(enemy_horiz_speed)
        enemy_vert_speeds.append(enemy_vert_speed)
        enemy_accel_sat_count += int(info["enemy_accel_saturated"])
        enemy_turn_sat_count += int(info["enemy_turn_saturated"])
        enemy_vert_sat_count += int(info["enemy_climb_saturated"])
        if abs(cur_enemy_pos[2] - 0.0) < 1e-6:
            ground_hits += 1
        if abs(cur_enemy_pos[2] - H) < 1e-6:
            ceiling_hits += 1
        if abs(abs(cur_enemy_pos[0]) - L) < 1e-6 or abs(abs(cur_enemy_pos[1]) - L) < 1e-6:
            wall_hits += 1

        activation = float(info["enemy_evasion_activation"])
        effective_weight = float(info["enemy_evasion_weight"])
        evasion_activations.append(activation)
        effective_evasion_weights.append(effective_weight)
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
    time_evasion_to_detection = (
        (detection_step - first_evasion_step) * dt
        if (first_evasion_step is not None and detection_step is not None and detection_step >= first_evasion_step)
        else None
    )

    row = {
        "eval_seed": seed,
        "enemy_evasion_gain": evasion_gain,
        "enemy_evasion_radius": config.enemy_evasion_radius,

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

        "evasion_activated_episode": 1 if evasion_active_count > 0 else 0,
        "evasion_active_before_detection": 1 if evasion_active_before_detection else 0,
        "first_evasion_step": first_evasion_step if first_evasion_step is not None else float("nan"),
        "first_evasion_time_seconds": _s2sec(first_evasion_step),
        "time_evasion_onset_to_detection_seconds": (
            time_evasion_to_detection if time_evasion_to_detection is not None else float("nan")
        ),
        "fraction_steps_evasion_active": evasion_active_count / n_steps if n_steps else float("nan"),
        "mean_evasion_activation": float(np.mean(evasion_activations)) if evasion_activations else float("nan"),
        "max_evasion_activation": float(np.max(evasion_activations)) if evasion_activations else float("nan"),
        "mean_effective_evasion_weight": float(np.mean(effective_evasion_weights)) if effective_evasion_weights else float("nan"),
        "max_effective_evasion_weight": float(np.max(effective_evasion_weights)) if effective_evasion_weights else float("nan"),

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
        "fraction_defender_accel_saturated": accel_sat_count / n_steps if n_steps else float("nan"),
        "fraction_defender_turn_saturated": turn_sat_count / n_steps if n_steps else float("nan"),
        "fraction_defender_climb_saturated": climb_sat_count / n_steps if n_steps else float("nan"),

        "mean_enemy_speed": float(np.mean(enemy_speeds)) if enemy_speeds else float("nan"),
        "max_enemy_speed": float(np.max(enemy_speeds)) if enemy_speeds else float("nan"),
        "mean_enemy_horizontal_speed": float(np.mean(enemy_horiz_speeds)) if enemy_horiz_speeds else float("nan"),
        "mean_abs_enemy_vertical_speed": float(np.mean(enemy_vert_speeds)) if enemy_vert_speeds else float("nan"),
        "fraction_enemy_accel_saturated": enemy_accel_sat_count / n_steps if n_steps else float("nan"),
        "fraction_enemy_turn_saturated": enemy_turn_sat_count / n_steps if n_steps else float("nan"),
        "fraction_enemy_climb_or_vertical_saturated": enemy_vert_sat_count / n_steps if n_steps else float("nan"),
        "enemy_path_length": float(enemy_path_length),
        "ground_boundary_hits": ground_hits,
        "ceiling_hits": ceiling_hits,
        "horizontal_wall_hits": wall_hits,

        "mean_position_estimation_error": float(np.mean(estimation_errors)) if estimation_errors else float("nan"),
        "position_estimation_rmse": (
            float(np.sqrt(np.mean(np.square(estimation_errors)))) if estimation_errors else float("nan")
        ),
        "final_position_estimation_error": float(estimation_errors[-1]) if estimation_errors else float("nan"),
        "n_position_estimation_samples": len(estimation_errors),
    }
    return row
