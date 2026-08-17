"""
Per-episode rollout for the defender-maneuverability (accel x turn-rate)
interaction grid. Superset of prior eval_utils' defender/hostile operational
diagnostics, acquisition/post-detection metrics, and estimator metrics, plus
a post-detection closing-efficiency metric (item 28):

    post_detection_closing_efficiency =
        (defender_enemy_dist_at_detection - min_defender_enemy_dist_post_detection)
        / defender_path_length_post_detection

Reuses the SAME physical-limit/numerical-health checks (1e-3 tolerance,
unchanged) via experiments.final_eval_utils. _check_physical_limits reads
config.defender_max_accel / config.defender_max_turn_rate_deg dynamically, so
it automatically enforces the CELL-SPECIFIC limits for whichever config is
passed in (item 42 / test Z).
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


def run_maneuverability_episode(env, policy, seed: int, dt: float) -> dict:
    """Run one episode for the maneuverability sweep and return the full metrics dict."""
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
    min_defender_enemy_dist_post = None
    min_enemy_soldier_dist_post = None

    detection_step = None
    intercept_step = None
    defender_pos_at_detection = None
    enemy_pos_at_detection = None
    defender_enemy_dist_at_detection = None
    enemy_soldier_dist_at_detection = None

    defender_path_length = 0.0
    defender_path_length_post_detection = 0.0
    prev_defender_pos = info["defender_pos"].copy()
    defender_speeds, defender_accels, defender_turn_rates = [], [], []
    accel_sat_count = turn_sat_count = climb_sat_count = 0
    accel_used_sum = turn_rate_used_sum = 0.0

    enemy_path_length = 0.0
    prev_enemy_pos = info["enemy_pos"].copy()
    enemy_speeds = []
    enemy_accel_sat_count = enemy_turn_sat_count = enemy_vert_sat_count = 0

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
        step_path = float(np.linalg.norm(cur_def_pos - prev_defender_pos))
        defender_path_length += step_path
        if detection_step is not None:
            defender_path_length_post_detection += step_path
        prev_defender_pos = cur_def_pos.copy()
        defender_speeds.append(info["defender_speed"])
        defender_accels.append(info["defender_accel"])
        defender_turn_rates.append(info["defender_turn_rate"])
        accel_used_sum += info["defender_accel"]
        turn_rate_used_sum += info["defender_turn_rate"]
        accel_sat_count += int(info["defender_accel_saturated"])
        turn_sat_count += int(info["defender_turn_saturated"])
        climb_sat_count += int(info["defender_climb_saturated"])

        cur_enemy_pos = info["enemy_pos"]
        enemy_path_length += float(np.linalg.norm(cur_enemy_pos - prev_enemy_pos))
        prev_enemy_pos = cur_enemy_pos.copy()
        enemy_speeds.append(float(np.linalg.norm(info["enemy_vel"])))
        enemy_accel_sat_count += int(info["enemy_accel_saturated"])
        enemy_turn_sat_count += int(info["enemy_turn_saturated"])
        enemy_vert_sat_count += int(info["enemy_climb_saturated"])

        activation = float(info["enemy_evasion_activation"])
        evasion_activations.append(activation)
        if info["enemy_evasion_active"]:
            evasion_active_count += 1

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

    closing_efficiency = float("nan")
    if (
        detection_step is not None
        and defender_enemy_dist_at_detection is not None
        and min_defender_enemy_dist_post is not None
        and defender_path_length_post_detection > 0.0
    ):
        net_reduction = defender_enemy_dist_at_detection - min_defender_enemy_dist_post
        closing_efficiency = net_reduction / defender_path_length_post_detection

    row = {
        "eval_seed": seed,
        "defender_max_accel": config.defender_max_accel,
        "defender_max_turn_rate_deg": config.defender_max_turn_rate_deg,

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
        "defender_path_length_post_detection": float(defender_path_length_post_detection),
        "post_detection_closing_efficiency": closing_efficiency,

        "mean_defender_speed": float(np.mean(defender_speeds)) if defender_speeds else float("nan"),
        "max_defender_speed": float(np.max(defender_speeds)) if defender_speeds else float("nan"),
        "mean_defender_accel": float(np.mean(defender_accels)) if defender_accels else float("nan"),
        "max_defender_accel": float(np.max(defender_accels)) if defender_accels else float("nan"),
        "mean_defender_turn_rate": float(np.mean(defender_turn_rates)) if defender_turn_rates else float("nan"),
        "max_defender_turn_rate": float(np.max(defender_turn_rates)) if defender_turn_rates else float("nan"),
        "fraction_defender_accel_saturated": accel_sat_count / n_steps if n_steps else float("nan"),
        "fraction_defender_turn_saturated": turn_sat_count / n_steps if n_steps else float("nan"),
        "fraction_defender_climb_saturated": climb_sat_count / n_steps if n_steps else float("nan"),
        # Dimensionless controller-utilization ratios (item 38): mean actual / allowed.
        "mean_accel_utilization": (
            (accel_used_sum / n_steps) / config.defender_max_accel if n_steps and config.defender_max_accel else float("nan")
        ),
        "mean_turn_rate_utilization": (
            (turn_rate_used_sum / n_steps) / config.defender_max_turn_rate_deg if n_steps and config.defender_max_turn_rate_deg else float("nan")
        ),

        "mean_enemy_speed": float(np.mean(enemy_speeds)) if enemy_speeds else float("nan"),
        "max_enemy_speed": float(np.max(enemy_speeds)) if enemy_speeds else float("nan"),
        "fraction_enemy_accel_saturated": enemy_accel_sat_count / n_steps if n_steps else float("nan"),
        "fraction_enemy_turn_saturated": enemy_turn_sat_count / n_steps if n_steps else float("nan"),
        "fraction_enemy_vertical_saturated": enemy_vert_sat_count / n_steps if n_steps else float("nan"),
        "enemy_path_length": float(enemy_path_length),
        "evasion_activated_episode": 1 if evasion_active_count > 0 else 0,
        "fraction_steps_evasion_active": evasion_active_count / n_steps if n_steps else float("nan"),
        "mean_evasion_activation": float(np.mean(evasion_activations)) if evasion_activations else float("nan"),

        "mean_position_estimation_error": float(np.mean(estimation_errors)) if estimation_errors else float("nan"),
        "position_estimation_rmse": (
            float(np.sqrt(np.mean(np.square(estimation_errors)))) if estimation_errors else float("nan")
        ),
        "final_position_estimation_error": float(estimation_errors[-1]) if estimation_errors else float("nan"),
        "n_position_estimation_samples": len(estimation_errors),
    }
    return row
