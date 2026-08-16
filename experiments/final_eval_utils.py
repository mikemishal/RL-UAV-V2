"""
Extended per-episode rollout for the locked final nominal evaluation.

experiments/eval_utils.run_episode() remains the general-purpose evaluator
used by other scripts; this module provides a SEPARATE, purpose-built
episode runner (run_final_episode) that additionally collects the defender
dynamics, hostile-evasion, and estimator-error metrics required by the final
nominal evaluation (experiments/final_experiment_protocol.py items 12-19),
plus live physical-limit and numerical-health checks (items 40-41).

Do NOT change environment dynamics/reward/sensing here -- this module only
OBSERVES the existing info dict fields already computed by SoldierEnv.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env

# Hard physical limits (EnvConfig defaults) -- a violation indicates a
# genuine dynamics bug, not something to silently patch (see item 40).
_LIMIT_TOLERANCE = 1e-6


class PhysicalLimitViolation(RuntimeError):
    """Raised when a recorded step violates a hard physical limit."""


class NumericalHealthViolation(RuntimeError):
    """Raised when NaN/Inf is found in an action, reward, position, or observation."""


def sha256_of_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes (used to freeze PPO model identities)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_physical_limits(info: dict, config) -> None:
    if info["defender_speed"] > config.v_d + _LIMIT_TOLERANCE:
        raise PhysicalLimitViolation(f"defender speed {info['defender_speed']} > v_d={config.v_d}")
    if info["defender_accel"] > config.defender_max_accel + _LIMIT_TOLERANCE:
        raise PhysicalLimitViolation(f"defender accel {info['defender_accel']} > {config.defender_max_accel}")
    if info["defender_turn_rate"] > config.defender_max_turn_rate_deg + _LIMIT_TOLERANCE:
        raise PhysicalLimitViolation(f"defender turn rate {info['defender_turn_rate']} > {config.defender_max_turn_rate_deg}")
    vz_d = info["defender_vel"][2]
    if vz_d > config.defender_max_climb_rate + _LIMIT_TOLERANCE:
        raise PhysicalLimitViolation(f"defender climb {vz_d} > {config.defender_max_climb_rate}")
    if -vz_d > config.defender_max_descent_rate + _LIMIT_TOLERANCE:
        raise PhysicalLimitViolation(f"defender descent {-vz_d} > {config.defender_max_descent_rate}")

    if info["enemy_speed"] > config.v_e + _LIMIT_TOLERANCE:
        raise PhysicalLimitViolation(f"enemy speed {info['enemy_speed']} > v_e={config.v_e}")
    if info["enemy_accel"] > config.enemy_max_accel + _LIMIT_TOLERANCE:
        raise PhysicalLimitViolation(f"enemy accel {info['enemy_accel']} > {config.enemy_max_accel}")
    if info["enemy_turn_rate"] > config.enemy_max_turn_rate_deg + _LIMIT_TOLERANCE:
        raise PhysicalLimitViolation(f"enemy turn rate {info['enemy_turn_rate']} > {config.enemy_max_turn_rate_deg}")
    vz_e = info["enemy_vel"][2]
    if vz_e > config.enemy_max_climb_rate + _LIMIT_TOLERANCE:
        raise PhysicalLimitViolation(f"enemy climb {vz_e} > {config.enemy_max_climb_rate}")
    if -vz_e > config.enemy_max_descent_rate + _LIMIT_TOLERANCE:
        raise PhysicalLimitViolation(f"enemy descent {-vz_e} > {config.enemy_max_descent_rate}")


def _check_numerical_health(action: np.ndarray, obs: np.ndarray, reward: float, info: dict) -> None:
    if not np.all(np.isfinite(action)):
        raise NumericalHealthViolation(f"non-finite action: {action}")
    if not np.all(np.isfinite(obs)):
        raise NumericalHealthViolation(f"non-finite observation: {obs}")
    if not math.isfinite(reward):
        raise NumericalHealthViolation(f"non-finite reward: {reward}")
    for key in ("soldier_pos", "defender_pos", "enemy_pos"):
        if not np.all(np.isfinite(info[key])):
            raise NumericalHealthViolation(f"non-finite {key}: {info[key]}")


def run_final_episode(env, policy, seed: int, dt: float) -> dict:
    """
    Run one episode for the final nominal evaluation and return an
    extended metrics dict (see experiments/final_experiment_protocol.py
    items 12-19 for the required field list). `seed` is the environment
    reset seed (== eval_seed for the caller).
    """
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
    detection_step = None
    intercept_step = None

    defender_path_length = 0.0
    prev_defender_pos = info["defender_pos"].copy()
    defender_speeds = []
    defender_accels = []
    defender_turn_rates = []
    accel_saturated_count = 0
    turn_saturated_count = 0
    climb_saturated_count = 0

    evasion_active_count = 0
    first_evasion_step = None
    evasion_activations = []          # activation value at EVERY step
    evasion_activations_when_active = []  # activation value only when active

    estimation_errors = []  # after-detection estimator error at each step

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

        # Defender dynamics
        cur_pos = info["defender_pos"]
        defender_path_length += float(np.linalg.norm(cur_pos - prev_defender_pos))
        prev_defender_pos = cur_pos.copy()
        defender_speeds.append(info["defender_speed"])
        defender_accels.append(info["defender_accel"])
        defender_turn_rates.append(info["defender_turn_rate"])
        accel_saturated_count += int(info["defender_accel_saturated"])
        turn_saturated_count += int(info["defender_turn_saturated"])
        climb_saturated_count += int(info["defender_climb_saturated"])

        # Hostile evasion
        activation = float(info["enemy_evasion_activation"])
        evasion_activations.append(activation)
        if info["enemy_evasion_active"]:
            evasion_active_count += 1
            evasion_activations_when_active.append(activation)
            if first_evasion_step is None:
                first_evasion_step = step

        # Estimator error (evaluation-only; NEVER exposed to the policy)
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

    def _steps_to_seconds(s):
        return float(s) * dt if s is not None else float("nan")

    post_detection_steps = (
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
        "detection_step": detection_step if detection_step is not None else float("nan"),
        "detection_time_seconds": _steps_to_seconds(detection_step),
        "intercept_step": intercept_step if intercept_step is not None else float("nan"),
        "intercept_time_seconds": _steps_to_seconds(intercept_step),
        "post_detection_intercept_steps": post_detection_steps if post_detection_steps is not None else float("nan"),
        "post_detection_intercept_seconds": _steps_to_seconds(post_detection_steps),

        "min_enemy_soldier_dist": float(min_enemy_soldier_dist),
        "min_defender_enemy_dist": float(min_defender_enemy_dist),
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
        "first_evasion_step": first_evasion_step if first_evasion_step is not None else float("nan"),
        "fraction_steps_evasion_active": evasion_active_count / n_steps if n_steps else float("nan"),
        "max_evasion_activation": float(np.max(evasion_activations)) if evasion_activations else float("nan"),
        "mean_evasion_activation_when_active": (
            float(np.mean(evasion_activations_when_active)) if evasion_activations_when_active else float("nan")
        ),

        "mean_position_estimation_error": float(np.mean(estimation_errors)) if estimation_errors else float("nan"),
        "position_estimation_rmse": (
            float(np.sqrt(np.mean(np.square(estimation_errors)))) if estimation_errors else float("nan")
        ),
        "final_position_estimation_error": float(estimation_errors[-1]) if estimation_errors else float("nan"),

        "total_reward": float(total_reward),
    }
    return row
