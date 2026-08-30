"""
Episode runner for the post-detection diagnostic (200-seed) evaluation.
Reuses the canonical SoldierEnv unmodified; no wrapper is used here (full
canonical mission, exactly as required for this diagnostic -- see
run_post_detection_diagnostic.py).
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


def run_diagnostic_episode(env, policy, seed: int, dt: float, record_trajectory: bool = False) -> dict:
    """
    Run one full canonical mission (reset -> standby -> detection -> control
    -> terminal) and return a metrics dict. If record_trajectory=True, also
    include a 'pre_detection_trajectory' list of per-step state snapshots
    (soldier_pos, defender_pos, defender_vel, enemy_pos, enemy_vel,
    enemy_detected) through and including the detection transition, used
    ONLY for the acquisition-fairness cross-method comparison (never written
    to the per-episode results CSV).
    """
    obs, info = env.reset(seed=seed)
    policy.reset()
    estimator_mode = estimator_mode_for_env(env)
    config = env.config

    physical_violations = 0
    max_exceedance = 0.0

    def _check(info_dict):
        nonlocal physical_violations, max_exceedance
        try:
            _check_physical_limits(info_dict, config)
        except PhysicalLimitViolation:
            physical_violations += 1

    _check(info)

    trajectory = []
    if record_trajectory:
        trajectory.append(dict(
            step=0, soldier_pos=info["soldier_pos"].copy(), defender_pos=info["defender_pos"].copy(),
            defender_vel=info["defender_vel"].copy(), enemy_pos=info["enemy_pos"].copy(),
            enemy_vel=info["enemy_vel"].copy(), enemy_detected=info["enemy_detected"],
        ))

    detection_step = None
    detection_time = None
    intercept_step = None
    min_enemy_soldier_dist = info["enemy_soldier_dist"]
    estimation_errors = []

    step = 0
    done = False
    while not done:
        policy_info = build_policy_info(info, estimator_mode)
        action = policy.act(obs, policy_info)
        obs, reward, term, trunc, info = env.step(action)
        step += 1
        _check(info)
        min_enemy_soldier_dist = min(min_enemy_soldier_dist, info["enemy_soldier_dist"])

        if record_trajectory and detection_step is None:
            trajectory.append(dict(
                step=step, soldier_pos=info["soldier_pos"].copy(), defender_pos=info["defender_pos"].copy(),
                defender_vel=info["defender_vel"].copy(), enemy_pos=info["enemy_pos"].copy(),
                enemy_vel=info["enemy_vel"].copy(), enemy_detected=info["enemy_detected"],
            ))

        if info["just_detected"]:
            detection_step = step
            detection_time = step * dt

        if info["enemy_detected"]:
            estimate = info["e_hat"] if estimator_mode == "kalman" else info["enemy_measurement"]
            if estimate is not None:
                estimation_errors.append(float(np.linalg.norm(info["enemy_pos"] - estimate)))

        if info["outcome"] == "intercepted" and intercept_step is None:
            intercept_step = step

        done = term or trunc

    total_steps = step
    outcome = info["outcome"]
    post_detection_steps = (total_steps - detection_step) if detection_step is not None else None

    row = {
        "environment_seed": seed,
        "outcome": outcome,
        "success": 1 if outcome == "intercepted" else 0,
        "soldier_caught": 1 if outcome == "soldier_caught" else 0,
        "unsafe_intercept": 1 if outcome == "unsafe_intercept" else 0,
        "timeout": 1 if outcome == "timeout" else 0,
        "total_steps": total_steps,
        "detection_step": detection_step,
        "detection_time_seconds": detection_time,
        "post_detection_steps": post_detection_steps,
        "intercept_step": intercept_step,
        "intercept_time_seconds": (intercept_step * dt) if intercept_step is not None else None,
        "min_enemy_soldier_dist": float(min_enemy_soldier_dist),
        "physical_limit_violations": physical_violations,
        "mean_tracking_error": float(np.mean(estimation_errors)) if estimation_errors else None,
        "final_tracking_error": float(estimation_errors[-1]) if estimation_errors else None,
        "tracking_rmse": float(np.sqrt(np.mean(np.square(estimation_errors)))) if estimation_errors else None,
        "n_tracking_samples": len(estimation_errors),
    }
    if record_trajectory:
        row["pre_detection_trajectory"] = trajectory
    return row
