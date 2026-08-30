"""
Episode runner extension for LR-PPO instances in the Lead-Residual 200-seed
diagnostic. Mirrors experiments/post_detection_diagnostic_eval_utils.py::
run_diagnostic_episode() exactly (same canonical mission loop, same row
schema, same physical/numerical checks) but additionally records the
Lead-relative hybrid-action angle at every CONTROLLED (post-detection,
info["controller_action_executed"]=True) timestep, needed for the residual
angle analysis (task items 18-19). Non-LR-PPO instances continue using the
unmodified, already-validated run_diagnostic_episode() directly.
"""

from __future__ import annotations

import numpy as np

from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env
from uav_defend.policies.residual.lead_residual_composition import angle_between_deg

from experiments.final_eval_utils import PhysicalLimitViolation, _check_physical_limits


def run_lr_ppo_diagnostic_episode(env, lr_policy, seed: int, dt: float, record_trajectory: bool = False) -> dict:
    """
    Run one full canonical mission for an LR-PPO instance (LeadResidualPPOPolicyWrapper),
    returning the SAME row schema as run_diagnostic_episode() PLUS a
    "step_angles" list (Lead-relative hybrid-action angle in degrees, one
    entry per controlled/post-detection step).
    """
    obs, info = env.reset(seed=seed)
    lr_policy.reset()
    estimator_mode = estimator_mode_for_env(env)
    assert estimator_mode == "measurement", "LR-PPO uses the measurement/Direct track only"
    config = env.config

    physical_violations = 0

    def _check(info_dict):
        nonlocal physical_violations
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
    step_angles = []

    step = 0
    done = False
    while not done:
        policy_info = build_policy_info(info, estimator_mode)
        lead_direction = np.asarray(lr_policy._lead_policy.act(obs, dict(policy_info)), dtype=np.float32)
        action = lr_policy.act(obs, policy_info)
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
            estimate = info["enemy_measurement"]
            if estimate is not None:
                estimation_errors.append(float(np.linalg.norm(info["enemy_pos"] - estimate)))

        if info["controller_action_executed"] and float(np.linalg.norm(lead_direction)) > 1e-8:
            step_angles.append(angle_between_deg(action, lead_direction))

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
        "step_angles": step_angles,
    }
    if record_trajectory:
        row["pre_detection_trajectory"] = trajectory
    return row
