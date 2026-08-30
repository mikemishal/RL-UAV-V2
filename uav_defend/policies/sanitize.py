"""
Policy-information sanitization.

Structural enforcement of ground-truth independence: rather than relying
solely on each policy's self-discipline to never read `info["enemy_pos"]` /
`info["enemy_vel"]`, evaluation pipelines should pass policies a SANITIZED
copy of the environment's info dict -- containing only the fields
legitimately available under the configured estimator track -- built by
`build_policy_info()` below.

This module is the single source of truth for "what is a legitimate policy
input" and is used by:
    - experiments/eval_utils.py (run_episode, run_episode_with_trajectory)
    - test_experiment_fairness.py (structural no-leakage tests)
    - any other rollout loop that calls policy.act(obs, info)

IMPORTANT: only the object passed to `policy.act()` should be sanitized.
Evaluators/metrics code should keep using the environment's full,
un-sanitized info dict for scoring (tracking_error, enemy_pos-based
distances, outcome, etc.) -- those are legitimate EVALUATION-time uses of
ground truth, never policy-time uses.

Usage:
    from uav_defend.policies.sanitize import build_policy_info

    estimator_mode = "kalman" if env.config.use_kalman_tracking else "measurement"
    while not done:
        policy_info = build_policy_info(info, estimator_mode)
        action = policy.act(obs, policy_info)
        obs, reward, done, truncated, info = env.step(action)  # full info kept for metrics
"""

from __future__ import annotations

VALID_ESTIMATOR_MODES = ("measurement", "kalman")

# Ground-truth/privileged fields that must NEVER reach policy.act(), under
# any estimator mode.
_GROUND_TRUTH_KEYS = frozenset({
    "enemy_pos",
    "enemy_vel",
})

# Fields legitimately available to every policy regardless of estimator mode:
# the defender's own true state (always legitimate -- it's the agent's own
# state, not the hostile's), the soldier's true state (the asset being
# defended, directly observable), and detection bookkeeping.
_COMMON_KEYS = (
    "soldier_pos",
    "defender_pos",
    "defender_vel",
    "enemy_detected",
)

# Direct/measurement track: raw noisy hostile-position measurement plus the
# environment's standardized finite-difference measurement velocity.
_MEASUREMENT_KEYS = _COMMON_KEYS + (
    "enemy_measurement",
    "enemy_measurement_velocity",
    "enemy_measurement_velocity_valid",
)

# Kalman track: filtered hostile position/velocity estimate.
_KALMAN_KEYS = _COMMON_KEYS + (
    "e_hat",
    "v_hat",
)


def build_policy_info(info: dict, estimator_mode: str) -> dict:
    """
    Return a sanitized copy of `info` containing ONLY the fields legitimately
    available to a policy under the given estimator_mode.

    Args:
        info: The environment's full info dict (as returned by SoldierEnv).
        estimator_mode: "measurement" (Direct track) or "kalman" (Kalman track).

    Returns:
        A new dict containing only legitimate fields. Keys absent from the
        source `info` are simply omitted (matching the `info.get(key)`
        pattern used throughout every existing policy) -- never set to None
        as a substitute for omission.

    Raises:
        ValueError: If estimator_mode is not one of VALID_ESTIMATOR_MODES.
    """
    if estimator_mode not in VALID_ESTIMATOR_MODES:
        raise ValueError(
            f"estimator_mode must be one of {VALID_ESTIMATOR_MODES}, got '{estimator_mode}'"
        )

    keys = _MEASUREMENT_KEYS if estimator_mode == "measurement" else _KALMAN_KEYS
    sanitized = {key: info[key] for key in keys if key in info}

    # Defense in depth: unconditionally strip ground-truth keys even if a
    # future edit accidentally added one to the allow-lists above.
    for gt_key in _GROUND_TRUTH_KEYS:
        sanitized.pop(gt_key, None)

    return sanitized


def estimator_mode_for_env(env) -> str:
    """
    Infer the canonical estimator_mode string ("measurement" | "kalman")
    from an environment instance's configuration.

    Args:
        env: A SoldierEnv instance (or any object exposing
            `env.config.use_kalman_tracking`).

    Returns:
        "kalman" if env.config.use_kalman_tracking else "measurement".
    """
    return "kalman" if env.config.use_kalman_tracking else "measurement"
