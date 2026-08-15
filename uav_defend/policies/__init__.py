# uav_defend/policies - Defender decision logic
#
# This module provides both baseline (scripted) and RL (trained) policies
# with a unified interface for evaluation.
#
# Policy Organization:
# --------------------
#   baseline/   = Scripted hand-designed controllers (Greedy, KalmanGreedy,
#                 ProportionalNavigation, Lead, Random)
#   rl/         = RL using the Direct/measurement estimator (raw noisy
#                 measurement + finite-difference velocity) -- never ground truth
#   rl_kalman/  = RL using the Kalman-filtered estimator (e_hat, v_hat)
#
# All policies implement:
#   - act(obs: np.ndarray, info: dict) -> np.ndarray
#   - reset() -> None
#
# This allows the same evaluation code to work for all policy types.
# See uav_defend.policies.sanitize.build_policy_info() for the structural
# ground-truth-independence enforcement used by evaluation pipelines.

# Baseline policies
from uav_defend.policies.baseline import (
    GreedyInterceptPolicy,
    KalmanGreedyInterceptPolicy,
    LeadInterceptPolicy,
    ProportionalNavigationPolicy,
    RandomPolicy,
)

# RL policy wrappers
from uav_defend.policies.rl import PPOPolicyWrapper

# RL-Kalman policy wrappers
from uav_defend.policies.rl_kalman import PPOKalmanPolicyWrapper

# Policy registry (factory pattern)
from uav_defend.policies.registry import get_policy, list_policies, get_policy_name

# Ground-truth-independence enforcement helper
from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env

__all__ = [
    # Baseline policies
    "GreedyInterceptPolicy",
    "KalmanGreedyInterceptPolicy",
    "LeadInterceptPolicy",
    "ProportionalNavigationPolicy",
    "RandomPolicy",
    # RL wrapper
    "PPOPolicyWrapper",
    # RL-Kalman wrapper
    "PPOKalmanPolicyWrapper",
    # Registry
    "get_policy",
    "list_policies",
    "get_policy_name",
    # Sanitization
    "build_policy_info",
    "estimator_mode_for_env",
]

