# uav_defend/policies - Defender decision logic
#
# This module provides both baseline (scripted) and RL (trained) policies
# with a unified interface for evaluation.
#
# Policy Organization:
# --------------------
#   baseline/   = Scripted hand-designed controllers (GreedyIntercept, Random)
#   rl/         = RL with direct environment observations (true/noisy enemy state)
#   rl_kalman/  = RL using Kalman-estimated attacker state (e_hat, v_hat)
#
# All policies implement:
#   - act(obs: np.ndarray, info: dict) -> np.ndarray
#   - reset() -> None
#
# This allows the same evaluation code to work for all policy types.

# Baseline policies
from uav_defend.policies.baseline import GreedyInterceptPolicy, RandomPolicy

# RL policy wrappers
from uav_defend.policies.rl import PPOPolicyWrapper

# RL-Kalman policy wrappers
from uav_defend.policies.rl_kalman import PPOKalmanPolicyWrapper

# Policy registry (factory pattern)
from uav_defend.policies.registry import get_policy, list_policies, get_policy_name

__all__ = [
    # Baseline policies
    "GreedyInterceptPolicy",
    "RandomPolicy",
    # RL wrapper
    "PPOPolicyWrapper",
    # RL-Kalman wrapper
    "PPOKalmanPolicyWrapper",
    # Registry
    "get_policy",
    "list_policies",
    "get_policy_name",
]
