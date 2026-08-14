"""
RL Policy Wrappers.

This module provides wrapper classes for trained RL models (e.g., PPO, SAC, TD3)
that expose the same interface as scripted baseline policies.

The unified interface allows both baseline and RL policies to be evaluated
using the same evaluation pipeline:

    from experiments.eval_utils import evaluate_policy
    from uav_defend.policies.rl import PPOPolicyWrapper
    from uav_defend.policies.baseline import GreedyInterceptPolicy
    
    # Evaluate baseline
    df_baseline, summary_baseline, _ = evaluate_policy(
        env_factory=lambda: SoldierEnv(),
        policy=GreedyInterceptPolicy(),
        seeds=range(1000),
    )
    
    # Evaluate RL with SAME interface
    df_ppo, summary_ppo, _ = evaluate_policy(
        env_factory=lambda: SoldierEnv(),
        policy=PPOPolicyWrapper.load("models/policies/ppo_defender.zip"),
        seeds=range(1000),
    )
"""

from uav_defend.policies.rl.ppo_policy_wrapper import PPOPolicyWrapper

__all__ = ["PPOPolicyWrapper"]
