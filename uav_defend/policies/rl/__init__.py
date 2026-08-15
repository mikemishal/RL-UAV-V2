"""
RL Policy Wrappers.

This module provides wrapper classes for trained RL models (e.g., PPO, SAC, TD3)
that expose the same interface as scripted baseline policies.

The unified interface allows both baseline and RL policies to be evaluated
using the same evaluation pipeline:

    from experiments.eval_utils import evaluate_policy
    from uav_defend.policies.rl import PPOPolicyWrapper
    from uav_defend.policies.baseline import GreedyInterceptPolicy
    from uav_defend.envs import SoldierEnv
    from uav_defend.config import EnvConfig
    
    # Evaluate baseline (explicit Direct/measurement-track configuration --
    # never rely on the environment's default use_kalman_tracking value)
    df_baseline, summary_baseline, _ = evaluate_policy(
        env_factory=lambda: SoldierEnv(config=EnvConfig(use_kalman_tracking=False)),
        policy=GreedyInterceptPolicy(),
        seeds=range(1000),
    )
    
    # Evaluate RL with SAME interface
    df_ppo, summary_ppo, _ = evaluate_policy(
        env_factory=lambda: SoldierEnv(config=EnvConfig(use_kalman_tracking=False)),
        policy=PPOPolicyWrapper.load("models/policies/ppo_defender.zip"),
        seeds=range(1000),
    )
"""

from uav_defend.policies.rl.ppo_policy_wrapper import PPOPolicyWrapper

__all__ = ["PPOPolicyWrapper"]
