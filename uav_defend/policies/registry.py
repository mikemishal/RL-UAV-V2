"""
Policy Registry - Factory functions for creating policies by name.

Provides a unified interface for instantiating policies without
hardcoding types in evaluation scripts.

Usage:
    from uav_defend.policies.registry import get_policy, list_policies
    
    policy = get_policy("greedy")
    policy = get_policy("ppo", model_path="models/policies/ppo_defender.zip")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Policy name aliases
POLICY_ALIASES = {
    "greedy": "greedy_intercept",
    "greedy_intercept": "greedy_intercept",
    "baseline": "greedy_intercept",
    "random": "random",
    "ppo": "ppo",
    "rl": "ppo",
}


def list_policies() -> list[str]:
    """
    List available policy names.
    
    Returns:
        List of policy names that can be passed to get_policy().
    """
    return ["greedy", "random", "ppo"]


def get_policy(name: str, **kwargs) -> Any:
    """
    Factory function to create a policy by name.
    
    Args:
        name: Policy name (see list_policies() for options).
        **kwargs: Additional arguments passed to the policy constructor.
        
        For "greedy": No additional arguments.
        For "random": Optional `seed` (int).
        For "ppo": Required `model_path` (str), optional `deterministic` (bool).
    
    Returns:
        Policy instance with act(obs, info) and reset() methods.
    
    Raises:
        ValueError: If policy name is unknown.
        FileNotFoundError: If PPO model path doesn't exist.
    
    Examples:
        >>> policy = get_policy("greedy")
        >>> policy = get_policy("random", seed=42)
        >>> policy = get_policy("ppo", model_path="models/policies/ppo_defender.zip")
    """
    # Normalize name
    name_lower = name.lower().strip()
    if name_lower not in POLICY_ALIASES:
        available = ", ".join(list_policies())
        raise ValueError(f"Unknown policy: '{name}'. Available: {available}")
    
    policy_type = POLICY_ALIASES[name_lower]
    
    if policy_type == "greedy_intercept":
        from uav_defend.policies.baseline.greedy_intercept_policy import GreedyInterceptPolicy
        return GreedyInterceptPolicy(**kwargs)
    
    elif policy_type == "random":
        from uav_defend.policies.baseline.random_policy import RandomPolicy
        return RandomPolicy(**kwargs)
    
    elif policy_type == "ppo":
        from uav_defend.policies.rl.ppo_policy_wrapper import PPOPolicyWrapper
        
        model_path = kwargs.pop("model_path", None)
        if model_path is None:
            raise ValueError("PPO policy requires 'model_path' argument")
        
        deterministic = kwargs.pop("deterministic", True)
        return PPOPolicyWrapper.load(model_path, deterministic=deterministic)
    
    else:
        raise ValueError(f"Policy type not implemented: {policy_type}")


def get_policy_name(policy) -> str:
    """
    Get the canonical name of a policy instance.
    
    Args:
        policy: A policy instance.
    
    Returns:
        String name of the policy type.
    """
    class_name = type(policy).__name__
    
    name_map = {
        "GreedyInterceptPolicy": "greedy",
        "RandomPolicy": "random",
        "PPOPolicyWrapper": "ppo",
    }
    
    return name_map.get(class_name, class_name.lower())
