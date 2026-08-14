"""
PPO Policy Wrapper - Unified interface for trained Stable-Baselines3 PPO models.

=============================================================================
PURPOSE: UNIFIED EVALUATION INTERFACE FOR RL AND BASELINE POLICIES
=============================================================================

This wrapper allows trained RL policies (PPO) to be evaluated using the exact
same evaluation pipeline as scripted baseline policies (e.g., GreedyInterceptPolicy).

The key insight is that both baseline and RL policies are just functions that
map (observation, info) -> action. The wrapper adapts the SB3 model.predict()
interface to match the baseline policy.act(obs, info) interface.

Example - Same evaluation code for baseline and RL:
    
    from experiments.eval_utils import evaluate_policy
    from uav_defend.envs import SoldierEnv
    from uav_defend.policies.baseline import GreedyInterceptPolicy
    from uav_defend.policies.rl import PPOPolicyWrapper
    
    env_factory = lambda: SoldierEnv()
    seeds = range(1000)
    
    # Evaluate baseline policy
    baseline_policy = GreedyInterceptPolicy()
    df_baseline, summary_baseline, _ = evaluate_policy(
        env_factory=env_factory,
        policy=baseline_policy,
        seeds=seeds,
    )
    
    # Evaluate RL policy with IDENTICAL interface
    ppo_policy = PPOPolicyWrapper.load("models/policies/ppo_defender.zip")
    df_ppo, summary_ppo, _ = evaluate_policy(
        env_factory=env_factory,
        policy=ppo_policy,
        seeds=seeds,
    )
    
    # Compare results
    print(f"Baseline success: {summary_baseline['success_rate']:.1%}")
    print(f"PPO success: {summary_ppo['success_rate']:.1%}")

=============================================================================
INTERFACE CONTRACT
=============================================================================

All policies (baseline and RL) must implement:

    class Policy(Protocol):
        def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
            '''Return action given observation and info.'''
            ...
        
        def reset(self) -> None:
            '''Reset any internal state (called at episode start).'''
            ...

This wrapper adapts SB3's model.predict(obs) to match this interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stable_baselines3 import PPO


class PPOPolicyWrapper:
    """
    Wrapper for Stable-Baselines3 PPO models with unified evaluation interface.
    
    This class wraps a trained SB3 PPO model and exposes the same `act(obs, info)`
    interface as scripted baseline policies. This allows RL policies to be
    evaluated using the same evaluation pipeline (experiments/eval_utils.py)
    as the baseline policies.
    
    Key Features:
        - Matches baseline policy interface: act(obs, info) -> action
        - Supports deterministic (evaluation) and stochastic (exploration) modes
        - Lazy import of SB3 (only imported when load() is called)
        - Works with the unified policy registry
    
    Attributes:
        model: The loaded Stable-Baselines3 PPO model.
        deterministic: If True, use deterministic actions (no exploration noise).
                      Set to True for evaluation, False if you need stochasticity.
    
    Example - Loading and using:
        >>> from uav_defend.policies.rl import PPOPolicyWrapper
        >>> 
        >>> # Load trained model
        >>> policy = PPOPolicyWrapper.load("models/policies/ppo_defender.zip")
        >>> 
        >>> # Use with environment (same as baseline!)
        >>> obs, info = env.reset(seed=42)
        >>> while not done:
        ...     action = policy.act(obs, info)  # Same interface as GreedyInterceptPolicy
        ...     obs, reward, done, _, info = env.step(action)
    
    Example - Deterministic vs Stochastic:
        >>> # Deterministic evaluation (default) - same action for same state
        >>> policy = PPOPolicyWrapper.load(path, deterministic=True)
        >>> 
        >>> # Stochastic inference - samples from policy distribution
        >>> policy = PPOPolicyWrapper.load(path, deterministic=False)
    """
    
    def __init__(self, model: "PPO", deterministic: bool = True):
        """
        Initialize the PPO policy wrapper.
        
        Usually you should use PPOPolicyWrapper.load() instead of calling
        this constructor directly.
        
        Args:
            model: A loaded Stable-Baselines3 PPO model instance.
            deterministic: If True, use deterministic actions during inference.
                          - True: Returns the mean of the policy distribution.
                                  Best for evaluation and comparison.
                          - False: Samples from the policy distribution.
                                   Useful for ensembling or uncertainty estimation.
        """
        self.model = model
        self.deterministic = deterministic
    
    @classmethod
    def load(
        cls,
        path: str | Path,
        deterministic: bool = True,
        device: str = "auto",
    ) -> "PPOPolicyWrapper":
        """
        Load a trained PPO model from a saved checkpoint.
        
        This is the recommended way to create a PPOPolicyWrapper instance.
        
        Args:
            path: Path to the saved model file (.zip format).
                  Example: "models/policies/ppo_defender.zip"
            deterministic: If True (default), use deterministic actions.
                          Set to True for fair comparison with baseline.
            device: Device to load the model on ("auto", "cpu", "cuda").
                   Default "auto" uses GPU if available.
        
        Returns:
            PPOPolicyWrapper instance wrapping the loaded model.
        
        Raises:
            ImportError: If stable-baselines3 is not installed.
            FileNotFoundError: If the model file doesn't exist.
        
        Example:
            >>> policy = PPOPolicyWrapper.load("models/policies/ppo_v1.zip")
            >>> action = policy.act(obs, info)
        """
        # Lazy import to avoid requiring SB3 for baseline-only usage
        try:
            from stable_baselines3 import PPO
        except ImportError as e:
            raise ImportError(
                "stable-baselines3 is required for PPOPolicyWrapper. "
                "Install with: pip install stable-baselines3"
            ) from e
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        
        model = PPO.load(str(path), device=device)
        return cls(model, deterministic=deterministic)
    
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        Get action from the PPO model.
        
        This method matches the baseline policy interface, allowing RL policies
        to be used interchangeably with scripted policies in the evaluation
        pipeline.
        
        Args:
            obs: Environment observation array, shape (9,) for SoldierEnv.
                 Expected to be normalized to [-1, 1].
            info: Environment info dictionary. Not used by PPO but included
                  for interface compatibility with baseline policies.
        
        Returns:
            action: 2D action vector, shape (2,), values in [-1, 1].
                   Represents the defender heading direction.
        
        Note:
            The info dict is provided for interface compatibility but ignored
            by RL policies. Baseline policies may use info for features like
            e_hat (Kalman estimate) that are already encoded in obs for RL.
        """
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)
    
    def reset(self) -> None:
        """
        Reset internal state at episode start.
        
        PPO is a stateless policy (no RNN hidden state), so this is a no-op.
        Included for interface compatibility with the Policy protocol.
        
        For recurrent policies (LSTM-PPO), this would reset the hidden state.
        """
        pass
    
    @property
    def name(self) -> str:
        """Return a human-readable policy name."""
        return "ppo"
    
    def __repr__(self) -> str:
        mode = "deterministic" if self.deterministic else "stochastic"
        return f"PPOPolicyWrapper({mode})"
