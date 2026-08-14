"""
PPO-Kalman Policy Wrapper - RL policy using Kalman-estimated attacker state.

=============================================================================
PURPOSE: UNIFIED EVALUATION INTERFACE FOR RL-WITH-KALMAN POLICIES
=============================================================================

This wrapper adapts trained PPO models that were trained using Kalman-filtered
observations (e_hat, v_hat) rather than direct environment observations.

The key difference from PPOPolicyWrapper (in rl/):
    - rl/PPOPolicyWrapper: For models trained on raw observations
    - rl_kalman/PPOKalmanPolicyWrapper: For models trained on Kalman estimates

Why This Matters:
-----------------
When use_kalman_tracking=True in the environment:
    - obs[5:7] = e_hat (Kalman estimated enemy position)
    - obs[7:9] = v_hat (Kalman estimated enemy velocity)
    
This provides the policy with:
    1. Smoothed position estimates (noise-filtered)
    2. Velocity estimates (not directly observable)
    3. Consistent observation semantics between train/eval

Example - Training vs Evaluation Consistency:
----------------------------------------------
    # Training: Environment configured with Kalman tracking
    config = EnvConfig(use_kalman_tracking=True)
    train_env = SoldierEnv(config=config)
    model = PPO("MlpPolicy", train_env, ...)
    model.learn(total_timesteps=1_000_000)
    model.save("models/rl_kalman/ppo_kalman.zip")
    
    # Evaluation: MUST use same Kalman configuration
    eval_config = EnvConfig(use_kalman_tracking=True)  # Match training!
    eval_env = SoldierEnv(config=eval_config)
    policy = PPOKalmanPolicyWrapper.load("models/rl_kalman/ppo_kalman.zip")
    
    # Same evaluation pipeline as baseline and rl/ policies
    df, summary, _ = evaluate_policy(
        env_factory=lambda: SoldierEnv(config=eval_config),
        policy=policy,
        seeds=range(1000),
    )

=============================================================================
INTERFACE CONTRACT
=============================================================================

Implements the standard policy interface:

    class Policy(Protocol):
        def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
            '''Return action given observation and info.'''
            ...
        
        def reset(self) -> None:
            '''Reset any internal state (called at episode start).'''
            ...

This is the same interface as:
    - baseline/GreedyInterceptPolicy
    - baseline/RandomPolicy
    - rl/PPOPolicyWrapper
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stable_baselines3 import PPO


class PPOKalmanPolicyWrapper:
    """
    Wrapper for PPO models trained with Kalman-filtered observations.
    
    This class is functionally identical to PPOPolicyWrapper, but serves
    as a semantic marker for policies in the RL-with-Kalman experiment track.
    
    The wrapper ensures:
        1. Consistent interface with baseline and rl/ policies
        2. Clear separation of experiment tracks
        3. Documentation of observation semantics (Kalman estimates)
    
    Observation Format (when use_kalman_tracking=True):
    ---------------------------------------------------
        obs = [soldier_x, soldier_y,    # Soldier position (normalized)
               defender_x, defender_y,  # Defender position (normalized)
               detected_flag,           # 0.0 or 1.0
               e_hat_x, e_hat_y,        # Kalman estimated enemy position
               v_hat_x, v_hat_y]        # Kalman estimated enemy velocity
               
        After detection:
            - e_hat: Smoothed position from Kalman filter
            - v_hat: Estimated velocity from Kalman filter
        Before detection:
            - e_hat, v_hat = [0, 0, 0, 0]
    
    Attributes:
        model: The loaded Stable-Baselines3 PPO model.
        deterministic: If True, use deterministic actions (no exploration noise).
    
    Example:
        >>> from uav_defend.policies.rl_kalman import PPOKalmanPolicyWrapper
        >>> from uav_defend.envs import SoldierEnv
        >>> from uav_defend.config import EnvConfig
        >>> 
        >>> # Load policy trained with Kalman tracking
        >>> policy = PPOKalmanPolicyWrapper.load("models/rl_kalman/ppo_kalman.zip")
        >>> 
        >>> # Environment MUST have Kalman tracking enabled
        >>> config = EnvConfig(use_kalman_tracking=True)
        >>> env = SoldierEnv(config=config)
        >>> 
        >>> obs, info = env.reset(seed=42)
        >>> while not done:
        ...     action = policy.act(obs, info)
        ...     obs, reward, done, _, info = env.step(action)
    """
    
    def __init__(self, model: "PPO", deterministic: bool = True):
        """
        Initialize the PPO-Kalman policy wrapper.
        
        Usually you should use PPOKalmanPolicyWrapper.load() instead of calling
        this constructor directly.
        
        Args:
            model: A loaded Stable-Baselines3 PPO model instance.
            deterministic: If True, use deterministic actions during inference.
        """
        self.model = model
        self.deterministic = deterministic
    
    @classmethod
    def load(
        cls,
        path: str | Path,
        deterministic: bool = True,
        device: str = "auto",
    ) -> "PPOKalmanPolicyWrapper":
        """
        Load a trained PPO-Kalman model from a saved checkpoint.
        
        Args:
            path: Path to the saved model file (.zip format).
            deterministic: If True (default), use deterministic actions.
            device: Device to load the model on ("auto", "cpu", "cuda").
        
        Returns:
            PPOKalmanPolicyWrapper instance wrapping the loaded model.
        
        Raises:
            ImportError: If stable-baselines3 is not installed.
            FileNotFoundError: If the model file doesn't exist.
        
        Note:
            The environment used for evaluation MUST have use_kalman_tracking=True
            to match the observation format the model was trained on.
        """
        # Lazy import to avoid requiring SB3 for baseline-only usage
        from stable_baselines3 import PPO
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model = PPO.load(str(path), device=device)
        return cls(model=model, deterministic=deterministic)
    
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        Select an action given the current observation.
        
        This method matches the baseline policy interface, allowing RL-Kalman
        policies to be evaluated with the same code as scripted baselines.
        
        The observation should come from an environment with use_kalman_tracking=True,
        where obs[5:9] contains Kalman estimates (e_hat, v_hat) rather than
        direct enemy state.
        
        Args:
            obs: Normalized observation array from SoldierEnv.
                 Shape: (9,), values in [-1, 1].
                 Expected format (with Kalman tracking):
                   [soldier_x, soldier_y, defender_x, defender_y,
                    detected_flag, e_hat_x, e_hat_y, v_hat_x, v_hat_y]
            info: Additional info dict from environment.
                  Contains: e_hat, v_hat, tracking_error (when detected)
        
        Returns:
            Action array of shape (2,), values in [-1, 1].
            Represents 2D direction vector for defender movement.
        """
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)
    
    def reset(self) -> None:
        """
        Reset any internal state (called at episode start).
        
        For stateless PPO policies, this is a no-op. Included for
        interface compatibility with stateful policies.
        """
        pass
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"PPOKalmanPolicyWrapper(deterministic={self.deterministic})"
