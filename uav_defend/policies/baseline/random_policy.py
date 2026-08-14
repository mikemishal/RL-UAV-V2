"""
Random Policy - Baseline for comparison.

A random policy that samples actions uniformly from the action space.
Used as a lower-bound baseline for policy comparison.
"""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    """
    Random baseline policy for UAV defense.
    
    Samples actions uniformly from [-1, 1]² at each step.
    Provides a lower-bound baseline for policy comparison.
    
    Example:
        >>> policy = RandomPolicy()
        >>> obs, info = env.reset()
        >>> action = policy.act(obs, info)
    """
    
    def __init__(self, seed: int | None = None):
        """
        Initialize the random policy.
        
        Args:
            seed: Optional random seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed)
    
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        Sample a random action.
        
        Args:
            obs: Environment observation (unused).
            info: Environment info dict (unused).
        
        Returns:
            action: Random 2D action vector in [-1, 1]².
        """
        return self._rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
    
    def reset(self) -> None:
        """Reset internal state (no-op for random policy)."""
        pass
    
    def __repr__(self) -> str:
        return "RandomPolicy()"
