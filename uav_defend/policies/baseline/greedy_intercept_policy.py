"""
Greedy Intercept Policy - Baseline Defender Controller

This is the hand-designed baseline policy used for comparison against
reinforcement learning agents. It implements a simple greedy pursuit
strategy that moves the defender directly toward the estimated enemy position.

Strategy:
    1. If enemy is detected (Kalman estimate available): pursue enemy directly
    2. If enemy is not detected: stay with soldier (escort behavior)

This policy serves as a performance baseline to evaluate whether learned
policies provide meaningful improvement over hand-crafted heuristics.
"""

from __future__ import annotations

import numpy as np


class GreedyInterceptPolicy:
    """
    Greedy baseline policy for UAV defense.
    
    A stateless, hand-designed controller that serves as the baseline
    for comparison against reinforcement learning policies.
    
    Behavior:
        - When enemy is detected: Move directly toward estimated enemy position (e_hat)
        - When enemy is not detected: Stay with soldier (escort mode)
    
    This represents the simplest reasonable interception strategy:
    pure pursuit without prediction or planning.
    
    Attributes:
        eps: Small threshold for numerical stability in normalization.
    
    Example:
        >>> policy = GreedyInterceptPolicy()
        >>> obs, info = env.reset()
        >>> action = policy.act(obs, info)
        >>> obs, reward, done, truncated, info = env.step(action)
    """
    
    def __init__(self, eps: float = 1e-8):
        """
        Initialize the greedy intercept policy.
        
        Args:
            eps: Small threshold for numerical stability when normalizing
                 direction vectors. Default: 1e-8.
        """
        self.eps = eps
    
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        Compute the greedy intercept action based on current state.
        
        This is the main policy interface compatible with the SoldierEnv.
        
        Args:
            obs: Environment observation array of shape (9,).
                 Format: [soldier_x, soldier_y, defender_x, defender_y, 
                         detected_flag, e_hat_x, e_hat_y, v_hat_x, v_hat_y]
                 All values normalized to [-1, 1].
            info: Environment info dict containing:
                 - 'defender_pos': True defender position (unnormalized)
                 - 'e_hat': Estimated enemy position from Kalman filter (or None)
                 - 'soldier_pos': True soldier position (unnormalized)
                 - 'enemy_detected': Boolean detection flag
        
        Returns:
            action: 2D action vector in [-1, 1]² representing heading direction.
                   Will be normalized by the environment.
        
        Policy Logic:
            1. If e_hat is available in info (enemy detected):
               Compute unit vector from defender to e_hat (pursuit)
            2. If e_hat is None (enemy not detected):
               Compute unit vector from defender to soldier (escort)
        """
        defender_pos = info.get('defender_pos')
        e_hat = info.get('e_hat')
        soldier_pos = info.get('soldier_pos')
        
        if e_hat is not None:
            # Enemy detected: pursue estimated enemy position
            target = np.asarray(e_hat, dtype=np.float32)
        elif soldier_pos is not None:
            # Enemy not detected: escort soldier
            target = np.asarray(soldier_pos, dtype=np.float32)
        else:
            # Fallback: no movement
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # Compute direction from defender to target
        defender = np.asarray(defender_pos, dtype=np.float32)
        direction = target - defender
        dist = np.linalg.norm(direction)
        
        if dist < self.eps:
            # Already at target, no movement needed
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # Normalize to unit vector
        action = direction / dist
        
        return action.astype(np.float32)
    
    def reset(self) -> None:
        """
        Reset any internal state.
        
        This policy is stateless, so reset does nothing.
        Provided for interface compatibility with stateful policies.
        """
        pass
    
    def __repr__(self) -> str:
        return f"GreedyInterceptPolicy(eps={self.eps})"
