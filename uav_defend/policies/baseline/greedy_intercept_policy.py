"""
Greedy Intercept Policy - Baseline Defender Controller

This is the hand-designed baseline policy used for comparison against
reinforcement learning agents. It implements a simple greedy pursuit
strategy that moves the defender directly toward the estimated/measured
hostile position exposed by the environment as `info['e_hat']`.

GROUND-TRUTH INDEPENDENCE: this policy NEVER reads `info['enemy_pos']` or
`info['enemy_vel']` (ground truth, exposed only for evaluation/debugging).
It pursues whichever legitimate estimate the environment is configured to
produce in `info['e_hat']`:
    - config.use_kalman_tracking=False (canonical registry name "greedy",
      the Direct/measurement track): e_hat mirrors the raw noisy hostile
      position measurement (see SoldierEnv._update_detection). This is NOT
      the true/ground-truth hostile position.
    - config.use_kalman_tracking=True (canonical registry name
      "kalman_greedy", via KalmanGreedyInterceptPolicy): e_hat is the
      Kalman filter's estimated hostile position.
This policy itself does not care which mode is active -- it simply pursues
`info['e_hat']`, whatever the environment provides.

Strategy:
    1. If enemy is detected (info['e_hat'] available): pursue e_hat directly
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
        - When enemy is detected: Move directly toward info['e_hat'] (the
          raw measurement in Direct/measurement mode, or the Kalman
          estimate in Kalman mode -- never ground truth)
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
            obs: Environment observation array of shape (16,).
                 Format: [soldier_x, soldier_y, soldier_z,
                         defender_x, defender_y, defender_z,
                         defender_vx, defender_vy, defender_vz,
                         detected_flag, e_hat(3), v_hat(3)]
                 All values normalized to [-1, 1].
            info: Environment info dict containing:
                 - 'defender_pos': True defender position (unnormalized), shape (3,)
                 - 'e_hat': Hostile position estimate (raw measurement in
                   Direct/measurement mode, or Kalman estimate in Kalman
                   mode; or None if not yet detected), shape (3,)
                 - 'soldier_pos': True soldier position (unnormalized), shape (3,)
                 - 'enemy_detected': Boolean detection flag
        
        Returns:
            action: 3D action vector in [-1, 1]³ representing heading direction.
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
            return np.zeros(3, dtype=np.float32)
        
        # Compute direction from defender to target
        defender = np.asarray(defender_pos, dtype=np.float32)
        direction = target - defender
        dist = np.linalg.norm(direction)
        
        if dist < self.eps:
            # Already at target, no movement needed
            return np.zeros(3, dtype=np.float32)
        
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
