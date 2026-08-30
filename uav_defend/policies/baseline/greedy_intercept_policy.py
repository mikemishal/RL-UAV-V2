"""
Greedy Intercept Policy - Baseline Defender Controller

This is the hand-designed baseline policy used for comparison against
reinforcement learning agents. It implements a simple greedy pursuit
strategy that moves the defender directly toward the legitimately available
hostile-position estimate for the Direct/measurement track:
`info['enemy_measurement']` (the raw noisy hostile-position measurement).

GROUND-TRUTH INDEPENDENCE: this policy NEVER reads `info['enemy_pos']` or
`info['enemy_vel']` (ground truth, exposed only for evaluation/debugging).

INPUT CONTRACT (corrected -- see the "Direct Greedy measurement-contract
bug" fix): this policy is the Direct/measurement-track baseline (canonical
registry name "greedy") and after detection pursues
`info['enemy_measurement']`, which is exactly what
`uav_defend.policies.sanitize.build_policy_info` supplies for the
"measurement" estimator mode. It does NOT read `info['e_hat']` --
`e_hat` is a Kalman-track-only field (see KalmanGreedyInterceptPolicy,
canonical registry name "kalman_greedy", which pursues `info['e_hat']`
instead). A previous version of this policy incorrectly read `info['e_hat']`
unconditionally; under the sanitized measurement-track pipeline that field
is never present, so the policy silently remained in escort mode for the
entire episode even after detection. This was a software input-contract
bug, not an intended design choice -- it is fixed here by reading
`enemy_measurement` (falling back to `e_hat` only if `enemy_measurement` is
absent, e.g. if this policy is ever run directly against the Kalman track),
and by raising an explicit error (rather than silently escorting) if
`enemy_detected` is True but NEITHER field is available.

Strategy:
    1. If enemy is NOT detected: escort the soldier (move toward soldier_pos).
    2. If enemy IS detected: pure pursuit of enemy_measurement (or e_hat as
       a fallback -- see above); raise ValueError if neither is present.

This policy serves as a performance baseline to evaluate whether learned
policies provide meaningful improvement over hand-crafted heuristics.
"""

from __future__ import annotations

import numpy as np


class GreedyInterceptPolicy:
    """
    Greedy baseline policy for UAV defense (Direct/measurement track).
    
    A stateless, hand-designed controller that serves as the baseline
    for comparison against reinforcement learning policies.
    
    Behavior:
        - When enemy is detected: Move directly toward info['enemy_measurement']
          (the raw noisy hostile-position measurement -- never ground truth).
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
                 - 'enemy_measurement': Raw noisy hostile-position measurement
                   (Direct/measurement track), or None if not yet detected, shape (3,)
                 - 'e_hat': Kalman-filtered hostile-position estimate -- used
                   ONLY as a fallback if 'enemy_measurement' is absent (e.g.
                   if this policy is run against a Kalman-track info dict).
                 - 'soldier_pos': True soldier position (unnormalized), shape (3,)
                 - 'enemy_detected': Boolean detection flag
        
        Returns:
            action: 3D action vector in [-1, 1]³ representing heading direction.
                   Will be normalized by the environment.
        
        Policy Logic:
            1. If enemy_detected is False: escort (pursue soldier_pos).
            2. If enemy_detected is True: pursue enemy_measurement; if that
               is absent, fall back to e_hat; if NEITHER is available, raise
               ValueError (an explicit input-contract failure) rather than
               silently continuing to escort -- see the module docstring's
               "Direct Greedy measurement-contract bug" note.
        
        Raises:
            ValueError: If enemy_detected is True but neither
                'enemy_measurement' nor 'e_hat' is present/non-None in info.
        """
        defender_pos = info.get('defender_pos')
        soldier_pos = info.get('soldier_pos')
        enemy_detected = bool(info.get('enemy_detected', False))
        
        if not enemy_detected:
            # Enemy not detected: escort soldier
            target = np.asarray(soldier_pos, dtype=np.float32)
        else:
            enemy_measurement = info.get('enemy_measurement')
            e_hat = info.get('e_hat')
            if enemy_measurement is not None:
                # Direct/measurement track: pursue the raw noisy measurement
                target = np.asarray(enemy_measurement, dtype=np.float32)
            elif e_hat is not None:
                # Fallback for a Kalman-track info dict (canonical use is
                # via KalmanGreedyInterceptPolicy, not this class)
                target = np.asarray(e_hat, dtype=np.float32)
            else:
                raise ValueError(
                    "GreedyInterceptPolicy: enemy_detected is True but neither "
                    "'enemy_measurement' nor 'e_hat' is available in info -- "
                    "refusing to silently fall back to escort mode (this indicates "
                    "a policy/sanitizer input-contract mismatch that must be fixed, "
                    "not hidden)."
                )
        
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
