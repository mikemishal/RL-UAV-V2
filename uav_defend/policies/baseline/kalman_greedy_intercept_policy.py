"""
Kalman Greedy Intercept Policy - Kalman-Enhanced Hand-Designed Baseline

This policy is a hand-designed baseline that combines Kalman-filtered state
estimation with simple greedy pursuit. It is designed to run alongside the
environment's Kalman tracker (use_kalman_tracking=True) and consumes the
filtered enemy position estimate (e_hat) provided in the info dictionary.

Purpose:
    This baseline isolates the effect of state estimation without reinforcement
    learning. By comparing this policy against:
      - GreedyInterceptPolicy  (greedy on true state, no estimation)
      - PPO Direct RL          (learned policy on true state)
      - PPO RL-Kalman          (learned policy on Kalman state)
    we can independently measure the value of Kalman filtering and the value
    of learned control, separated from each other.

Strategy:
    1. If a Kalman-estimated enemy position (e_hat) is available in info:
       pursue e_hat directly (greedy pursuit of filtered estimate).
    2. If no estimate is available (enemy not yet detected):
       escort the soldier (defensive positioning until detection occurs).

Design Decision — Stateless:
    This policy maintains no internal state. All estimation is delegated to
    the environment's EnemyKalmanFilter. This makes the policy straightforward
    to evaluate and removes tracking as a per-policy variable.

Usage:
    This policy must be used with an environment configured for Kalman tracking:
        env = SoldierEnv(EnvConfig(use_kalman_tracking=True))
    Using it with use_kalman_tracking=False will result in it receiving raw
    noisy enemy measurements in e_hat (without Kalman smoothing).
"""

from __future__ import annotations

import numpy as np


class KalmanGreedyInterceptPolicy:
    """
    Kalman-enhanced greedy baseline policy for UAV defense.

    A stateless, hand-designed controller that applies simple greedy pursuit
    to the Kalman-filtered enemy state estimate provided by the environment.
    This is the fourth comparison method in the research, alongside:

        - GreedyInterceptPolicy  : greedy pursuit with true enemy state
        - PPO Direct RL          : learned policy with true enemy state
        - PPO RL-Kalman          : learned policy with Kalman-filtered state

    By pairing Kalman estimation with a hand-designed (non-RL) controller,
    this baseline quantifies how much of the RL-Kalman performance gain (or
    loss) comes from the estimation step vs. the learned control.

    Behavior:
        - When e_hat is available (enemy detected, Kalman estimate active):
          Move directly toward the Kalman-filtered enemy position e_hat.
        - When e_hat is None (enemy not yet detected):
          Escort the soldier (move toward soldier_pos).

    Attributes:
        eps: Small threshold for numerical stability in vector normalization.

    Example:
        >>> env = SoldierEnv(EnvConfig(use_kalman_tracking=True))
        >>> policy = KalmanGreedyInterceptPolicy()
        >>> obs, info = env.reset()
        >>> action = policy.act(obs, info)
        >>> obs, reward, done, truncated, info = env.step(action)
    """

    def __init__(self, eps: float = 1e-8):
        """
        Initialize the Kalman greedy intercept policy.

        Args:
            eps: Small threshold for numerical stability when normalizing
                 direction vectors. Default: 1e-8.
        """
        self.eps = eps

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        Compute the greedy intercept action using the Kalman-filtered estimate.

        This is the main policy interface compatible with SoldierEnv.

        Args:
            obs: Environment observation array of shape (16,).
                 Format: [soldier(3), defender(3), defender_vel(3),
                          detected_flag, e_hat(3), v_hat(3)]
                 All values normalized to [-1, 1]. Not used directly; raw
                 positions are read from info for clarity and precision.
            info: Environment info dict containing:
                 - 'defender_pos': Unnormalized defender position (np.ndarray), shape (3,)
                 - 'e_hat': Kalman-filtered enemy position estimate, or None
                   if the enemy has not been detected yet (np.ndarray | None), shape (3,)
                 - 'soldier_pos': Unnormalized soldier position (np.ndarray), shape (3,)
                 - 'enemy_detected': Boolean detection flag

        Returns:
            action: 3D action vector in [-1, 1]^3 representing heading direction.
                   The environment normalizes the magnitude; only direction matters.

        Policy Logic:
            1. If e_hat is available in info (enemy detected, Kalman active):
               Compute unit vector from defender to e_hat and return it.
            2. If e_hat is None (enemy not yet detected):
               Compute unit vector from defender to soldier_pos (escort mode).
            3. If neither target is resolvable (degenerate state):
               Return zero vector (no movement).
        """
        defender_pos = info.get("defender_pos")
        e_hat = info.get("e_hat")
        soldier_pos = info.get("soldier_pos")

        if e_hat is not None:
            # Enemy detected: pursue Kalman-estimated enemy position
            target = np.asarray(e_hat, dtype=np.float32)
        elif soldier_pos is not None:
            # Enemy not yet detected: escort soldier until detection occurs
            target = np.asarray(soldier_pos, dtype=np.float32)
        else:
            # Degenerate state: no useful target information available
            return np.zeros(3, dtype=np.float32)

        # Compute direction vector from defender to target
        defender = np.asarray(defender_pos, dtype=np.float32)
        direction = target - defender
        dist = np.linalg.norm(direction)

        if dist < self.eps:
            # Defender is already at target; no movement needed
            return np.zeros(3, dtype=np.float32)

        # Return normalized unit vector (environment scales by defender speed)
        return (direction / dist).astype(np.float32)

    def reset(self) -> None:
        """
        Reset any internal state.

        This policy is stateless, so reset is a no-op.
        Provided for interface compatibility with stateful policies.
        """
        pass

    def __repr__(self) -> str:
        return f"KalmanGreedyInterceptPolicy(eps={self.eps})"
