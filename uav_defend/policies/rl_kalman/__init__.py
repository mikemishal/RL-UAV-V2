"""
RL-with-Kalman Policy Wrappers.

=============================================================================
EXPERIMENT TRACK: RL USING KALMAN-ESTIMATED ATTACKER STATE
=============================================================================

This module provides policy wrappers for RL models trained on Kalman-filtered
enemy state estimates rather than direct (true or noisy) observations.

Policy Organization:
--------------------
    uav_defend/policies/
    ├── baseline/       # Scripted hand-designed controllers (GreedyIntercept, Random)
    ├── rl/             # RL with direct environment observations (true/noisy enemy state)
    └── rl_kalman/      # RL using Kalman-estimated attacker state (e_hat, v_hat)

Key Difference from rl/:
------------------------
    - rl/ policies: Trained on raw observations from environment
    - rl_kalman/ policies: Trained on Kalman-filtered state estimates
    
    The Kalman filter provides:
        - Smoothed position estimates (reduces measurement noise)
        - Velocity estimates (not directly observable)
        - Prediction capability (lead_time extrapolation)

Why Separate Track:
-------------------
    1. Different observation semantics (estimated vs direct)
    2. Different training considerations (filter convergence time)
    3. Fair comparison requires consistent tracking for train/eval
    4. Enables ablation: RL alone vs RL + Kalman

Usage:
------
    from uav_defend.envs import SoldierEnv
    from uav_defend.config import EnvConfig
    from uav_defend.policies.rl_kalman import PPOKalmanPolicyWrapper
    
    # Ensure environment uses Kalman tracking
    config = EnvConfig(use_kalman_tracking=True, process_var=1.0, measurement_var=0.5)
    env = SoldierEnv(config=config)
    
    # Load RL policy trained with Kalman observations
    policy = PPOKalmanPolicyWrapper.load("models/rl_kalman/ppo_kalman.zip")
    
    obs, info = env.reset(seed=42)
    while not done:
        action = policy.act(obs, info)  # Uses e_hat, v_hat from Kalman filter
        obs, reward, done, _, info = env.step(action)

See Also:
---------
    - uav_defend.tracking.EnemyKalmanFilter: The Kalman filter implementation
    - uav_defend.config.env_config.EnvConfig: Kalman tracking configuration
    - experiments/rl_kalman/: Training and evaluation scripts for this track
"""

from uav_defend.policies.rl_kalman.ppo_kalman_policy_wrapper import PPOKalmanPolicyWrapper

__all__ = ["PPOKalmanPolicyWrapper"]
