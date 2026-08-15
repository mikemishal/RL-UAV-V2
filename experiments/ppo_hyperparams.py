"""
Centralized PPO training hyperparameters, shared by BOTH experiment tracks.

This module is the SINGLE SOURCE OF TRUTH for PPO hyperparameters so that
experiments/rl/train_ppo.py (Direct/measurement track) and
experiments/rl_kalman/train_ppo_kalman.py (Kalman track) can never silently
drift apart. Both scripts import from here rather than redefining these
constants locally.

Any script may still override individual values via its own CLI flags; the
values below are only the shared DEFAULTS.
"""

from __future__ import annotations

# =============================================================================
# PPO ALGORITHM HYPERPARAMETERS
# =============================================================================
TOTAL_TIMESTEPS = 200_000      # Total training timesteps
LEARNING_RATE = 3e-4           # PPO learning rate (default: 3e-4)
GAMMA = 0.99                   # Discount factor
BATCH_SIZE = 64                # Minibatch size for PPO updates
N_STEPS = 2048                 # Steps per rollout (before update)
N_EPOCHS = 10                  # Number of epochs per update
CLIP_RANGE = 0.2               # PPO clipping parameter
ENT_COEF = 0.01                # Entropy coefficient (encourages exploration)
VF_COEF = 0.5                  # Value function loss coefficient
MAX_GRAD_NORM = 0.5            # Max gradient norm for clipping
GAE_LAMBDA = 0.95              # GAE lambda for advantage estimation

# =============================================================================
# EVALUATION / CHECKPOINTING (shared cadence and model-selection sample size)
# =============================================================================
# N_EVAL_EPISODES was increased from 20 -> 100 (2026-08, multi-seed training
# protocol): best-model selection based on 20 Bernoulli success/failure
# trials is too noisy (a true 60% success-rate policy has a binomial
# standard error of ~11pp at n=20, vs ~5pp at n=100) to reliably pick the
# best checkpoint across a 200k-timestep run. EVAL_FREQ was reduced from
# 10_000 -> 25_000 so the larger per-evaluation episode budget does not
# excessively inflate total wall-clock cost (100 episodes every 25k steps
# gives 8 evaluations per 200k-timestep run, vs 20 evaluations previously).
# This is a symmetric reliability improvement applied identically to BOTH
# the Direct and Kalman tracks -- NOT controller-specific tuning.
CHECKPOINT_FREQ = 50_000       # Save checkpoint every N timesteps
EVAL_FREQ = 25_000             # Evaluate policy every N timesteps
N_EVAL_EPISODES = 100          # Episodes per evaluation (was 20)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42                      # Random seed for reproducibility
