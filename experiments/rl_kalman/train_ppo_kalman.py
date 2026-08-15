"""
PPO Training Script for UAV Defender with Kalman Tracking.

=============================================================================
EXPERIMENT TRACK: RL WITH KALMAN-ESTIMATED ENEMY STATE
=============================================================================

This script trains a PPO agent on SoldierEnv with Kalman filtering enabled.
The key difference from experiments/rl/train_ppo.py:

    - Environment uses use_kalman_tracking=True
    - Observations contain Kalman estimates (e_hat, v_hat) instead of raw state
    - Policy learns to act on filtered/smoothed enemy state estimates

Why Train with Kalman Tracking:
-------------------------------
1. More realistic: In deployment, enemy state comes from sensors + estimation
2. Velocity estimates: Kalman filter provides v_hat (not directly observable)
3. Noise filtering: Smoothed estimates may enable better policy learning
4. Lead-time prediction: Can extrapolate enemy position for intercept planning

Prerequisites:
    pip install stable-baselines3

Usage:
    python experiments/rl_kalman/train_ppo_kalman.py
    
    # With custom timesteps
    python experiments/rl_kalman/train_ppo_kalman.py --total-timesteps 500000
    
    # Custom Kalman parameters
    python experiments/rl_kalman/train_ppo_kalman.py --process-var 1.0 --measurement-var 0.5

Output:
    - Final model: results/rl_kalman/models/ppo_kalman_final.zip
    - Checkpoints: results/rl_kalman/models/ppo_kalman_checkpoint_{timestep}.zip
    - Training logs: results/rl_kalman/logs/

Evaluate with:
    python experiments/rl_kalman/evaluate_rl_kalman.py \\
        --model results/rl_kalman/models/ppo_kalman_final.zip \\
        --n-episodes 1000
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_config import KALMAN_CONFIG
from experiments.ppo_hyperparams import (
    TOTAL_TIMESTEPS,
    LEARNING_RATE,
    GAMMA,
    BATCH_SIZE,
    N_STEPS,
    N_EPOCHS,
    CLIP_RANGE,
    ENT_COEF,
    VF_COEF,
    MAX_GRAD_NORM,
    GAE_LAMBDA,
    CHECKPOINT_FREQ,
    EVAL_FREQ,
    N_EVAL_EPISODES,
    SEED,
)

# =============================================================================
# KALMAN FILTER PARAMETERS
# =============================================================================
# From shared config for reproducibility. All other PPO hyperparameters are
# imported from experiments.ppo_hyperparams above -- the SINGLE SOURCE OF
# TRUTH shared with experiments/rl/train_ppo.py (Direct track), so the two
# tracks can never silently drift apart.
PROCESS_VAR = KALMAN_CONFIG["process_var"]
MEASUREMENT_VAR = KALMAN_CONFIG["measurement_var"]
LEAD_TIME = KALMAN_CONFIG["lead_time"]

# =============================================================================
# DIRECTORIES
# =============================================================================
MODEL_DIR = PROJECT_ROOT / "results" / "rl_kalman" / "models"
LOG_DIR = PROJECT_ROOT / "results" / "rl_kalman" / "logs"
TRAINING_LOG_DIR = PROJECT_ROOT / "results" / "rl_kalman" / "training_logs"

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import json

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
        BaseCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.utils import set_random_seed
except ImportError as e:
    print("ERROR: stable-baselines3 is required for training.")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from experiments.success_rate_eval_callback import SuccessRateEvalCallback


# =============================================================================
# NOTE ON MODEL-SELECTION CALLBACK
# =============================================================================
# The success-rate-based evaluation/model-selection callback is now shared
# with experiments/rl/train_ppo.py (Direct track) via
# experiments.success_rate_eval_callback.SuccessRateEvalCallback, so both
# tracks use an IDENTICAL best-model-selection criterion (success rate).
# See create_callbacks() below.


def make_env(
    seed: int = 0,
    rank: int = 0,
    process_var: float = PROCESS_VAR,
    measurement_var: float = MEASUREMENT_VAR,
    lead_time: float = LEAD_TIME,
) -> callable:
    """
    Create a factory function for making Kalman-enabled environments.
    
    Args:
        seed: Base random seed.
        rank: Environment rank (for vectorized envs).
        process_var: Kalman filter process noise variance.
        measurement_var: Kalman filter measurement noise variance.
        lead_time: Prediction lead time for extrapolation.
    
    Returns:
        Callable that creates a SoldierEnv with Kalman tracking enabled.
    """
    def _init():
        # Configure environment with Kalman tracking
        config = EnvConfig(
            use_kalman_tracking=True,
            process_var=process_var,
            measurement_var=measurement_var,
            lead_time=lead_time,
        )
        env = SoldierEnv(config=config)
        env.reset(seed=seed + rank)
        
        # Monitor wrapper for logging
        log_dir = LOG_DIR / f"env_{rank}"
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(log_dir))
        return env
    return _init


def create_training_env(
    seed: int = SEED,
    process_var: float = PROCESS_VAR,
    measurement_var: float = MEASUREMENT_VAR,
    lead_time: float = LEAD_TIME,
) -> DummyVecEnv:
    """Create the training environment with Kalman tracking."""
    set_random_seed(seed)
    env = DummyVecEnv([make_env(
        seed=seed,
        rank=0,
        process_var=process_var,
        measurement_var=measurement_var,
        lead_time=lead_time,
    )])
    return env


def create_eval_env(
    seed: int = SEED + 1000,
    process_var: float = PROCESS_VAR,
    measurement_var: float = MEASUREMENT_VAR,
    lead_time: float = LEAD_TIME,
) -> DummyVecEnv:
    """Create evaluation environment with same Kalman configuration."""
    env = DummyVecEnv([make_env(
        seed=seed,
        rank=0,
        process_var=process_var,
        measurement_var=measurement_var,
        lead_time=lead_time,
    )])
    return env


def create_callbacks(
    eval_env: DummyVecEnv,
    checkpoint_freq: int = CHECKPOINT_FREQ,
    eval_freq: int = EVAL_FREQ,
    n_eval_episodes: int = N_EVAL_EPISODES,
    process_var: float = PROCESS_VAR,
    measurement_var: float = MEASUREMENT_VAR,
    lead_time: float = LEAD_TIME,
    eval_seed: int = SEED + 2000,
) -> CallbackList:
    """Create training callbacks for checkpointing and custom evaluation."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_kalman_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )
    
    # Shared success-rate evaluation callback (use_kalman_tracking=True) --
    # IDENTICAL selection criterion to the Direct track (see
    # experiments/rl/train_ppo.py and experiments/success_rate_eval_callback.py).
    success_rate_eval_callback = SuccessRateEvalCallback(
        use_kalman_tracking=True,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        process_var=process_var,
        measurement_var=measurement_var,
        lead_time=lead_time,
        eval_seed=eval_seed,
        log_dir=str(TRAINING_LOG_DIR),
        best_model_save_path=str(MODEL_DIR),
        verbose=1,
    )
    
    return CallbackList([checkpoint_callback, success_rate_eval_callback])


def train_ppo_kalman(
    total_timesteps: int = TOTAL_TIMESTEPS,
    learning_rate: float = LEARNING_RATE,
    gamma: float = GAMMA,
    batch_size: int = BATCH_SIZE,
    n_steps: int = N_STEPS,
    n_epochs: int = N_EPOCHS,
    clip_range: float = CLIP_RANGE,
    ent_coef: float = ENT_COEF,
    vf_coef: float = VF_COEF,
    max_grad_norm: float = MAX_GRAD_NORM,
    gae_lambda: float = GAE_LAMBDA,
    process_var: float = PROCESS_VAR,
    measurement_var: float = MEASUREMENT_VAR,
    lead_time: float = LEAD_TIME,
    seed: int = SEED,
    resume_path: str | None = None,
    verbose: int = 1,
) -> PPO:
    """
    Train a PPO agent on SoldierEnv with Kalman tracking.
    
    Args:
        total_timesteps: Total training timesteps.
        learning_rate: Learning rate.
        gamma: Discount factor.
        batch_size: Minibatch size.
        n_steps: Steps per rollout.
        n_epochs: Epochs per update.
        clip_range: PPO clip range.
        ent_coef: Entropy coefficient.
        vf_coef: Value function coefficient.
        max_grad_norm: Max gradient norm.
        gae_lambda: GAE lambda.
        process_var: Kalman filter process noise variance.
        measurement_var: Kalman filter measurement noise variance.
        lead_time: Prediction extrapolation time.
        seed: Random seed.
        resume_path: Path to checkpoint to resume from.
        verbose: Verbosity level.
    
    Returns:
        Trained PPO model.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create environments with Kalman tracking
    print("=" * 60)
    print("CREATING KALMAN-ENABLED ENVIRONMENTS")
    print("=" * 60)
    train_env = create_training_env(
        seed=seed,
        process_var=process_var,
        measurement_var=measurement_var,
        lead_time=lead_time,
    )
    eval_env = create_eval_env(
        seed=seed + 1000,
        process_var=process_var,
        measurement_var=measurement_var,
        lead_time=lead_time,
    )
    
    print(f"Training env created with seed {seed}")
    print(f"Eval env created with seed {seed + 1000}")
    print(f"\nKalman Filter Configuration:")
    print(f"  use_kalman_tracking: True")
    print(f"  process_var:         {process_var}")
    print(f"  measurement_var:     {measurement_var}")
    print(f"  lead_time:           {lead_time}")
    
    # Create or load model
    print("\n" + "=" * 60)
    print("INITIALIZING PPO MODEL")
    print("=" * 60)
    
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from checkpoint: {resume_path}")
        model = PPO.load(resume_path, env=train_env, verbose=verbose)
        model.learning_rate = learning_rate
    else:
        print("Creating new PPO model")
        
        try:
            import tensorboard
            tb_log = str(LOG_DIR / "tensorboard")
        except ImportError:
            tb_log = None
            print("  (TensorBoard not installed - logging disabled)")
        
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            n_steps=n_steps,
            n_epochs=n_epochs,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            seed=seed,
            verbose=verbose,
            tensorboard_log=tb_log,
        )
    
    # Print hyperparameters
    print(f"\nHyperparameters:")
    print(f"  total_timesteps:  {total_timesteps:,}")
    print(f"  learning_rate:    {learning_rate}")
    print(f"  gamma:            {gamma}")
    print(f"  batch_size:       {batch_size}")
    print(f"  n_steps:          {n_steps}")
    print(f"  n_epochs:         {n_epochs}")
    print(f"  clip_range:       {clip_range}")
    print(f"  ent_coef:         {ent_coef}")
    print(f"  vf_coef:          {vf_coef}")
    print(f"  gae_lambda:       {gae_lambda}")
    print(f"  seed:             {seed}")
    
    # Create callbacks with custom Kalman evaluation
    callbacks = create_callbacks(
        eval_env=eval_env,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        process_var=process_var,
        measurement_var=measurement_var,
        lead_time=lead_time,
        eval_seed=seed + 2000,
    )
    
    # Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING (with Kalman Tracking)")
    print("=" * 60)
    print(f"Training for {total_timesteps:,} timesteps...")
    print(f"Checkpoints every {CHECKPOINT_FREQ:,} steps")
    print(f"Evaluation every {EVAL_FREQ:,} steps ({N_EVAL_EPISODES} episodes)")
    print(f"Evaluation logs: {TRAINING_LOG_DIR}")
    print()
    
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    training_time = datetime.now() - start_time
    
    # Save final model
    print("\n" + "=" * 60)
    print("SAVING FINAL MODEL")
    print("=" * 60)
    
    final_model_path = MODEL_DIR / "ppo_kalman_final.zip"
    model.save(str(final_model_path))
    print(f"Final model saved to: {final_model_path}")
    
    # Print training summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total timesteps:    {total_timesteps:,}")
    print(f"Training time:      {training_time}")
    print(f"Final model:        {final_model_path}")
    print()
    print("To evaluate the trained model:")
    print(f"  python experiments/rl_kalman/evaluate_rl_kalman.py \\")
    print(f"      --model {final_model_path} --n-episodes 1000")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on SoldierEnv with Kalman tracking"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS:,})"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--process-var", type=float, default=PROCESS_VAR,
        help=f"Kalman process noise variance (default: {PROCESS_VAR})"
    )
    parser.add_argument(
        "--measurement-var", type=float, default=MEASUREMENT_VAR,
        help=f"Kalman measurement noise variance (default: {MEASUREMENT_VAR})"
    )
    parser.add_argument(
        "--lead-time", type=float, default=LEAD_TIME,
        help=f"Prediction lead time in seconds (default: {LEAD_TIME})"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="Verbosity level (0=none, 1=info, 2=debug)"
    )
    
    args = parser.parse_args()
    
    train_ppo_kalman(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        process_var=args.process_var,
        measurement_var=args.measurement_var,
        lead_time=args.lead_time,
        seed=args.seed,
        resume_path=args.resume,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
