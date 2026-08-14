"""
PPO Training Script for UAV Defender.

This script trains a PPO agent on the SoldierEnv using Stable-Baselines3.
The trained policy can be evaluated using the same pipeline as baseline policies.

Prerequisites:
    pip install stable-baselines3
    
    Note: On Windows, you may need to enable Long Paths support:
    https://pip.pypa.io/warnings/enable-long-paths

Usage:
    python experiments/rl/train_ppo.py
    
    # With custom timesteps
    python experiments/rl/train_ppo.py --total-timesteps 500000
    
    # Resume from checkpoint
    python experiments/rl/train_ppo.py --resume results/rl/models/ppo_checkpoint_100000.zip

Output:
    - Final model: results/rl/models/ppo_defender_final.zip
    - Checkpoints: results/rl/models/ppo_checkpoint_{timestep}.zip
    - Training logs: results/rl/logs/

Evaluate with:
    python experiments/evaluate_policy.py --policy ppo \\
        --model-path results/rl/models/ppo_defender_final.zip \\
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

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# These are exposed at the top for easy tuning.
# Override via command line: --total-timesteps 500000 --learning-rate 0.0001

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

# Checkpointing
CHECKPOINT_FREQ = 50_000       # Save checkpoint every N timesteps
EVAL_FREQ = 10_000             # Evaluate policy every N timesteps
N_EVAL_EPISODES = 20           # Episodes per evaluation

# Reproducibility
SEED = 42                      # Random seed for reproducibility

# =============================================================================
# DIRECTORIES
# =============================================================================
MODEL_DIR = PROJECT_ROOT / "results" / "rl" / "models"
LOG_DIR = PROJECT_ROOT / "results" / "rl" / "logs"

# =============================================================================
# IMPORTS (after path setup)
# =============================================================================
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed
except ImportError as e:
    print("ERROR: stable-baselines3 is required for training.")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig


def make_env(seed: int = 0, rank: int = 0) -> callable:
    """
    Create a factory function for making environments.
    
    Args:
        seed: Base random seed.
        rank: Environment rank (for vectorized envs).
    
    Returns:
        Callable that creates a SoldierEnv wrapped with Monitor.
    """
    def _init():
        env = SoldierEnv()
        env.reset(seed=seed + rank)
        # Monitor wrapper for logging episode stats
        log_dir = LOG_DIR / f"env_{rank}"
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(log_dir))
        return env
    return _init


def create_training_env(seed: int = SEED) -> DummyVecEnv:
    """
    Create the training environment with monitoring.
    
    Args:
        seed: Random seed for reproducibility.
    
    Returns:
        Vectorized environment with monitoring.
    """
    set_random_seed(seed)
    env = DummyVecEnv([make_env(seed=seed, rank=0)])
    return env


def create_eval_env(seed: int = SEED + 1000) -> DummyVecEnv:
    """
    Create a separate evaluation environment.
    
    Uses different seeds than training to test generalization.
    
    Args:
        seed: Random seed (offset from training seed).
    
    Returns:
        Vectorized environment for evaluation.
    """
    env = DummyVecEnv([make_env(seed=seed, rank=0)])
    return env


def create_callbacks(
    eval_env: DummyVecEnv,
    checkpoint_freq: int = CHECKPOINT_FREQ,
    eval_freq: int = EVAL_FREQ,
    n_eval_episodes: int = N_EVAL_EPISODES,
) -> CallbackList:
    """
    Create training callbacks for checkpointing and evaluation.
    
    Args:
        eval_env: Environment for evaluation.
        checkpoint_freq: Steps between checkpoints.
        eval_freq: Steps between evaluations.
        n_eval_episodes: Episodes per evaluation.
    
    Returns:
        CallbackList with checkpoint and eval callbacks.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )
    
    # Evaluation callback - evaluate and save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(LOG_DIR),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    return CallbackList([checkpoint_callback, eval_callback])


def train_ppo(
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
    seed: int = SEED,
    resume_path: str | None = None,
    verbose: int = 1,
) -> PPO:
    """
    Train a PPO agent on SoldierEnv.
    
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
        seed: Random seed.
        resume_path: Path to checkpoint to resume from.
        verbose: Verbosity level (0=none, 1=info, 2=debug).
    
    Returns:
        Trained PPO model.
    """
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create environments
    print("=" * 60)
    print("CREATING ENVIRONMENTS")
    print("=" * 60)
    train_env = create_training_env(seed=seed)
    eval_env = create_eval_env(seed=seed + 1000)
    
    print(f"Training env created with seed {seed}")
    print(f"Eval env created with seed {seed + 1000}")
    
    # Create or load model
    print("\n" + "=" * 60)
    print("INITIALIZING PPO MODEL")
    print("=" * 60)
    
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from checkpoint: {resume_path}")
        model = PPO.load(resume_path, env=train_env, verbose=verbose)
        # Update learning rate if different
        model.learning_rate = learning_rate
    else:
        print("Creating new PPO model")
        
        # Check if tensorboard is available
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
    
    # Create callbacks
    callbacks = create_callbacks(
        eval_env=eval_env,
        checkpoint_freq=CHECKPOINT_FREQ,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
    )
    
    # Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Training for {total_timesteps:,} timesteps...")
    print(f"Checkpoints every {CHECKPOINT_FREQ:,} steps")
    print(f"Evaluation every {EVAL_FREQ:,} steps ({N_EVAL_EPISODES} episodes)")
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
    
    final_model_path = MODEL_DIR / "ppo_defender_final.zip"
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
    print(f"  python experiments/evaluate_policy.py --policy ppo \\")
    print(f"      --model-path {final_model_path} --n-episodes 1000")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent on UAV Defender environment"
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
        "--gamma", type=float, default=GAMMA,
        help=f"Discount factor (default: {GAMMA})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Minibatch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--n-steps", type=int, default=N_STEPS,
        help=f"Steps per rollout (default: {N_STEPS})"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, choices=[0, 1, 2],
        help="Verbosity level: 0=none, 1=info, 2=debug (default: 1)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("PPO TRAINING SCRIPT")
    print("UAV Defender - SoldierEnv")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    model = train_ppo(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        seed=args.seed,
        resume_path=args.resume,
        verbose=args.verbose,
    )
    
    return model


if __name__ == "__main__":
    main()
