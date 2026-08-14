"""
RL Training Experiments.

This module contains scripts for training RL policies on the UAV Defender
environment using Stable-Baselines3.

Scripts:
    train_ppo.py - Train a PPO agent on SoldierEnv

Usage:
    # Train with default settings
    python experiments/rl/train_ppo.py
    
    # Train with custom timesteps
    python experiments/rl/train_ppo.py --total-timesteps 500000
    
    # Resume from checkpoint
    python experiments/rl/train_ppo.py --resume results/rl/models/ppo_checkpoint_100000.zip

After training, evaluate with:
    python experiments/evaluate_policy.py --policy ppo \\
        --model-path results/rl/models/ppo_defender_final.zip \\
        --n-episodes 1000
"""
