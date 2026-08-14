"""
Single-episode rollout script for baseline policy debugging.

Runs exactly one episode with the GreedyInterceptPolicy and prints
detailed diagnostics. Useful for debugging before Monte Carlo evaluation.

Usage:
    python experiments/baseline/run_baseline_episode.py [--seed SEED] [--save]
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from uav_defend.envs import SoldierEnv
from uav_defend.policies import GreedyInterceptPolicy


def run_episode(seed: int = 42, verbose: bool = True) -> dict:
    """
    Run a single episode with the greedy intercept policy.
    
    Args:
        seed: Random seed for reproducibility.
        verbose: If True, print step-by-step progress.
    
    Returns:
        Dictionary containing trajectory data and episode summary.
    """
    env = SoldierEnv()
    policy = GreedyInterceptPolicy()
    
    # Initialize trajectory storage
    trajectory = {
        "soldier_pos": [],
        "defender_pos": [],
        "enemy_pos": [],
        "e_hat": [],
        "actions": [],
        "rewards": [],
        "detected": [],
        "enemy_soldier_dist": [],
        "defender_enemy_dist": [],
    }
    
    # Reset environment and policy
    obs, info = env.reset(seed=seed)
    policy.reset()
    
    # Track detection timing
    detection_step = None
    intercept_step = None
    
    # Store initial state
    trajectory["soldier_pos"].append(info["soldier_pos"].copy())
    trajectory["defender_pos"].append(info["defender_pos"].copy())
    trajectory["enemy_pos"].append(info["enemy_pos"].copy())
    trajectory["e_hat"].append(info["e_hat"].copy() if info["e_hat"] is not None else None)
    trajectory["detected"].append(info["enemy_detected"])
    trajectory["enemy_soldier_dist"].append(info["enemy_soldier_dist"])
    trajectory["defender_enemy_dist"].append(info["defender_enemy_dist"])
    
    if verbose:
        print("=" * 60)
        print(f"BASELINE EPISODE ROLLOUT (seed={seed})")
        print("=" * 60)
        print(f"Initial enemy position: {info['enemy_pos']}")
        print(f"Initial enemy-soldier distance: {info['enemy_soldier_dist']:.2f}")
        print("-" * 60)
    
    step = 0
    total_reward = 0.0
    done = False
    
    while not done:
        # Get action from policy
        action = policy.act(obs, info)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        total_reward += reward
        
        # Store trajectory data
        trajectory["actions"].append(action.copy())
        trajectory["rewards"].append(reward)
        trajectory["soldier_pos"].append(info["soldier_pos"].copy())
        trajectory["defender_pos"].append(info["defender_pos"].copy())
        trajectory["enemy_pos"].append(info["enemy_pos"].copy())
        trajectory["e_hat"].append(info["e_hat"].copy() if info["e_hat"] is not None else None)
        trajectory["detected"].append(info["enemy_detected"])
        trajectory["enemy_soldier_dist"].append(info["enemy_soldier_dist"])
        trajectory["defender_enemy_dist"].append(info["defender_enemy_dist"])
        
        # Track detection timing
        if info["enemy_detected"] and detection_step is None:
            detection_step = step
            if verbose:
                print(f"Step {step:3d}: DETECTION! d_de={info['defender_enemy_dist']:.2f}")
        
        # Verbose output for key events
        if verbose and step % 25 == 0:
            status = "tracking" if info["enemy_detected"] else "searching"
            print(f"Step {step:3d}: [{status}] d_es={info['enemy_soldier_dist']:.2f}, "
                  f"d_de={info['defender_enemy_dist']:.2f}, r={reward:.2f}")
    
    # Record intercept timing
    if info["outcome"] == "intercepted":
        intercept_step = step
    
    # Build summary
    summary = {
        "seed": seed,
        "outcome": info["outcome"],
        "episode_length": step,
        "total_reward": total_reward,
        "detection_occurred": detection_step is not None,
        "detection_step": detection_step,
        "intercept_step": intercept_step,
        "final_enemy_soldier_dist": info["enemy_soldier_dist"],
        "final_defender_enemy_dist": info["defender_enemy_dist"],
        "final_tracking_error": info.get("tracking_error"),
    }
    
    if verbose:
        print("-" * 60)
        print("EPISODE SUMMARY")
        print("-" * 60)
        print(f"  Outcome:           {summary['outcome']}")
        print(f"  Episode length:    {summary['episode_length']} steps")
        print(f"  Total reward:      {summary['total_reward']:.2f}")
        print(f"  Detection:         {'Yes' if summary['detection_occurred'] else 'No'}")
        if summary['detection_step']:
            print(f"  Detection step:    {summary['detection_step']}")
        if summary['intercept_step']:
            print(f"  Intercept step:    {summary['intercept_step']}")
        print(f"  Final d(e,s):      {summary['final_enemy_soldier_dist']:.2f}")
        print(f"  Final d(d,e):      {summary['final_defender_enemy_dist']:.2f}")
        if summary['final_tracking_error'] is not None:
            print(f"  Final track error: {summary['final_tracking_error']:.2f}")
        print("=" * 60)
    
    return {
        "trajectory": trajectory,
        "summary": summary,
    }


def save_trajectory(data: dict, filepath: str) -> None:
    """
    Save trajectory data to a .npz file.
    
    Args:
        data: Dictionary containing trajectory and summary.
        filepath: Output file path.
    """
    # Convert trajectory lists to arrays
    traj = data["trajectory"]
    
    # Handle None values in e_hat
    e_hat_list = traj["e_hat"]
    e_hat_array = np.array([
        e if e is not None else [np.nan, np.nan] 
        for e in e_hat_list
    ])
    
    np.savez(
        filepath,
        # Trajectory arrays
        soldier_pos=np.array(traj["soldier_pos"]),
        defender_pos=np.array(traj["defender_pos"]),
        enemy_pos=np.array(traj["enemy_pos"]),
        e_hat=e_hat_array,
        actions=np.array(traj["actions"]) if traj["actions"] else np.array([]),
        rewards=np.array(traj["rewards"]),
        detected=np.array(traj["detected"]),
        enemy_soldier_dist=np.array(traj["enemy_soldier_dist"]),
        defender_enemy_dist=np.array(traj["defender_enemy_dist"]),
        # Summary (stored as single-element arrays)
        seed=data["summary"]["seed"],
        outcome=data["summary"]["outcome"],
        episode_length=data["summary"]["episode_length"],
        total_reward=data["summary"]["total_reward"],
        detection_step=data["summary"]["detection_step"] if data["summary"]["detection_step"] else -1,
    )
    print(f"Trajectory saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Run single baseline episode for debugging"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save trajectory to experiments/baseline/results/"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress step-by-step output"
    )
    
    args = parser.parse_args()
    
    # Run episode
    data = run_episode(seed=args.seed, verbose=not args.quiet)
    
    # Optionally save
    if args.save:
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, f"trajectory_seed{args.seed}.npz")
        save_trajectory(data, filepath)
    
    return data


if __name__ == "__main__":
    main()
