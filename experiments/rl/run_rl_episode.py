"""
Single-Episode RL Rollout Script.

Run one episode with a trained PPO model for quick qualitative debugging.
Use this to verify trained behavior before running Monte Carlo evaluation.

Usage:
    python experiments/rl/run_rl_episode.py --model results/rl/models/ppo_defender_final.zip
    
    # Specific seed
    python experiments/rl/run_rl_episode.py --model results/rl/models/ppo_defender_final.zip --seed 7
    
    # Save trajectory
    python experiments/rl/run_rl_episode.py --model results/rl/models/ppo_defender_final.zip --save-traj

After verification, run full evaluation:
    python experiments/evaluate_policy.py --policy ppo --model-path <model.zip> --n-episodes 1000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from uav_defend.envs import SoldierEnv
from uav_defend.policies.rl import PPOPolicyWrapper


def run_episode(
    model_path: str,
    seed: int = 0,
    deterministic: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run a single episode with a trained PPO model.
    
    Args:
        model_path: Path to the trained PPO model (.zip file).
        seed: Random seed for the environment.
        deterministic: If True, use deterministic actions.
        verbose: If True, print step-by-step info.
    
    Returns:
        Dictionary containing episode results and trajectory data.
    """
    # Load policy
    if verbose:
        print("=" * 60)
        print("SINGLE-EPISODE RL ROLLOUT")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Seed:  {seed}")
        print(f"Mode:  {'deterministic' if deterministic else 'stochastic'}")
        print()
    
    policy = PPOPolicyWrapper.load(model_path, deterministic=deterministic)
    
    # Create environment
    env = SoldierEnv()
    obs, info = env.reset(seed=seed)
    policy.reset()
    
    # Trajectory recording
    soldier_positions = [env._soldier_pos.copy()]
    defender_positions = [env._defender_pos.copy()]
    enemy_positions = [env._enemy_pos.copy()]
    e_hat_positions = []
    actions = []
    rewards = []
    
    # Episode metrics
    total_reward = 0.0
    detection_time = None
    intercept_time = None
    step = 0
    
    if verbose:
        print("Initial state:")
        print(f"  Soldier:  ({env._soldier_pos[0]:6.2f}, {env._soldier_pos[1]:6.2f})")
        print(f"  Defender: ({env._defender_pos[0]:6.2f}, {env._defender_pos[1]:6.2f})")
        print(f"  Enemy:    ({env._enemy_pos[0]:6.2f}, {env._enemy_pos[1]:6.2f})")
        print()
        print("-" * 60)
        print(f"{'Step':>4} {'Action':>16} {'Reward':>8} {'d(e,s)':>8} {'d(d,e)':>8} {'Det':>4}")
        print("-" * 60)
    
    done = False
    while not done:
        # Get action from policy
        action = policy.act(obs, info)
        actions.append(action.copy())
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        total_reward += reward
        rewards.append(reward)
        
        # Record positions
        soldier_positions.append(env._soldier_pos.copy())
        defender_positions.append(env._defender_pos.copy())
        enemy_positions.append(env._enemy_pos.copy())
        
        # Record e_hat if detected
        if info.get("e_hat") is not None:
            e_hat_positions.append(info["e_hat"].copy())
        else:
            e_hat_positions.append(np.array([np.nan, np.nan]))
        
        # Track detection time
        if info["enemy_detected"] and detection_time is None:
            detection_time = step
        
        # Print step info
        if verbose:
            det_str = "Y" if info["enemy_detected"] else "N"
            print(f"{step:4d} ({action[0]:+.2f}, {action[1]:+.2f}) "
                  f"{reward:+8.2f} {info['enemy_soldier_dist']:8.2f} "
                  f"{info['defender_enemy_dist']:8.2f} {det_str:>4}")
        
        if done or truncated:
            break
    
    # Record intercept time if successful
    if info["outcome"] == "intercepted":
        intercept_time = step
    
    # Build results
    results = {
        "seed": seed,
        "outcome": info["outcome"],
        "episode_length": step,
        "total_reward": total_reward,
        "detection_time": detection_time,
        "intercept_time": intercept_time,
        "final_enemy_soldier_dist": info["enemy_soldier_dist"],
        "final_defender_enemy_dist": info["defender_enemy_dist"],
        # Trajectory data
        "soldier_positions": np.array(soldier_positions),
        "defender_positions": np.array(defender_positions),
        "enemy_positions": np.array(enemy_positions),
        "e_hat_positions": np.array(e_hat_positions) if e_hat_positions else None,
        "actions": np.array(actions),
        "rewards": np.array(rewards),
    }
    
    # Print summary
    if verbose:
        print("-" * 60)
        print()
        print("=" * 60)
        print("EPISODE SUMMARY")
        print("=" * 60)
        
        outcome_emoji = {
            "intercepted": "✓ WIN",
            "soldier_caught": "✗ LOSS (soldier caught)",
            "unsafe_intercept": "✗ LOSS (unsafe intercept)",
            "timeout": "✗ LOSS (timeout)",
        }
        
        print(f"\nOutcome: {outcome_emoji.get(info['outcome'], info['outcome'])}")
        print(f"\nMetrics:")
        print(f"  Episode length:     {step} steps")
        print(f"  Total reward:       {total_reward:.2f}")
        print(f"  Detection time:     {detection_time if detection_time else 'never'}")
        print(f"  Intercept time:     {intercept_time if intercept_time else 'N/A'}")
        print(f"\nFinal distances:")
        print(f"  d(enemy, soldier):  {info['enemy_soldier_dist']:.2f}")
        print(f"  d(defender, enemy): {info['defender_enemy_dist']:.2f}")
        print("=" * 60)
    
    return results


def save_trajectory(results: dict, output_path: str) -> None:
    """
    Save trajectory data to a .npz file for later plotting.
    
    Args:
        results: Dictionary from run_episode().
        output_path: Path to save the .npz file.
    """
    np.savez(
        output_path,
        seed=results["seed"],
        outcome=results["outcome"],
        episode_length=results["episode_length"],
        total_reward=results["total_reward"],
        detection_time=results["detection_time"] if results["detection_time"] else -1,
        intercept_time=results["intercept_time"] if results["intercept_time"] else -1,
        soldier_positions=results["soldier_positions"],
        defender_positions=results["defender_positions"],
        enemy_positions=results["enemy_positions"],
        e_hat_positions=results["e_hat_positions"] if results["e_hat_positions"] is not None else np.array([]),
        actions=results["actions"],
        rewards=results["rewards"],
    )
    print(f"\nTrajectory saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a single episode with a trained PPO model"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained PPO model (.zip file)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for the episode (default: 0)"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic actions (default: deterministic)"
    )
    parser.add_argument(
        "--save-traj", action="store_true",
        help="Save trajectory data to .npz file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for trajectory (default: results/rl/trajectories/)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress step-by-step output"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Run episode
    results = run_episode(
        model_path=args.model,
        seed=args.seed,
        deterministic=not args.stochastic,
        verbose=not args.quiet,
    )
    
    # Save trajectory if requested
    if args.save_traj:
        if args.output:
            output_path = args.output
        else:
            output_dir = PROJECT_ROOT / "results" / "rl" / "trajectories"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"ppo_trajectory_seed{args.seed}.npz"
        
        save_trajectory(results, str(output_path))
    
    return results


if __name__ == "__main__":
    main()
