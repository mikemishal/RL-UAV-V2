"""
Parameter sweep: Enemy speed vs success probability (RL policy).

Evaluates the trained PPO policy across a range of enemy speeds
to quantify how performance degrades as the attacker gets faster.

Mirrors the baseline sweep script exactly for direct comparison.

Usage:
    python experiments/rl/sweep_enemy_speed_rl.py --model results/rl/models/ppo_defender_final.zip
    python experiments/rl/sweep_enemy_speed_rl.py --model <path> --n-episodes 200 --speeds "8,10,12,14,16"
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from uav_defend.envs import SoldierEnv
from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.rl import PPOPolicyWrapper

from experiments.eval_utils import evaluate_policy
from experiments.experiment_config import (
    CONFIG,
    SWEEP_CONFIG,
    format_speeds_for_cli,
    parse_speeds_from_cli,
)


def sweep_enemy_speed(
    model_path: str,
    speeds: list[float],
    n_episodes: int = 200,
    seed_offset: int = 0,
    deterministic: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run parameter sweep over enemy speeds using trained PPO policy.
    
    Args:
        model_path: Path to trained PPO model.
        speeds: List of enemy speeds to evaluate.
        n_episodes: Number of episodes per speed setting.
        seed_offset: Starting seed for reproducibility.
        deterministic: If True, use deterministic actions.
        verbose: If True, print progress.
    
    Returns:
        DataFrame with one row per speed setting.
    """
    if verbose:
        print("=" * 60)
        print("PARAMETER SWEEP: ENEMY SPEED (PPO)")
        print("=" * 60)
        print(f"Model:        {model_path}")
        print(f"Speeds:       {speeds}")
        print(f"Episodes:     {n_episodes} per speed")
        print(f"Total runs:   {len(speeds) * n_episodes}")
        print("-" * 60)
    
    # Load PPO model
    policy = PPOPolicyWrapper.load(model_path, deterministic=deterministic)
    
    results = []
    seeds = range(seed_offset, seed_offset + n_episodes)
    
    # Get default defender speed for reference
    default_config = EnvConfig()
    defender_speed = default_config.v_d
    
    for v_e in speeds:
        if verbose:
            print(f"\nEvaluating v_e = {v_e:.1f}...")
        
        # Create environment factory with custom config
        def env_factory(speed=v_e):
            config = EnvConfig(v_e=speed)
            return SoldierEnv(config=config)
        
        # Run evaluation
        df, summary, _ = evaluate_policy(
            env_factory=env_factory,
            policy=policy,
            seeds=seeds,
            verbose=False,
        )
        
        # Compute standard error of success rate
        p = summary["success_rate"]
        n = summary["n_episodes"]
        se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        
        # Get mean intercept time for successful episodes
        success_df = df[df["success"] == 1]
        if len(success_df) > 0:
            mean_intercept_time = success_df["intercept_time"].mean()
        else:
            mean_intercept_time = float("nan")
        
        # Store results
        results.append({
            "enemy_speed": v_e,
            "defender_speed": defender_speed,
            "speed_ratio": defender_speed / v_e,  # v_d / v_e
            "n_episodes": n,
            "n_success": int(p * n),
            "success_rate": p,
            "success_se": se,
            "failure_rate": summary["failure_rate"],
            "timeout_rate": summary["timeout_rate"],
            "mean_episode_length": summary["mean_episode_length"],
            "std_episode_length": summary["std_episode_length"],
            "mean_detection_time": summary["mean_detection_time"],
            "mean_intercept_time": mean_intercept_time,
            "mean_min_enemy_soldier_dist": summary["mean_min_enemy_soldier_dist"],
        })
        
        if verbose:
            print(f"  Success rate: {p:.1%} ± {se:.1%}")
    
    if verbose:
        print("-" * 60)
        print("Sweep complete!")
    
    return pd.DataFrame(results)


def plot_sweep(df: pd.DataFrame, output_path: str, show: bool = False) -> None:
    """
    Generate enemy speed sweep plot with error bars.
    
    Args:
        df: DataFrame from sweep_enemy_speed().
        output_path: Path to save the plot.
        show: If True, display the plot interactively.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot success rate with error bars
    ax.errorbar(
        df["enemy_speed"],
        df["success_rate"] * 100,
        yerr=df["success_se"] * 100,
        fmt='o-',
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=8,
        color='#9B59B6',  # Purple for PPO
        ecolor='#34495E',
        label='PPO Policy'
    )
    
    # Add reference line for defender speed
    defender_speed = df["defender_speed"].iloc[0]
    ax.axvline(x=defender_speed, color='blue', linestyle='--', alpha=0.7,
               label=f'Defender speed (v_d = {defender_speed})')
    
    # Formatting
    ax.set_xlabel('Enemy Speed (v_e)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('PPO Policy: Success Rate vs Enemy Speed', fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_xlim(df["enemy_speed"].min() - 1, df["enemy_speed"].max() + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add speed ratio as secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    speed_ratios = defender_speed / df["enemy_speed"]
    ax2.set_xticks(df["enemy_speed"])
    ax2.set_xticklabels([f'{r:.2f}' for r in speed_ratios])
    ax2.set_xlabel('Speed Ratio (v_d / v_e)', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    # Get defaults from shared config
    sweep_defaults = SWEEP_CONFIG["enemy_speed"]
    default_speeds = format_speeds_for_cli(sweep_defaults["parameter_values"])
    default_episodes = sweep_defaults["n_episodes"]
    
    parser = argparse.ArgumentParser(
        description="Parameter sweep: enemy speed vs success rate (PPO)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained PPO model (.zip file)"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=default_episodes,
        help=f"Number of episodes per speed setting (default: {default_episodes})"
    )
    parser.add_argument(
        "--speeds", type=str, default=default_speeds,
        help=f"Comma-separated list of enemy speeds (default: {default_speeds})"
    )
    parser.add_argument(
        "--seed-offset", type=int, default=CONFIG.SEED_OFFSET,
        help=f"Starting seed for reproducibility (default: {CONFIG.SEED_OFFSET})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/rl/)"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    parser.add_argument(
        "--show-plot", action="store_true",
        help="Display plot interactively"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Parse speeds using shared utility
    speeds = parse_speeds_from_cli(args.speeds)
    
    # Run sweep
    df = sweep_enemy_speed(
        model_path=args.model,
        speeds=speeds,
        n_episodes=args.n_episodes,
        seed_offset=args.seed_offset,
        deterministic=not args.stochastic,
        verbose=not args.quiet,
    )
    
    # Set up output paths
    if args.output_dir:
        output_dir = args.output_dir
    else:
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        output_dir = os.path.join(project_root, 'results', 'rl')
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'sweep_enemy_speed.csv')
    plot_path = os.path.join(output_dir, 'sweep_enemy_speed.png')
    
    # Save results
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SWEEP RESULTS (PPO)")
    print("=" * 60)
    print(f"{'Speed':>8} {'Ratio':>8} {'Success':>10} {'SE':>8} {'Ep Len':>8}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['enemy_speed']:>8.1f} {row['speed_ratio']:>8.2f} "
              f"{row['success_rate']*100:>9.1f}% {row['success_se']*100:>7.1f}% "
              f"{row['mean_episode_length']:>8.1f}")
    print("=" * 60)
    
    # Generate plot
    plot_sweep(df, plot_path, show=args.show_plot)
    
    return df


if __name__ == "__main__":
    main()
