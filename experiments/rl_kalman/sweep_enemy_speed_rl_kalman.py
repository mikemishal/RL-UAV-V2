"""
Parameter sweep: Enemy speed vs success probability (RL-Kalman policy).

=============================================================================
EXPERIMENT TRACK: RL WITH KALMAN-ESTIMATED ENEMY STATE
=============================================================================

Evaluates the RL-Kalman PPO policy across a range of enemy speeds.
Mirrors experiments/rl/sweep_enemy_speed_rl.py for comparison.

Usage:
    python experiments/rl_kalman/sweep_enemy_speed_rl_kalman.py \\
        --model results/rl_kalman/models/ppo_kalman_final.zip
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from uav_defend.envs import SoldierEnv
from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.rl_kalman import PPOKalmanPolicyWrapper

from experiments.eval_utils import evaluate_policy
from experiments.experiment_config import (
    CONFIG,
    KALMAN_CONFIG,
    SWEEP_CONFIG,
    format_speeds_for_cli,
    parse_speeds_from_cli,
)

# Kalman configuration from shared config
PROCESS_VAR = KALMAN_CONFIG["process_var"]
MEASUREMENT_VAR = KALMAN_CONFIG["measurement_var"]
LEAD_TIME = KALMAN_CONFIG["lead_time"]


def sweep_enemy_speed(
    model_path: str,
    speeds: list[float],
    n_episodes: int = 200,
    seed_offset: int = 0,
    deterministic: bool = True,
    process_var: float = PROCESS_VAR,
    measurement_var: float = MEASUREMENT_VAR,
    lead_time: float = LEAD_TIME,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run parameter sweep over enemy speeds using RL-Kalman policy."""
    if verbose:
        print("=" * 60)
        print("PARAMETER SWEEP: ENEMY SPEED (RL-KALMAN)")
        print("=" * 60)
        print(f"Model:        {model_path}")
        print(f"Speeds:       {speeds}")
        print(f"Episodes:     {n_episodes} per speed")
        print(f"Kalman:       process_var={process_var}, measurement_var={measurement_var}")
        print("-" * 60)
    
    policy = PPOKalmanPolicyWrapper.load(model_path, deterministic=deterministic)
    
    results = []
    seeds = range(seed_offset, seed_offset + n_episodes)
    default_defender_speed = 18.0
    
    for v_e in speeds:
        if verbose:
            print(f"\nEvaluating v_e = {v_e:.1f}...")
        
        def env_factory(speed=v_e):
            config = EnvConfig(
                v_e=speed,
                use_kalman_tracking=True,
                process_var=process_var,
                measurement_var=measurement_var,
                lead_time=lead_time,
            )
            return SoldierEnv(config=config)
        
        df, summary, _ = evaluate_policy(
            env_factory=env_factory,
            policy=policy,
            seeds=seeds,
            verbose=False,
        )
        
        p = summary["success_rate"]
        n = summary["n_episodes"]
        se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        
        success_df = df[df["success"] == 1]
        mean_intercept_time = success_df["intercept_time"].mean() if len(success_df) > 0 else float("nan")
        
        # Compute tracking error metrics (Kalman-specific)
        if "mean_tracking_error" in df.columns:
            valid_tracking = df["mean_tracking_error"].dropna()
            mean_tracking_error = valid_tracking.mean() if len(valid_tracking) > 0 else float("nan")
        else:
            mean_tracking_error = float("nan")
        
        results.append({
            "enemy_speed": v_e,
            "defender_speed": default_defender_speed,
            "speed_ratio": default_defender_speed / v_e,
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
            "mean_tracking_error": mean_tracking_error,
        })
        
        if verbose:
            track_str = f", tracking_err={mean_tracking_error:.3f}" if not np.isnan(mean_tracking_error) else ""
            print(f"  Success rate: {p:.1%} ± {se:.1%}{track_str}")
    
    if verbose:
        print("-" * 60)
        print("Sweep complete!")
    
    return pd.DataFrame(results)


def plot_sweep(df: pd.DataFrame, output_path: str, show: bool = False) -> None:
    """Generate enemy speed sweep plot with error bars."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(
        df["enemy_speed"],
        df["success_rate"] * 100,
        yerr=df["success_se"] * 100,
        fmt='o-',
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=8,
        color='#E74C3C',
        ecolor='#34495E',
        label='RL-Kalman (PPO)'
    )
    
    ax.set_xlabel('Enemy Speed (v_e)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('RL-Kalman Policy: Success Rate vs Enemy Speed', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    sweep_defaults = SWEEP_CONFIG["enemy_speed"]
    default_speeds = format_speeds_for_cli(sweep_defaults["parameter_values"])
    default_episodes = sweep_defaults["n_episodes"]
    
    parser = argparse.ArgumentParser(
        description="Parameter sweep: enemy speed (RL-Kalman)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n-episodes", type=int, default=default_episodes)
    parser.add_argument("--speeds", type=str, default=default_speeds)
    parser.add_argument("--seed-offset", type=int, default=CONFIG.SEED_OFFSET)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--process-var", type=float, default=PROCESS_VAR)
    parser.add_argument("--measurement-var", type=float, default=MEASUREMENT_VAR)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    speeds = parse_speeds_from_cli(args.speeds)
    
    df = sweep_enemy_speed(
        model_path=args.model,
        speeds=speeds,
        n_episodes=args.n_episodes,
        seed_offset=args.seed_offset,
        deterministic=not args.stochastic,
        process_var=args.process_var,
        measurement_var=args.measurement_var,
        verbose=not args.quiet,
    )
    
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '..', '..', 'results', 'rl_kalman'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'sweep_enemy_speed.csv')
    plot_path = os.path.join(output_dir, 'sweep_enemy_speed.png')
    
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    print("\n" + "=" * 60)
    print("SWEEP RESULTS (RL-KALMAN)")
    print("=" * 60)
    print(f"{'v_e':>8} {'Ratio':>8} {'Success':>10} {'SE':>8} {'Ep Len':>8}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['enemy_speed']:>8.1f} {row['speed_ratio']:>8.2f} "
              f"{row['success_rate']*100:>9.1f}% {row['success_se']*100:>7.1f}% "
              f"{row['mean_episode_length']:>8.1f}")
    print("=" * 60)
    
    plot_sweep(df, plot_path, show=args.show_plot)


if __name__ == "__main__":
    main()
