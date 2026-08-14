"""
Parameter sweep: Detection radius vs success probability (RL-Kalman policy).

=============================================================================
EXPERIMENT TRACK: RL WITH KALMAN-ESTIMATED ENEMY STATE
=============================================================================

Evaluates the RL-Kalman PPO policy across a range of detection radii.
Mirrors experiments/rl/sweep_detection_radius_rl.py for comparison.

Usage:
    python experiments/rl_kalman/sweep_detection_radius_rl_kalman.py \\
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
from experiments.experiment_config import CONFIG, KALMAN_CONFIG, SWEEP_CONFIG

# Kalman configuration from shared config
PROCESS_VAR = KALMAN_CONFIG["process_var"]
MEASUREMENT_VAR = KALMAN_CONFIG["measurement_var"]
LEAD_TIME = KALMAN_CONFIG["lead_time"]


def sweep_detection_radius(
    model_path: str,
    radii: list[float],
    n_episodes: int = 200,
    seed_offset: int = 0,
    deterministic: bool = True,
    process_var: float = PROCESS_VAR,
    measurement_var: float = MEASUREMENT_VAR,
    lead_time: float = LEAD_TIME,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run parameter sweep over detection radii using RL-Kalman policy."""
    if verbose:
        print("=" * 60)
        print("PARAMETER SWEEP: DETECTION RADIUS (RL-KALMAN)")
        print("=" * 60)
        print(f"Model:        {model_path}")
        print(f"Radii:        {radii}")
        print(f"Episodes:     {n_episodes} per radius")
        print(f"Kalman:       process_var={process_var}, measurement_var={measurement_var}")
        print("-" * 60)
    
    policy = PPOKalmanPolicyWrapper.load(model_path, deterministic=deterministic)
    
    results = []
    seeds = range(seed_offset, seed_offset + n_episodes)
    
    for radius in radii:
        if verbose:
            print(f"\nEvaluating detection_radius = {radius:.1f}...")
        
        def env_factory(r=radius):
            config = EnvConfig(
                detection_radius=r,
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
        
        # Compute detection rate
        detected_count = df["detected"].sum() if "detected" in df.columns else n
        detection_rate = detected_count / n if n > 0 else 0.0
        
        # Compute tracking error metrics (Kalman-specific)
        if "mean_tracking_error" in df.columns:
            valid_tracking = df["mean_tracking_error"].dropna()
            mean_tracking_error = valid_tracking.mean() if len(valid_tracking) > 0 else float("nan")
        else:
            mean_tracking_error = float("nan")
        
        results.append({
            "detection_radius": radius,
            "n_episodes": n,
            "n_success": int(p * n),
            "success_rate": p,
            "success_se": se,
            "detection_rate": detection_rate,
            "failure_rate": summary["failure_rate"],
            "timeout_rate": summary["timeout_rate"],
            "mean_episode_length": summary["mean_episode_length"],
            "std_episode_length": summary["std_episode_length"],
            "mean_detection_time": summary["mean_detection_time"],
            "mean_min_enemy_soldier_dist": summary["mean_min_enemy_soldier_dist"],
            "mean_tracking_error": mean_tracking_error,
        })
        
        if verbose:
            track_str = f", track_err={mean_tracking_error:.3f}" if not np.isnan(mean_tracking_error) else ""
            print(f"  Success rate: {p:.1%} ± {se:.1%}, Detection: {detection_rate:.0%}{track_str}")
    
    if verbose:
        print("-" * 60)
        print("Sweep complete!")
    
    return pd.DataFrame(results)


def plot_sweep(df: pd.DataFrame, output_path: str, show: bool = False) -> None:
    """Generate detection radius sweep plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(
        df["detection_radius"],
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
    
    ax.set_xlabel('Detection Radius', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('RL-Kalman Policy: Success Rate vs Detection Radius', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    sweep_defaults = SWEEP_CONFIG["detection_radius"]
    default_radii = sweep_defaults["parameter_values"]
    default_episodes = sweep_defaults["n_episodes"]
    
    parser = argparse.ArgumentParser(
        description="Parameter sweep: detection radius (RL-Kalman)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n-episodes", type=int, default=default_episodes)
    parser.add_argument(
        "--radii", type=str, default=",".join(str(r) for r in default_radii),
        help=f"Comma-separated list of detection radii (default: {default_radii})"
    )
    parser.add_argument("--seed-offset", type=int, default=CONFIG.SEED_OFFSET)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--process-var", type=float, default=PROCESS_VAR)
    parser.add_argument("--measurement-var", type=float, default=MEASUREMENT_VAR)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    radii = [float(r) for r in args.radii.split(",")]
    
    df = sweep_detection_radius(
        model_path=args.model,
        radii=radii,
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
    
    csv_path = os.path.join(output_dir, 'sweep_detection_radius.csv')
    plot_path = os.path.join(output_dir, 'sweep_detection_radius.png')
    
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    print("\n" + "=" * 60)
    print("SWEEP RESULTS (RL-KALMAN)")
    print("=" * 60)
    print(f"{'Radius':>8} {'Success':>10} {'SE':>8} {'Detect':>8} {'Ep Len':>8}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{row['detection_radius']:>8.1f} "
              f"{row['success_rate']*100:>9.1f}% {row['success_se']*100:>7.1f}% "
              f"{row['detection_rate']*100:>7.0f}% "
              f"{row['mean_episode_length']:>8.1f}")
    print("=" * 60)
    
    plot_sweep(df, plot_path, show=args.show_plot)


if __name__ == "__main__":
    main()
