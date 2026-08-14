"""
2D Parameter sweep: Defender speed × Enemy speed grid (RL-Kalman policy).

=============================================================================
EXPERIMENT TRACK: RL WITH KALMAN-ESTIMATED ENEMY STATE
=============================================================================

Evaluates the RL-Kalman PPO policy over a 2D grid of defender/enemy speeds.
Mirrors experiments/rl/sweep_speed_grid_rl.py for comparison.

Usage:
    python experiments/rl_kalman/sweep_speed_grid_rl_kalman.py \\
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


def sweep_speed_grid(
    model_path: str,
    defender_speeds: list[float],
    enemy_speeds: list[float],
    n_episodes: int = 100,
    seed_offset: int = 0,
    deterministic: bool = True,
    process_var: float = PROCESS_VAR,
    measurement_var: float = MEASUREMENT_VAR,
    lead_time: float = LEAD_TIME,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run 2D parameter sweep over speed grid using RL-Kalman policy."""
    if verbose:
        print("=" * 60)
        print("2D PARAMETER SWEEP: DEFENDER SPEED VS ENEMY SPEED (RL-KALMAN)")
        print("=" * 60)
        print(f"Model:           {model_path}")
        print(f"Defender speeds: {defender_speeds}")
        print(f"Enemy speeds:    {enemy_speeds}")
        print(f"Grid points:     {len(defender_speeds) * len(enemy_speeds)}")
        print(f"Episodes/point:  {n_episodes}")
        print(f"Total runs:      {len(defender_speeds) * len(enemy_speeds) * n_episodes}")
        print(f"Kalman:          process_var={process_var}, measurement_var={measurement_var}")
        print("-" * 60)
    
    policy = PPOKalmanPolicyWrapper.load(model_path, deterministic=deterministic)
    
    results = []
    seeds = range(seed_offset, seed_offset + n_episodes)
    total_configs = len(defender_speeds) * len(enemy_speeds)
    config_num = 0
    
    for v_d in defender_speeds:
        for v_e in enemy_speeds:
            config_num += 1
            if verbose:
                print(f"\n[{config_num}/{total_configs}] v_d={v_d:.1f}, v_e={v_e:.1f}...", end=" ")
            
            def env_factory(d_speed=v_d, e_speed=v_e):
                config = EnvConfig(
                    v_d=d_speed,
                    v_e=e_speed,
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
            
            # Compute tracking error metrics (Kalman-specific)
            if "mean_tracking_error" in df.columns:
                valid_tracking = df["mean_tracking_error"].dropna()
                mean_tracking_error = valid_tracking.mean() if len(valid_tracking) > 0 else float("nan")
            else:
                mean_tracking_error = float("nan")
            
            results.append({
                "defender_speed": v_d,
                "enemy_speed": v_e,
                "speed_ratio": v_d / v_e if v_e > 0 else float("inf"),
                "n_episodes": n,
                "n_success": int(p * n),
                "success_rate": p,
                "success_se": se,
                "mean_episode_length": summary["mean_episode_length"],
                "failure_rate": summary["failure_rate"],
                "timeout_rate": summary["timeout_rate"],
                "mean_tracking_error": mean_tracking_error,
            })
            
            if verbose:
                track_str = f", track={mean_tracking_error:.2f}" if not np.isnan(mean_tracking_error) else ""
                print(f"success={p:.0%}{track_str}")
    
    if verbose:
        print("\n" + "-" * 60)
        print("Sweep complete!")
    
    return pd.DataFrame(results)


def plot_heatmap(df: pd.DataFrame, output_path: str, show: bool = False) -> None:
    """Generate success rate heatmap."""
    # Pivot data for heatmap
    pivot = df.pivot(
        index="defender_speed",
        columns="enemy_speed",
        values="success_rate"
    ) * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        pivot.values,
        cmap="RdYlGn",
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=100,
    )
    
    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{v:.0f}" for v in pivot.columns])
    ax.set_yticklabels([f"{v:.0f}" for v in pivot.index])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Success Rate (%)")
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val < 40 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                   color=color, fontsize=9, fontweight="bold")
    
    ax.set_xlabel("Enemy Speed (v_e)", fontsize=12)
    ax.set_ylabel("Defender Speed (v_d)", fontsize=12)
    ax.set_title("RL-Kalman Policy: Success Rate Heat Map\n(Defender Speed vs Enemy Speed)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_contour(df: pd.DataFrame, output_path: str, show: bool = False) -> None:
    """Generate success rate contour plot."""
    pivot = df.pivot(
        index="defender_speed",
        columns="enemy_speed",
        values="success_rate"
    ) * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values
    
    contour = ax.contourf(X, Y, Z, levels=10, cmap="RdYlGn", vmin=0, vmax=100)
    ax.contour(X, Y, Z, levels=[20, 40, 50, 60, 80], colors="black", linewidths=0.5)
    
    # Add v_d = v_e line
    min_speed = min(pivot.columns.min(), pivot.index.min())
    max_speed = max(pivot.columns.max(), pivot.index.max())
    ax.plot([min_speed, max_speed], [min_speed, max_speed], 'k--', 
            linewidth=2, alpha=0.7, label='v_d = v_e')
    
    plt.colorbar(contour, ax=ax, label="Success Rate (%)")
    
    ax.set_xlabel("Enemy Speed (v_e)", fontsize=12)
    ax.set_ylabel("Defender Speed (v_d)", fontsize=12)
    ax.set_title("RL-Kalman Policy: Success Rate Contours\n(Defender Speed vs Enemy Speed)", fontsize=14)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    grid_defaults = SWEEP_CONFIG["speed_grid"]
    default_defender = grid_defaults["defender_speeds"]
    default_enemy = grid_defaults["enemy_speeds"]
    default_episodes = grid_defaults["n_episodes"]
    
    parser = argparse.ArgumentParser(
        description="2D parameter sweep: speed grid (RL-Kalman)"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n-episodes", type=int, default=default_episodes)
    parser.add_argument(
        "--defender-speeds", type=str,
        default=",".join(str(s) for s in default_defender)
    )
    parser.add_argument(
        "--enemy-speeds", type=str,
        default=",".join(str(s) for s in default_enemy)
    )
    parser.add_argument("--seed-offset", type=int, default=CONFIG.SEED_OFFSET)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--process-var", type=float, default=PROCESS_VAR)
    parser.add_argument("--measurement-var", type=float, default=MEASUREMENT_VAR)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    defender_speeds = [float(s) for s in args.defender_speeds.split(",")]
    enemy_speeds = [float(s) for s in args.enemy_speeds.split(",")]
    
    df = sweep_speed_grid(
        model_path=args.model,
        defender_speeds=defender_speeds,
        enemy_speeds=enemy_speeds,
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
    
    csv_path = os.path.join(output_dir, 'sweep_speed_grid.csv')
    heatmap_path = os.path.join(output_dir, 'sweep_speed_grid_heatmap.png')
    contour_path = os.path.join(output_dir, 'sweep_speed_grid_contour.png')
    
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    print("\n" + "=" * 70)
    print("SWEEP RESULTS (RL-KALMAN)")
    print("=" * 70)
    print(f"{'v_d':>6} {'v_e':>6} {'Ratio':>8} {'Success':>10} {'SE':>8} {'Ep Len':>8}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['defender_speed']:>6.1f} {row['enemy_speed']:>6.1f} "
              f"{row['speed_ratio']:>8.2f} "
              f"{row['success_rate']*100:>9.1f}% {row['success_se']*100:>7.1f}% "
              f"{row['mean_episode_length']:>8.1f}")
    print("=" * 70)
    
    plot_heatmap(df, heatmap_path, show=args.show_plot)
    plot_contour(df, contour_path, show=args.show_plot)


if __name__ == "__main__":
    main()
