"""
2D Parameter sweep: Defender speed vs Enemy speed grid.

Evaluates the greedy baseline policy across a 2D grid of speed combinations
to map the feasibility region for successful interception.

Usage:
    python experiments/baseline/sweep_speed_grid.py [--n-episodes N]
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
from matplotlib.colors import LinearSegmentedColormap

from uav_defend.envs import SoldierEnv
from uav_defend.config.env_config import EnvConfig
from uav_defend.policies import GreedyInterceptPolicy

from experiments.eval_utils import evaluate_policy
from experiments.experiment_config import (
    CONFIG,
    SWEEP_CONFIG,
    format_speeds_for_cli,
    parse_speeds_from_cli,
)


def sweep_speed_grid(
    defender_speeds: list[float],
    enemy_speeds: list[float],
    n_episodes: int = 100,
    seed_offset: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run 2D parameter sweep over defender and enemy speeds.
    
    Args:
        defender_speeds: List of defender speeds to evaluate.
        enemy_speeds: List of enemy speeds to evaluate.
        n_episodes: Number of episodes per grid point.
        seed_offset: Starting seed for reproducibility.
        verbose: If True, print progress.
    
    Returns:
        DataFrame with one row per (defender_speed, enemy_speed) combination.
    """
    n_grid = len(defender_speeds) * len(enemy_speeds)
    
    if verbose:
        print("=" * 60)
        print("2D PARAMETER SWEEP: DEFENDER SPEED VS ENEMY SPEED")
        print("=" * 60)
        print(f"Defender speeds: {defender_speeds}")
        print(f"Enemy speeds:    {enemy_speeds}")
        print(f"Grid points:     {n_grid}")
        print(f"Episodes/point:  {n_episodes}")
        print(f"Total runs:      {n_grid * n_episodes}")
        print("-" * 60)
    
    results = []
    seeds = range(seed_offset, seed_offset + n_episodes)
    
    grid_idx = 0
    for v_d in defender_speeds:
        for v_e in enemy_speeds:
            grid_idx += 1
            
            if verbose:
                print(f"\n[{grid_idx}/{n_grid}] v_d={v_d:.1f}, v_e={v_e:.1f}...", end=" ")
            
            # Create environment factory with custom config
            def env_factory(d_speed=v_d, e_speed=v_e):
                config = EnvConfig(v_d=d_speed, v_e=e_speed)
                return SoldierEnv(config=config)
            
            policy = GreedyInterceptPolicy()
            
            # Run evaluation
            df, summary = evaluate_policy(
                env_factory=env_factory,
                policy=policy,
                seeds=seeds,
                verbose=False,
            )
            
            # Compute standard error of success rate
            p = summary["success_rate"]
            n = summary["n_episodes"]
            se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
            
            # Store results
            results.append({
                "defender_speed": v_d,
                "enemy_speed": v_e,
                "speed_ratio": v_d / v_e,
                "n_episodes": n,
                "n_success": int(p * n),
                "success_rate": p,
                "standard_error": se,
                "failure_rate": summary["failure_rate"],
                "timeout_rate": summary["timeout_rate"],
                "mean_episode_length": summary["mean_episode_length"],
                "detection_rate": summary["detection_rate"],
            })
            
            if verbose:
                print(f"success={p:.1%}")
    
    if verbose:
        print("\n" + "-" * 60)
        print("Sweep complete!")
    
    return pd.DataFrame(results)


def plot_heatmap(df: pd.DataFrame, output_path: str, show: bool = False) -> None:
    """
    Generate heatmap of success rate over speed grid.
    
    Args:
        df: DataFrame from sweep_speed_grid().
        output_path: Path to save the plot.
        show: If True, display the plot interactively.
    """
    # Pivot data for heatmap
    pivot = df.pivot(
        index="defender_speed",
        columns="enemy_speed",
        values="success_rate"
    )
    
    # Sort indices for proper display (defender speed high at top)
    pivot = pivot.sort_index(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap (red -> yellow -> green)
    colors = ['#E94F37', '#F2C14E', '#44AF69']
    cmap = LinearSegmentedColormap.from_list('success', colors)
    
    # Plot heatmap
    im = ax.imshow(
        pivot.values * 100,  # Convert to percentage
        aspect='auto',
        cmap=cmap,
        vmin=0,
        vmax=100,
    )
    
    # Set tick labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{v:.0f}' for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{v:.0f}' for v in pivot.index])
    
    # Add value annotations
    for i, v_d in enumerate(pivot.index):
        for j, v_e in enumerate(pivot.columns):
            value = pivot.loc[v_d, v_e] * 100
            # Choose text color based on background
            text_color = 'white' if value < 40 or value > 70 else 'black'
            ax.text(j, i, f'{value:.0f}%',
                   ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   color=text_color)
    
    # Labels
    ax.set_xlabel('Enemy Speed (v_e)', fontsize=12)
    ax.set_ylabel('Defender Speed (v_d)', fontsize=12)
    ax.set_title('Greedy Baseline: Success Rate Heat Map\n(Defender Speed vs Enemy Speed)', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Success Rate (%)')
    cbar.ax.tick_params(labelsize=10)
    
    # Add diagonal reference line where v_d = v_e
    # Find positions where speeds are equal
    defender_speeds = list(pivot.index)
    enemy_speeds = list(pivot.columns)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_contour(df: pd.DataFrame, output_path: str, show: bool = False) -> None:
    """
    Generate contour plot of success rate over speed grid.
    
    Args:
        df: DataFrame from sweep_speed_grid().
        output_path: Path to save the plot.
        show: If True, display the plot interactively.
    """
    # Pivot data
    pivot = df.pivot(
        index="defender_speed",
        columns="enemy_speed",
        values="success_rate"
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid
    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values * 100
    
    # Contour plot
    levels = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    cs = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=9, fmt='%d%%')
    
    # Filled contours
    cf = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn', vmin=0, vmax=100)
    
    # Add v_d = v_e line
    min_speed = min(pivot.columns.min(), pivot.index.min())
    max_speed = max(pivot.columns.max(), pivot.index.max())
    ax.plot([min_speed, max_speed], [min_speed, max_speed], 
            'k--', linewidth=2, alpha=0.7, label='v_d = v_e')
    
    # Labels
    ax.set_xlabel('Enemy Speed (v_e)', fontsize=12)
    ax.set_ylabel('Defender Speed (v_d)', fontsize=12)
    ax.set_title('Greedy Baseline: Success Rate Contours\n(Defender Speed vs Enemy Speed)', fontsize=14)
    ax.legend(loc='lower right')
    
    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, label='Success Rate (%)')
    
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
    grid_defaults = SWEEP_CONFIG["speed_grid"]
    default_defender = format_speeds_for_cli(grid_defaults["defender_speeds"])
    default_enemy = format_speeds_for_cli(grid_defaults["enemy_speeds"])
    default_episodes = grid_defaults["n_episodes"]
    
    parser = argparse.ArgumentParser(
        description="2D parameter sweep: defender speed vs enemy speed"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=default_episodes,
        help=f"Number of episodes per grid point (default: {default_episodes})"
    )
    parser.add_argument(
        "--defender-speeds", type=str, default=default_defender,
        help=f"Comma-separated list of defender speeds (default: {default_defender})"
    )
    parser.add_argument(
        "--enemy-speeds", type=str, default=default_enemy,
        help=f"Comma-separated list of enemy speeds (default: {default_enemy})"
    )
    parser.add_argument(
        "--seed-offset", type=int, default=CONFIG.SEED_OFFSET,
        help=f"Starting seed for reproducibility (default: {CONFIG.SEED_OFFSET})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/baseline/)"
    )
    parser.add_argument(
        "--show-plot", action="store_true",
        help="Display plots interactively"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Parse speeds using shared utility
    defender_speeds = parse_speeds_from_cli(args.defender_speeds)
    enemy_speeds = parse_speeds_from_cli(args.enemy_speeds)
    
    # Run sweep
    df = sweep_speed_grid(
        defender_speeds=defender_speeds,
        enemy_speeds=enemy_speeds,
        n_episodes=args.n_episodes,
        seed_offset=args.seed_offset,
        verbose=not args.quiet,
    )
    
    # Set up output paths
    if args.output_dir:
        output_dir = args.output_dir
    else:
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        output_dir = os.path.join(project_root, 'results', 'baseline')
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'sweep_speed_grid.csv')
    heatmap_path = os.path.join(output_dir, 'sweep_speed_grid_heatmap.png')
    contour_path = os.path.join(output_dir, 'sweep_speed_grid_contour.png')
    
    # Save results
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SWEEP RESULTS (Success Rate %)")
    print("=" * 70)
    
    # Pivot for display
    pivot = df.pivot(
        index="defender_speed",
        columns="enemy_speed",
        values="success_rate"
    ).sort_index(ascending=False)
    
    # Print header
    header = "v_d \\ v_e |" + "".join([f" {v:>6.0f}" for v in pivot.columns])
    print(header)
    print("-" * len(header))
    
    for v_d in pivot.index:
        row = f"    {v_d:>5.0f} |"
        for v_e in pivot.columns:
            row += f" {pivot.loc[v_d, v_e]*100:>5.0f}%"
        print(row)
    
    print("=" * 70)
    
    # Generate plots
    plot_heatmap(df, heatmap_path, show=args.show_plot)
    plot_contour(df, contour_path, show=args.show_plot)
    
    return df


if __name__ == "__main__":
    main()
