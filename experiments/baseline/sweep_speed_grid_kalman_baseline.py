"""
2D Parameter sweep: Defender speed × Enemy speed grid (Kalman baseline).

Evaluates the KalmanGreedyInterceptPolicy over every (v_d, v_e) combination
with use_kalman_tracking=True.  Directly comparable to:
    - experiments/baseline/sweep_speed_grid.py              (greedy, no Kalman)
    - experiments/rl/sweep_speed_grid_rl.py                 (PPO Direct RL)
    - experiments/rl_kalman/sweep_speed_grid_rl_kalman.py   (PPO RL-Kalman)

Usage:
    python experiments/baseline/sweep_speed_grid_kalman_baseline.py [--n-episodes N]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from uav_defend.envs import SoldierEnv
from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.baseline import KalmanGreedyInterceptPolicy

from experiments.eval_utils import evaluate_policy
from experiments.experiment_config import (
    CONFIG,
    SWEEP_CONFIG,
    get_method_style,
    get_output_dir,
    get_sweep_csv_filename,
    get_sweep_plot_filename,
    format_speeds_for_cli,
    parse_speeds_from_cli,
)

POLICY_KEY = "kalman_baseline"
METHOD_STYLE = get_method_style(POLICY_KEY)


def sweep_speed_grid(
    defender_speeds: list[float],
    enemy_speeds: list[float],
    n_episodes: int = 100,
    seed_offset: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run 2D defender × enemy speed sweep with the Kalman greedy baseline.

    Args:
        defender_speeds: Defender speeds to evaluate.
        enemy_speeds: Enemy speeds to evaluate.
        n_episodes: Episodes per grid point.
        seed_offset: Starting seed.
        verbose: Print progress if True.

    Returns:
        DataFrame with one row per (defender_speed, enemy_speed) pair,
        including mean_tracking_error.
    """
    n_grid = len(defender_speeds) * len(enemy_speeds)

    if verbose:
        print("=" * 65)
        print("2D PARAMETER SWEEP: DEFENDER SPEED × ENEMY SPEED  [Kalman Baseline]")
        print("=" * 65)
        print(f"Defender speeds: {defender_speeds}")
        print(f"Enemy speeds:    {enemy_speeds}")
        print(f"Grid points:     {n_grid}")
        print(f"Episodes/point:  {n_episodes}")
        print(f"Total runs:      {n_grid * n_episodes}")
        print("-" * 65)

    results = []
    seeds = range(seed_offset, seed_offset + n_episodes)
    grid_idx = 0

    for v_d in defender_speeds:
        for v_e in enemy_speeds:
            grid_idx += 1

            if verbose:
                print(f"[{grid_idx:3d}/{n_grid}] v_d={v_d:.1f}, v_e={v_e:.1f}...", end="  ")

            def env_factory(d_speed=v_d, e_speed=v_e):
                config = EnvConfig(v_d=d_speed, v_e=e_speed, use_kalman_tracking=True)
                return SoldierEnv(config=config)

            policy = KalmanGreedyInterceptPolicy()

            df, summary, _ = evaluate_policy(
                env_factory=env_factory,
                policy=policy,
                seeds=seeds,
                verbose=False,
            )

            p = summary["success_rate"]
            n = summary["n_episodes"]
            se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0

            tracked_df = df[df["mean_tracking_error"].notna()] if "mean_tracking_error" in df else pd.DataFrame()
            mean_tracking_error = tracked_df["mean_tracking_error"].mean() if len(tracked_df) > 0 else float("nan")

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
                "mean_tracking_error": mean_tracking_error,
                "policy_name": POLICY_KEY,
            })

            if verbose:
                print(f"success={p:.1%}")

    if verbose:
        print("\n" + "-" * 65)
        print("Sweep complete!")

    return pd.DataFrame(results)


def plot_heatmap(df: pd.DataFrame, output_path: str, show: bool = False) -> None:
    """
    Generate success rate heatmap with an optional tracking error overlay.

    Produces a two-panel figure:
      - Left:  Success rate heatmap (matches style of other method grids).
      - Right: Mean tracking error heatmap (Kalman-specific diagnostic).
    """
    colors = ["#E94F37", "#F2C14E", "#44AF69"]
    cmap_success = LinearSegmentedColormap.from_list("success", colors)

    pivot_success = df.pivot(
        index="defender_speed", columns="enemy_speed", values="success_rate"
    ).sort_index(ascending=False)

    has_tracking = "mean_tracking_error" in df.columns and df["mean_tracking_error"].notna().any()
    if has_tracking:
        pivot_tracking = df.pivot(
            index="defender_speed", columns="enemy_speed", values="mean_tracking_error"
        ).sort_index(ascending=False)

    ncols = 2 if has_tracking else 1
    fig, axes = plt.subplots(1, ncols, figsize=(CONFIG.GRID_FIGSIZE[0] * ncols / 2, 8))
    if ncols == 1:
        axes = [axes]

    # --- Success rate heatmap ---
    ax = axes[0]
    im = ax.imshow(
        pivot_success.values * 100,
        aspect="auto",
        cmap=cmap_success,
        vmin=0,
        vmax=100,
    )
    ax.set_xticks(range(len(pivot_success.columns)))
    ax.set_xticklabels([f"{v:.0f}" for v in pivot_success.columns])
    ax.set_yticks(range(len(pivot_success.index)))
    ax.set_yticklabels([f"{v:.0f}" for v in pivot_success.index])

    for i, v_d in enumerate(pivot_success.index):
        for j, v_e in enumerate(pivot_success.columns):
            val = pivot_success.loc[v_d, v_e] * 100
            text_color = "white" if val < 40 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    ax.set_xlabel("Enemy Speed (v_e)", fontsize=12)
    ax.set_ylabel("Defender Speed (v_d)", fontsize=12)
    ax.set_title("Kalman Baseline: Success Rate\nDefender Speed vs Enemy Speed", fontsize=13)
    plt.colorbar(im, ax=ax, label="Success Rate (%)")

    # --- Tracking error heatmap ---
    if has_tracking:
        ax2 = axes[1]
        im2 = ax2.imshow(
            pivot_tracking.values,
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
        )
        ax2.set_xticks(range(len(pivot_tracking.columns)))
        ax2.set_xticklabels([f"{v:.0f}" for v in pivot_tracking.columns])
        ax2.set_yticks(range(len(pivot_tracking.index)))
        ax2.set_yticklabels([f"{v:.0f}" for v in pivot_tracking.index])

        for i, v_d in enumerate(pivot_tracking.index):
            for j, v_e in enumerate(pivot_tracking.columns):
                val = pivot_tracking.loc[v_d, v_e]
                if not np.isnan(val):
                    ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                             fontsize=9, fontweight="bold", color="black")

        ax2.set_xlabel("Enemy Speed (v_e)", fontsize=12)
        ax2.set_ylabel("Defender Speed (v_d)", fontsize=12)
        ax2.set_title("Kalman Baseline: Mean Tracking Error (m)\nDefender Speed vs Enemy Speed", fontsize=13)
        plt.colorbar(im2, ax=ax2, label="Mean Tracking Error (m)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.PLOT_DPI, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_contour(df: pd.DataFrame, output_path: str, show: bool = False) -> None:
    """Generate contour plot of success rate over the speed grid."""
    pivot = df.pivot(index="defender_speed", columns="enemy_speed", values="success_rate")

    fig, ax = plt.subplots(figsize=(CONFIG.GRID_FIGSIZE[0], 8))
    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values * 100

    levels = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    cs = ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=9, fmt="%d%%")
    cf = ax.contourf(X, Y, Z, levels=20, cmap="RdYlGn", vmin=0, vmax=100)

    min_speed = min(pivot.columns.min(), pivot.index.min())
    max_speed = max(pivot.columns.max(), pivot.index.max())
    ax.plot([min_speed, max_speed], [min_speed, max_speed],
            "k--", linewidth=2, alpha=0.7, label="v_d = v_e")

    ax.set_xlabel("Enemy Speed (v_e)", fontsize=12)
    ax.set_ylabel("Defender Speed (v_d)", fontsize=12)
    ax.set_title("Kalman Baseline: Success Rate Contours\n(Defender Speed vs Enemy Speed)", fontsize=13)
    ax.legend(loc="lower right")
    plt.colorbar(cf, ax=ax, label="Success Rate (%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.PLOT_DPI, bbox_inches="tight")
    print(f"Contour plot saved to: {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    grid_defaults = SWEEP_CONFIG["speed_grid"]
    default_defender = format_speeds_for_cli(grid_defaults["defender_speeds"])
    default_enemy = format_speeds_for_cli(grid_defaults["enemy_speeds"])
    default_episodes = grid_defaults["n_episodes"]

    parser = argparse.ArgumentParser(
        description="2D speed grid sweep (Kalman baseline)"
    )
    parser.add_argument("--n-episodes", type=int, default=default_episodes,
                        help=f"Episodes per grid point (default: {default_episodes})")
    parser.add_argument("--defender-speeds", type=str, default=default_defender,
                        help=f"Comma-separated defender speeds (default: {default_defender})")
    parser.add_argument("--enemy-speeds", type=str, default=default_enemy,
                        help=f"Comma-separated enemy speeds (default: {default_enemy})")
    parser.add_argument("--seed-offset", type=int, default=CONFIG.SEED_OFFSET,
                        help=f"Starting seed (default: {CONFIG.SEED_OFFSET})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/baseline/)")
    parser.add_argument("--show-plot", action="store_true", help="Display plots interactively")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()
    defender_speeds = parse_speeds_from_cli(args.defender_speeds)
    enemy_speeds = parse_speeds_from_cli(args.enemy_speeds)

    df = sweep_speed_grid(
        defender_speeds=defender_speeds,
        enemy_speeds=enemy_speeds,
        n_episodes=args.n_episodes,
        seed_offset=args.seed_offset,
        verbose=not args.quiet,
    )

    if args.output_dir:
        output_dir = args.output_dir
    else:
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        output_dir = os.path.join(project_root, get_output_dir(POLICY_KEY))
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, get_sweep_csv_filename("speed_grid", POLICY_KEY))
    heatmap_path = os.path.join(output_dir, get_sweep_plot_filename("speed_grid", POLICY_KEY, variant="heatmap"))
    contour_path = os.path.join(output_dir, get_sweep_plot_filename("speed_grid", POLICY_KEY, variant="contour"))

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("\n" + "=" * 70)
    print("SWEEP RESULTS (sample - first 10 rows)")
    print("=" * 70)
    print(f"{'v_d':>6} {'v_e':>6} {'Ratio':>7} {'Success':>10} {'SE':>8} {'Track Err':>12}")
    print("-" * 70)
    for _, row in df.head(10).iterrows():
        te_str = f"{row['mean_tracking_error']:.3f}" if not np.isnan(row.get("mean_tracking_error", float("nan"))) else "  N/A "
        print(f"{row['defender_speed']:>6.1f} {row['enemy_speed']:>6.1f} "
              f"{row['speed_ratio']:>7.2f} {row['success_rate']*100:>9.1f}% "
              f"{row['standard_error']*100:>7.1f}% {te_str:>12}")
    print("=" * 70)

    plot_heatmap(df, heatmap_path, show=args.show_plot)
    plot_contour(df, contour_path, show=args.show_plot)
    return df


if __name__ == "__main__":
    main()
