"""
Parameter sweep: Defender speed vs success probability (Kalman baseline).

Evaluates the KalmanGreedyInterceptPolicy across a range of defender speeds
with use_kalman_tracking=True.  Directly comparable to:
    - experiments/baseline/sweep_defender_speed.py         (greedy, no Kalman)
    - experiments/rl/sweep_defender_speed_rl.py            (PPO Direct RL)
    - experiments/rl_kalman/sweep_defender_speed_rl_kalman.py (PPO RL-Kalman)

Usage:
    python experiments/baseline/sweep_defender_speed_kalman_baseline.py [--n-episodes N] [--speeds "10,12,14,16,18,20"]
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


def sweep_defender_speed(
    speeds: list[float],
    n_episodes: int = 200,
    seed_offset: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run defender-speed sweep with the Kalman greedy baseline.

    Args:
        speeds: Defender speeds to evaluate.
        n_episodes: Episodes per speed value.
        seed_offset: Starting seed.
        verbose: Print progress if True.

    Returns:
        DataFrame with one row per speed value, including tracking error columns.
    """
    if verbose:
        print("=" * 65)
        print("PARAMETER SWEEP: DEFENDER SPEED  [Kalman Baseline]")
        print("=" * 65)
        print(f"Speeds:       {speeds}")
        print(f"Episodes:     {n_episodes} per speed")
        print(f"Total runs:   {len(speeds) * n_episodes}")
        print("-" * 65)

    results = []
    seeds = range(seed_offset, seed_offset + n_episodes)

    for v_d in speeds:
        if verbose:
            print(f"\nEvaluating v_d = {v_d:.1f}...")

        def env_factory(speed=v_d):
            config = EnvConfig(v_d=speed, use_kalman_tracking=True)
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

        success_df = df[df["success"] == 1]
        mean_intercept_time = (
            success_df["intercept_time"].mean() if len(success_df) > 0 else float("nan")
        )

        # Kalman tracking error — averaged over episodes that had tracked steps
        tracked_df = df[df["mean_tracking_error"].notna()] if "mean_tracking_error" in df else pd.DataFrame()
        mean_tracking_error = tracked_df["mean_tracking_error"].mean() if len(tracked_df) > 0 else float("nan")

        results.append({
            "defender_speed": v_d,
            "enemy_speed": df["enemy_speed"].iloc[0] if len(df) > 0 else CONFIG.DEFAULT_ENEMY_SPEED,
            "speed_ratio": v_d / CONFIG.DEFAULT_ENEMY_SPEED,
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
            "policy_name": POLICY_KEY,
        })

        if verbose:
            te_str = f"  mean_tracking_error={mean_tracking_error:.3f} m" if not np.isnan(mean_tracking_error) else ""
            print(f"  Success rate: {p:.1%} ± {se:.1%}{te_str}")

    if verbose:
        print("-" * 65)
        print("Sweep complete!")

    return pd.DataFrame(results)


def plot_sweep(df: pd.DataFrame, output_path: str, show: bool = False) -> None:
    """
    Plot success rate vs defender speed with error bars.

    Styled consistently with the other method sweep plots for overlay comparisons.
    """
    fig, axes = plt.subplots(1, 2, figsize=(CONFIG.PLOT_FIGSIZE[0] * 1.4, CONFIG.PLOT_FIGSIZE[1]))

    enemy_speed = df["enemy_speed"].iloc[0]

    # --- Left: success rate ---
    ax = axes[0]
    ax.errorbar(
        df["defender_speed"],
        df["success_rate"] * 100,
        yerr=df["success_se"] * 100,
        fmt="o-",
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=8,
        color=METHOD_STYLE["color"],
        ecolor="#7F8C8D",
        label="Kalman Baseline",
    )
    ax.axvline(x=enemy_speed, color="red", linestyle="--", alpha=0.7,
               label=f"Enemy speed (v_e = {enemy_speed})")
    ax.set_xlabel("Defender Speed (v_d)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Kalman Baseline: Success Rate vs Defender Speed", fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_xlim(df["defender_speed"].min() - 1, df["defender_speed"].max() + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Secondary x-axis: speed ratio
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(df["defender_speed"])
    ax2.set_xticklabels([f"{v_d / enemy_speed:.2f}" for v_d in df["defender_speed"]])
    ax2.set_xlabel("Speed Ratio (v_d / v_e)", fontsize=10)

    # --- Right: mean tracking error ---
    ax3 = axes[1]
    tracked = df[df["mean_tracking_error"].notna()]
    if len(tracked) > 0:
        ax3.plot(
            tracked["defender_speed"],
            tracked["mean_tracking_error"],
            "s--",
            linewidth=2,
            markersize=8,
            color=METHOD_STYLE["color"],
            label="Mean tracking error",
        )
        ax3.set_xlabel("Defender Speed (v_d)", fontsize=12)
        ax3.set_ylabel("Mean Tracking Error (m)", fontsize=12)
        ax3.set_title("Kalman Baseline: Tracking Error vs Defender Speed", fontsize=13)
        ax3.set_ylim(bottom=0)
        ax3.set_xlim(df["defender_speed"].min() - 1, df["defender_speed"].max() + 1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No tracking error data", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.PLOT_DPI, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    sweep_defaults = SWEEP_CONFIG["defender_speed"]
    default_speeds = format_speeds_for_cli(sweep_defaults["parameter_values"])
    default_episodes = sweep_defaults["n_episodes"]

    parser = argparse.ArgumentParser(
        description="Parameter sweep: defender speed vs success rate (Kalman baseline)"
    )
    parser.add_argument("--n-episodes", type=int, default=default_episodes,
                        help=f"Episodes per speed value (default: {default_episodes})")
    parser.add_argument("--speeds", type=str, default=default_speeds,
                        help=f"Comma-separated speeds (default: {default_speeds})")
    parser.add_argument("--seed-offset", type=int, default=CONFIG.SEED_OFFSET,
                        help=f"Starting seed (default: {CONFIG.SEED_OFFSET})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/baseline/)")
    parser.add_argument("--show-plot", action="store_true", help="Display plot interactively")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()
    speeds = parse_speeds_from_cli(args.speeds)

    df = sweep_defender_speed(
        speeds=speeds,
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

    csv_path = os.path.join(output_dir, get_sweep_csv_filename("defender_speed", POLICY_KEY))
    plot_path = os.path.join(output_dir, get_sweep_plot_filename("defender_speed", POLICY_KEY))

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("\n" + "=" * 65)
    print("SWEEP RESULTS")
    print("=" * 65)
    print(f"{'Speed':>8} {'Ratio':>8} {'Success':>10} {'SE':>8} {'Track Err':>12} {'Ep Len':>8}")
    print("-" * 65)
    for _, row in df.iterrows():
        te_str = f"{row['mean_tracking_error']:.3f}" if not np.isnan(row.get("mean_tracking_error", float("nan"))) else "  N/A "
        print(f"{row['defender_speed']:>8.1f} {row['speed_ratio']:>8.2f} "
              f"{row['success_rate']*100:>9.1f}% {row['success_se']*100:>7.1f}%"
              f" {te_str:>12} {row['mean_episode_length']:>8.1f}")
    print("=" * 65)

    plot_sweep(df, plot_path, show=args.show_plot)
    return df


if __name__ == "__main__":
    main()
