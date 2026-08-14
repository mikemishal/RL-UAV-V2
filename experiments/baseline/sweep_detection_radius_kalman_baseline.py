"""
Parameter sweep: Detection radius vs success probability (Kalman baseline).

Evaluates the KalmanGreedyInterceptPolicy across a range of detection radii
with use_kalman_tracking=True.  Directly comparable to:
    - experiments/baseline/sweep_detection_radius.py             (greedy, no Kalman)
    - experiments/rl/sweep_detection_radius_rl.py                (PPO Direct RL)
    - experiments/rl_kalman/sweep_detection_radius_rl_kalman.py  (PPO RL-Kalman)

Usage:
    python experiments/baseline/sweep_detection_radius_kalman_baseline.py [--n-episodes N] [--radii "5,8,10,12,15"]
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


def sweep_detection_radius(
    radii: list[float],
    n_episodes: int = 200,
    seed_offset: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run detection-radius sweep with the Kalman greedy baseline.

    Args:
        radii: Detection radii to evaluate.
        n_episodes: Episodes per radius value.
        seed_offset: Starting seed.
        verbose: Print progress if True.

    Returns:
        DataFrame with one row per radius value, including tracking error columns.
    """
    if verbose:
        print("=" * 65)
        print("PARAMETER SWEEP: DETECTION RADIUS  [Kalman Baseline]")
        print("=" * 65)
        print(f"Radii:        {radii}")
        print(f"Episodes:     {n_episodes} per radius")
        print(f"Total runs:   {len(radii) * n_episodes}")
        print("-" * 65)

    results = []
    seeds = range(seed_offset, seed_offset + n_episodes)

    for r_det in radii:
        if verbose:
            print(f"\nEvaluating detection_radius = {r_det:.1f}...")

        def env_factory(radius=r_det):
            config = EnvConfig(detection_radius=radius, use_kalman_tracking=True)
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

        tracked_df = df[df["mean_tracking_error"].notna()] if "mean_tracking_error" in df else pd.DataFrame()
        mean_tracking_error = tracked_df["mean_tracking_error"].mean() if len(tracked_df) > 0 else float("nan")

        ref_config = EnvConfig(detection_radius=r_det)
        results.append({
            "detection_radius": r_det,
            "defender_speed": ref_config.v_d,
            "enemy_speed": ref_config.v_e,
            "intercept_radius": ref_config.intercept_radius,
            "det_intercept_ratio": r_det / ref_config.intercept_radius,
            "n_episodes": n,
            "n_success": int(p * n),
            "success_rate": p,
            "success_se": se,
            "failure_rate": summary["failure_rate"],
            "timeout_rate": summary["timeout_rate"],
            "mean_episode_length": summary["mean_episode_length"],
            "std_episode_length": summary["std_episode_length"],
            "detection_rate": summary["detection_rate"],
            "mean_detection_time": summary["mean_detection_time"],
            "mean_intercept_time": mean_intercept_time,
            "mean_min_enemy_soldier_dist": summary["mean_min_enemy_soldier_dist"],
            "mean_tracking_error": mean_tracking_error,
            "policy_name": POLICY_KEY,
        })

        if verbose:
            te_str = f"  mean_tracking_error={mean_tracking_error:.3f} m" if not np.isnan(mean_tracking_error) else ""
            print(f"  Success rate: {p:.1%} ± {se:.1%}  "
                  f"det_rate={summary['detection_rate']:.1%}{te_str}")

    if verbose:
        print("-" * 65)
        print("Sweep complete!")

    return pd.DataFrame(results)


def plot_sweep(df: pd.DataFrame, output_path: str, show: bool = False) -> None:
    """Plot success rate vs detection radius with error bars and tracking error panel."""
    fig, axes = plt.subplots(1, 2, figsize=(CONFIG.PLOT_FIGSIZE[0] * 1.4, CONFIG.PLOT_FIGSIZE[1]))

    intercept_radius = df["intercept_radius"].iloc[0]

    # --- Left: success rate ---
    ax = axes[0]
    ax.errorbar(
        df["detection_radius"],
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
    ax.axvline(x=intercept_radius, color="red", linestyle="--", alpha=0.7,
               label=f"Intercept radius (r_int = {intercept_radius})")
    ax.set_xlabel("Detection Radius (r_det)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Kalman Baseline: Success Rate vs Detection Radius", fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_xlim(df["detection_radius"].min() - 1, df["detection_radius"].max() + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(df["detection_radius"])
    ax2.set_xticklabels([f"{r / intercept_radius:.1f}" for r in df["detection_radius"]])
    ax2.set_xlabel("Detection / Intercept Ratio (r_det / r_int)", fontsize=10)

    # --- Right: tracking error vs detection radius ---
    ax3 = axes[1]
    tracked = df[df["mean_tracking_error"].notna()]
    if len(tracked) > 0:
        ax3.plot(
            tracked["detection_radius"],
            tracked["mean_tracking_error"],
            "s--",
            linewidth=2,
            markersize=8,
            color=METHOD_STYLE["color"],
            label="Mean tracking error",
        )
        ax3.set_xlabel("Detection Radius (r_det)", fontsize=12)
        ax3.set_ylabel("Mean Tracking Error (m)", fontsize=12)
        ax3.set_title("Kalman Baseline: Tracking Error vs Detection Radius", fontsize=13)
        ax3.set_ylim(bottom=0)
        ax3.set_xlim(df["detection_radius"].min() - 1, df["detection_radius"].max() + 1)
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
    sweep_defaults = SWEEP_CONFIG["detection_radius"]
    default_radii = format_speeds_for_cli(sweep_defaults["parameter_values"])
    default_episodes = sweep_defaults["n_episodes"]

    parser = argparse.ArgumentParser(
        description="Parameter sweep: detection radius vs success rate (Kalman baseline)"
    )
    parser.add_argument("--n-episodes", type=int, default=default_episodes,
                        help=f"Episodes per radius value (default: {default_episodes})")
    parser.add_argument("--radii", type=str, default=default_radii,
                        help=f"Comma-separated radii (default: {default_radii})")
    parser.add_argument("--seed-offset", type=int, default=CONFIG.SEED_OFFSET,
                        help=f"Starting seed (default: {CONFIG.SEED_OFFSET})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/baseline/)")
    parser.add_argument("--show-plot", action="store_true", help="Display plot interactively")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()
    radii = parse_speeds_from_cli(args.radii)

    df = sweep_detection_radius(
        radii=radii,
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

    csv_path = os.path.join(output_dir, get_sweep_csv_filename("detection_radius", POLICY_KEY))
    plot_path = os.path.join(output_dir, get_sweep_plot_filename("detection_radius", POLICY_KEY))

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("\n" + "=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)
    print(f"{'Radius':>8} {'Det Rate':>10} {'Success':>10} {'SE':>8} {'Track Err':>12} {'Ep Len':>8}")
    print("-" * 70)
    for _, row in df.iterrows():
        te_str = f"{row['mean_tracking_error']:.3f}" if not np.isnan(row.get("mean_tracking_error", float("nan"))) else "  N/A "
        print(f"{row['detection_radius']:>8.1f} {row['detection_rate']*100:>9.1f}% "
              f"{row['success_rate']*100:>9.1f}% {row['success_se']*100:>7.1f}%"
              f" {te_str:>12} {row['mean_episode_length']:>8.1f}")
    print("=" * 70)

    plot_sweep(df, plot_path, show=args.show_plot)
    return df


if __name__ == "__main__":
    main()
