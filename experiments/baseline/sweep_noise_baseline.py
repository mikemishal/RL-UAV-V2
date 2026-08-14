"""
Parameter sweep: Measurement noise vs success probability (Greedy baseline).

Evaluates the standard GreedyInterceptPolicy across shared measurement-noise
values. Noise values represent measurement variance (sigma^2), not standard
 deviation.

Usage:
    python experiments/baseline/sweep_noise_baseline.py [--n-episodes N] [--noise-values "0.0,0.1,0.25,0.5,1.0,2.0"]

Outputs:
    - CSV:  results/baseline/sweep_noise_baseline.csv
    - Plot: results/baseline/sweep_noise_baseline.png
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.policies import GreedyInterceptPolicy

from experiments.eval_utils import evaluate_policy
from experiments.experiment_config import (
    CONFIG,
    SWEEP_CONFIG,
    format_speeds_for_cli,
    parse_speeds_from_cli,
)
from experiments.noise_sweep_utils import build_noise_summary_row, make_eval_seeds

POLICY_NAME = "baseline"


def sweep_noise_baseline(
    noise_values: list[float],
    n_episodes: int,
    seed_offset: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a Monte Carlo noise sweep for the greedy baseline policy."""
    if verbose:
        print("=" * 68)
        print("PARAMETER SWEEP: MEASUREMENT NOISE (Greedy Baseline)")
        print("=" * 68)
        print("Noise values (variance sigma^2):", noise_values)
        print(f"Episodes per value:             {n_episodes}")
        print(f"Total episodes:                 {len(noise_values) * n_episodes}")
        print("-" * 68)

    results: list[dict] = []
    seeds = make_eval_seeds(seed_offset, n_episodes)

    for noise_value in noise_values:
        if verbose:
            print(f"\nEvaluating measurement_var = {noise_value:.4g} ...")

        def env_factory(var=noise_value):
            # Standard greedy baseline track: no Kalman filtering.
            config = EnvConfig(use_kalman_tracking=False, measurement_var=var)
            return SoldierEnv(config=config)

        policy = GreedyInterceptPolicy()

        _, summary, _ = evaluate_policy(
            env_factory=env_factory,
            policy=policy,
            seeds=seeds,
            verbose=False,
        )

        row = build_noise_summary_row(
            policy_name=POLICY_NAME,
            noise_value=noise_value,
            summary=summary,
        )
        results.append(row)

        if verbose:
            print(
                f"  success={row['success_rate']:.2%}, failure={row['failure_rate']:.2%}, "
                f"SE={row['standard_error']:.2%}, 95% CI=[{row['success_ci_low']:.2%}, {row['success_ci_high']:.2%}]"
            )

    return pd.DataFrame(results)


def plot_sweep(df: pd.DataFrame, output_path: str, show: bool = False) -> None:
    """Plot success probability vs measurement noise with 95% CI error bars."""
    df_sorted = df.sort_values("noise_value").reset_index(drop=True)

    success = df_sorted["success_rate"].to_numpy()
    ci_low = df_sorted["success_ci_low"].to_numpy()
    ci_high = df_sorted["success_ci_high"].to_numpy()

    yerr = np.vstack((success - ci_low, ci_high - success))

    fig, ax = plt.subplots(figsize=CONFIG.PLOT_FIGSIZE)
    ax.errorbar(
        df_sorted["noise_value"],
        success,
        yerr=yerr,
        fmt="o-",
        capsize=5,
        capthick=1.5,
        linewidth=2,
        markersize=7,
        color="#2E86AB",
        ecolor="#4A4A4A",
        label="Greedy Baseline (95% CI)",
    )

    ax.set_xlabel("Measurement Noise Variance (sigma^2)", fontsize=12)
    ax.set_ylabel("Success Probability", fontsize=12)
    ax.set_title("Greedy Baseline: Success Probability vs Measurement Noise", fontsize=13)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.PLOT_DPI, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    sweep_defaults = SWEEP_CONFIG["measurement_var"]
    default_noise_values = format_speeds_for_cli(sweep_defaults["parameter_values"])
    default_episodes = sweep_defaults["n_episodes"]

    parser = argparse.ArgumentParser(
        description="Parameter sweep: measurement noise vs success probability (greedy baseline)"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=default_episodes,
        help=(
            f"Episodes per noise value (default: {default_episodes}). "
            f"For final runs use {CONFIG.NUM_EPISODES}."
        ),
    )
    parser.add_argument(
        "--noise-values",
        type=str,
        default=default_noise_values,
        help=f"Comma-separated noise variance values (default: {default_noise_values})",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=CONFIG.SEED_OFFSET,
        help=f"Starting seed (default: {CONFIG.SEED_OFFSET})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/baseline)",
    )
    parser.add_argument("--show-plot", action="store_true", help="Display plot interactively")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    noise_values = parse_speeds_from_cli(args.noise_values)

    df = sweep_noise_baseline(
        noise_values=noise_values,
        n_episodes=args.n_episodes,
        seed_offset=args.seed_offset,
        verbose=not args.quiet,
    )

    if args.output_dir:
        output_dir = args.output_dir
    else:
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")
        output_dir = os.path.join(project_root, "results", "baseline")

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "sweep_noise_baseline.csv")
    plot_path = os.path.join(output_dir, "sweep_noise_baseline.png")

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("\n" + "=" * 88)
    print("NOISE SWEEP RESULTS (Greedy Baseline)")
    print("=" * 88)
    print(
        f"{'noise':>8} {'success':>10} {'failure':>10} {'SE':>9} {'CI low':>10} {'CI high':>10} {'ep len':>10}"
    )
    print("-" * 88)
    for _, row in df.sort_values("noise_value").iterrows():
        print(
            f"{row['noise_value']:>8.3f} "
            f"{row['success_rate']*100:>9.2f}% "
            f"{row['failure_rate']*100:>9.2f}% "
            f"{row['standard_error']*100:>8.2f}% "
            f"{row['success_ci_low']*100:>9.2f}% "
            f"{row['success_ci_high']*100:>9.2f}% "
            f"{row['mean_episode_length']:>10.2f}"
        )
    print("=" * 88)

    plot_sweep(df, plot_path, show=args.show_plot)


if __name__ == "__main__":
    main()
