"""
Parameter sweep: Measurement noise vs success probability (Direct RL PPO).

Evaluates a trained direct-RL PPO policy across shared measurement-noise
values to quantify sensitivity to sensing noise.

Noise values represent measurement variance (sigma^2), not standard deviation.

Usage:
    python experiments/rl/sweep_noise_rl.py
    python experiments/rl/sweep_noise_rl.py --model <path> --n-episodes 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.policies.rl import PPOPolicyWrapper

from experiments.eval_utils import evaluate_policy
from experiments.experiment_config import (
    CONFIG,
    SWEEP_CONFIG,
    format_speeds_for_cli,
    parse_speeds_from_cli,
)
from experiments.noise_sweep_utils import build_noise_summary_row, make_eval_seeds

POLICY_NAME = "rl"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "rl" / "models" / "ppo_defender_final.zip"


def sweep_noise_rl(
    model_path: str,
    noise_values: list[float],
    n_episodes: int,
    seed_offset: int = 0,
    deterministic: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a Monte Carlo noise sweep for a trained direct-RL PPO policy."""
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

    model_version = model_file.stem

    if verbose:
        print("=" * 76)
        print("PARAMETER SWEEP: MEASUREMENT NOISE (Direct RL PPO)")
        print("=" * 76)
        print(f"Model:                          {model_file}")
        print(f"Model version:                  {model_version}")
        print(f"Noise values (variance sigma^2): {noise_values}")
        print(f"Episodes per value:             {n_episodes}")
        print(f"Total episodes:                 {len(noise_values) * n_episodes}")
        print("-" * 76)

    policy = PPOPolicyWrapper.load(str(model_file), deterministic=deterministic)

    seeds = make_eval_seeds(seed_offset, n_episodes)
    results: list[dict] = []

    for noise_value in noise_values:
        if verbose:
            print(f"\nEvaluating measurement_var = {noise_value:.4g} ...")

        def env_factory(var=noise_value):
            # Direct RL track: no Kalman filter in the observation pipeline.
            config = EnvConfig(use_kalman_tracking=False, measurement_var=var)
            return SoldierEnv(config=config)

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
        row["model_version"] = model_version
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
        color="#9B59B6",
        ecolor="#4A4A4A",
        label="Direct RL PPO (95% CI)",
    )

    ax.set_xlabel("Measurement Noise Variance (sigma^2)", fontsize=12)
    ax.set_ylabel("Success Probability", fontsize=12)
    ax.set_title("Direct RL PPO: Success Probability vs Measurement Noise", fontsize=13)
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


def main() -> None:
    sweep_defaults = SWEEP_CONFIG["measurement_var"]
    default_noise_values = format_speeds_for_cli(sweep_defaults["parameter_values"])
    default_episodes = sweep_defaults["n_episodes"]

    parser = argparse.ArgumentParser(
        description="Parameter sweep: measurement noise vs success probability (Direct RL PPO)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to trained PPO model (default: {DEFAULT_MODEL_PATH})",
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
        help="Output directory (default: results/rl)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (default: deterministic)",
    )
    parser.add_argument("--show-plot", action="store_true", help="Display plot interactively")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    noise_values = parse_speeds_from_cli(args.noise_values)

    df = sweep_noise_rl(
        model_path=args.model,
        noise_values=noise_values,
        n_episodes=args.n_episodes,
        seed_offset=args.seed_offset,
        deterministic=not args.stochastic,
        verbose=not args.quiet,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / "rl"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sweep_noise_rl.csv"
    plot_path = output_dir / "sweep_noise_rl.png"

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("\n" + "=" * 95)
    print("NOISE SWEEP RESULTS (Direct RL PPO)")
    print("=" * 95)
    print(
        f"{'noise':>8} {'success':>10} {'failure':>10} {'SE':>9} {'CI low':>10} {'CI high':>10} {'ep len':>10}"
    )
    print("-" * 95)
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
    print("=" * 95)

    plot_sweep(df, str(plot_path), show=args.show_plot)


if __name__ == "__main__":
    main()
