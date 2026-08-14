"""
Parameter sweep: Measurement noise vs success probability (RL-Kalman PPO).

Evaluates a trained RL-Kalman PPO policy with use_kalman_tracking=True across
shared measurement-noise values to test robustness under sensing uncertainty.

Noise values represent measurement variance (sigma^2), not standard deviation.

Usage:
    python experiments/rl_kalman/sweep_noise_rl_kalman.py
    python experiments/rl_kalman/sweep_noise_rl_kalman.py --model <path> --n-episodes 200
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
from uav_defend.policies.rl_kalman import PPOKalmanPolicyWrapper

from experiments.eval_utils import evaluate_policy
from experiments.experiment_config import (
    CONFIG,
    KALMAN_CONFIG,
    SWEEP_CONFIG,
    format_speeds_for_cli,
    parse_speeds_from_cli,
)
from experiments.noise_sweep_utils import (
    build_noise_summary_row,
    extract_tracking_metrics,
    make_eval_seeds,
)

POLICY_NAME = "rl_kalman"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "rl_kalman" / "models" / "ppo_kalman_final.zip"

# Shared Kalman defaults
PROCESS_VAR = KALMAN_CONFIG["process_var"]
LEAD_TIME = KALMAN_CONFIG["lead_time"]


def sweep_noise_rl_kalman(
    model_path: str,
    noise_values: list[float],
    n_episodes: int,
    seed_offset: int = 0,
    deterministic: bool = True,
    process_var: float = PROCESS_VAR,
    lead_time: float = LEAD_TIME,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a Monte Carlo noise sweep for a trained RL-Kalman PPO policy."""
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

    model_version = model_file.stem

    if verbose:
        print("=" * 84)
        print("PARAMETER SWEEP: MEASUREMENT NOISE (RL-Kalman PPO)")
        print("=" * 84)
        print(f"Model:                           {model_file}")
        print(f"Model version:                   {model_version}")
        print(f"Noise values (variance sigma^2): {noise_values}")
        print(f"Episodes per value:              {n_episodes}")
        print(f"Total episodes:                  {len(noise_values) * n_episodes}")
        print(f"Kalman fixed params:             process_var={process_var}, lead_time={lead_time}")
        print("-" * 84)

    policy = PPOKalmanPolicyWrapper.load(str(model_file), deterministic=deterministic)

    seeds = make_eval_seeds(seed_offset, n_episodes)
    results: list[dict] = []

    for noise_value in noise_values:
        if verbose:
            print(f"\nEvaluating measurement_var = {noise_value:.4g} ...")

        def env_factory(var=noise_value):
            config = EnvConfig(
                use_kalman_tracking=True,
                process_var=process_var,
                measurement_var=var,
                lead_time=lead_time,
            )
            return SoldierEnv(config=config)

        df, summary, _ = evaluate_policy(
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
        row.update(extract_tracking_metrics(df))
        results.append(row)

        if verbose:
            print(
                f"  success={row['success_rate']:.2%}, failure={row['failure_rate']:.2%}, "
                f"SE={row['standard_error']:.2%}, 95% CI=[{row['success_ci_low']:.2%}, {row['success_ci_high']:.2%}], "
                f"mean_track_err={row['mean_tracking_error']:.3f}, final_track_err={row['final_tracking_error']:.3f}"
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
        color="#E74C3C",
        ecolor="#4A4A4A",
        label="RL-Kalman PPO (95% CI)",
    )

    ax.set_xlabel("Measurement Noise Variance (sigma^2)", fontsize=12)
    ax.set_ylabel("Success Probability", fontsize=12)
    ax.set_title("RL-Kalman PPO: Success Probability vs Measurement Noise", fontsize=13)
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
        description="Parameter sweep: measurement noise vs success probability (RL-Kalman PPO)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to trained PPO-Kalman model (default: {DEFAULT_MODEL_PATH})",
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
        "--process-var",
        type=float,
        default=PROCESS_VAR,
        help=f"Kalman process variance (default: {PROCESS_VAR})",
    )
    parser.add_argument(
        "--lead-time",
        type=float,
        default=LEAD_TIME,
        help=f"Kalman lead time (default: {LEAD_TIME})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/rl_kalman)",
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

    df = sweep_noise_rl_kalman(
        model_path=args.model,
        noise_values=noise_values,
        n_episodes=args.n_episodes,
        seed_offset=args.seed_offset,
        deterministic=not args.stochastic,
        process_var=args.process_var,
        lead_time=args.lead_time,
        verbose=not args.quiet,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / "rl_kalman"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sweep_noise_rl_kalman.csv"
    plot_path = output_dir / "sweep_noise_rl_kalman.png"

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("\n" + "=" * 116)
    print("NOISE SWEEP RESULTS (RL-Kalman PPO)")
    print("=" * 116)
    print(
        f"{'noise':>8} {'success':>10} {'failure':>10} {'SE':>9} {'CI low':>10} {'CI high':>10} {'mean trk':>11} {'final trk':>11}"
    )
    print("-" * 116)
    for _, row in df.sort_values("noise_value").iterrows():
        print(
            f"{row['noise_value']:>8.3f} "
            f"{row['success_rate']*100:>9.2f}% "
            f"{row['failure_rate']*100:>9.2f}% "
            f"{row['standard_error']*100:>8.2f}% "
            f"{row['success_ci_low']*100:>9.2f}% "
            f"{row['success_ci_high']*100:>9.2f}% "
            f"{row['mean_tracking_error']:>11.3f} "
            f"{row['final_tracking_error']:>11.3f}"
        )
    print("=" * 116)

    plot_sweep(df, str(plot_path), show=args.show_plot)


if __name__ == "__main__":
    main()
