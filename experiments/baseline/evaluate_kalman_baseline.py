"""
Monte Carlo evaluation of the Kalman greedy baseline policy.

Runs N episodes with different seeds and collects performance statistics.
Results are saved to results/baseline/kalman_baseline_results.csv.

This is the Kalman-only baseline: it pairs hand-designed greedy pursuit
with Kalman-filtered state estimation, without any reinforcement learning.
Comparing against evaluate_baseline.py isolates the effect of state
estimation; comparing against the RL-Kalman method isolates the effect of
learned control.

Reuses the shared evaluation pipeline from experiments/eval_utils.py for
identical episode rollout semantics across all methods.

Usage:
    python experiments/baseline/evaluate_kalman_baseline.py [--n-episodes N] [--seed-offset OFFSET]
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.policies.baseline import KalmanGreedyInterceptPolicy

from experiments.eval_utils import (
    evaluate_policy,
    print_summary,
    summarize_results,
)
from experiments.experiment_config import (
    EVAL_CONFIG,
    get_eval_csv_filename,
    get_output_dir,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLICY_NAME = "kalman_baseline"
DEFAULT_N_EPISODES = EVAL_CONFIG["n_episodes"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.

    Args:
        successes: Number of positive outcomes.
        n: Total number of trials.
        z: Z-score for the desired confidence level (default 1.96 for 95%).

    Returns:
        (lower, upper) bounds as fractions in [0, 1].
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p_hat = successes / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))) / denom
    return float(centre - margin), float(centre + margin)


def _format_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the standard column schema for Kalman baseline results.

    Matches the CSV schema used by the other methods, extended with:
        - mean_tracking_error
        - final_tracking_error
        - policy_name

    Args:
        df: Raw results DataFrame from evaluate_policy().

    Returns:
        DataFrame with columns ordered per the comparison schema.
    """
    result = df.copy()
    result["num_episodes"] = len(df)
    result["policy_name"] = POLICY_NAME

    # Fill Kalman columns with NaN if absent (e.g. episodes without detection)
    if "mean_tracking_error" not in result.columns:
        result["mean_tracking_error"] = float("nan")
    if "final_tracking_error" not in result.columns:
        result["final_tracking_error"] = float("nan")

    columns = [
        "seed",
        "outcome",
        "success",
        "episode_length",
        "detected",
        "detection_time",
        "intercept_time",
        "enemy_speed",
        "defender_speed",
        "detection_radius",
        "intercept_radius",
        "threat_radius",
        "min_enemy_soldier_dist",
        "min_defender_enemy_dist",
        "num_episodes",
        "mean_tracking_error",
        "final_tracking_error",
        "policy_name",
    ]

    # Keep only schema columns; any extras from eval_utils are dropped
    return result[[c for c in columns if c in result.columns]]


def _print_kalman_summary(df: pd.DataFrame, summary: dict) -> None:
    """
    Print expanded summary that includes Kalman tracking metrics and
    a 95% Wilson confidence interval for the success rate.

    Args:
        df: Full results DataFrame (one row per episode).
        summary: Output of summarize_results().
    """
    n = summary["n_episodes"]
    oc = summary["outcome_counts"]

    n_success = oc.get("intercepted", 0)
    n_caught = oc.get("soldier_caught", 0)
    n_unsafe = oc.get("unsafe_intercept", 0)
    n_timeout = oc.get("timeout", 0)

    ci_lo, ci_hi = _wilson_ci(n_success, n)

    # Tracking error statistics (over episodes that had any tracked steps)
    tracked_df = df[df["mean_tracking_error"].notna()]
    if len(tracked_df) > 0:
        mean_te = tracked_df["mean_tracking_error"].mean()
        std_te = tracked_df["mean_tracking_error"].std()
        median_te = tracked_df["mean_tracking_error"].median()
    else:
        mean_te = std_te = median_te = float("nan")

    print("\n" + "=" * 65)
    print(f"KALMAN BASELINE POLICY SUMMARY  (n={n})")
    print("=" * 65)

    print(f"\nOutcome Distribution:")
    print(f"  Intercepted (WIN):       {n_success:5d}  ({n_success/n:6.1%})")
    print(f"  Soldier caught:          {n_caught:5d}  ({n_caught/n:6.1%})")
    print(f"  Unsafe intercept:        {n_unsafe:5d}  ({n_unsafe/n:6.1%})")
    print(f"  Timeout:                 {n_timeout:5d}  ({n_timeout/n:6.1%})")

    print(f"\nAggregate Rates:")
    print(f"  Success rate:            {summary['success_rate']:6.1%}")
    print(f"  95% CI (Wilson):        [{ci_lo:.1%}, {ci_hi:.1%}]")
    print(f"  Failure rate:            {summary['failure_rate']:6.1%}")
    print(f"  Timeout rate:            {summary['timeout_rate']:6.1%}")

    print(f"\nEpisode Length:")
    print(f"  Mean:                    {summary['mean_episode_length']:6.1f} steps")
    print(f"  Std:                     {summary['std_episode_length']:6.1f} steps")

    print(f"\nDetection:")
    print(f"  Detection rate:          {summary['detection_rate']:6.1%}")
    if not np.isnan(summary["mean_detection_time"]):
        print(f"  Mean detection time:     {summary['mean_detection_time']:6.1f} steps")

    print(f"\nIntercept (successful episodes only):")
    if not np.isnan(summary["mean_intercept_time"]):
        print(f"  Mean intercept time:     {summary['mean_intercept_time']:6.1f} steps")
    else:
        print(f"  Mean intercept time:     N/A")

    print(f"\nKalman Tracking Error (post-detection, {len(tracked_df)} episodes):")
    if not np.isnan(mean_te):
        print(f"  Mean  tracking error:    {mean_te:.4f} m")
        print(f"  Std   tracking error:    {std_te:.4f} m")
        print(f"  Median tracking error:   {median_te:.4f} m")
    else:
        print(f"  No tracked episodes.")

    print(f"\nDistance Metrics:")
    print(f"  Mean min d(enemy,soldier):   {summary['mean_min_enemy_soldier_dist']:.2f} m")
    print(f"  Mean min d(defender,enemy):  {summary['mean_min_defender_enemy_dist']:.2f} m")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo evaluation of the Kalman greedy baseline policy",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=DEFAULT_N_EPISODES,
        help=f"Number of episodes to run (default: {DEFAULT_N_EPISODES})",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=EVAL_CONFIG["seed_offset"],
        help=f"Starting seed for deterministic evaluation (default: {EVAL_CONFIG['seed_offset']})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: results/baseline/kalman_baseline_results.csv)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-episode progress output",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Environment and policy
    # ------------------------------------------------------------------
    config = EnvConfig(use_kalman_tracking=True)
    env_factory = lambda: SoldierEnv(config=config)
    policy = KalmanGreedyInterceptPolicy()

    seeds = range(args.seed_offset, args.seed_offset + args.n_episodes)

    # ------------------------------------------------------------------
    # Run evaluation via shared pipeline
    # ------------------------------------------------------------------
    df_raw, summary, _ = evaluate_policy(
        env_factory=env_factory,
        policy=policy,
        seeds=seeds,
        verbose=not args.quiet,
    )

    # ------------------------------------------------------------------
    # Print Kalman-extended summary
    # ------------------------------------------------------------------
    _print_kalman_summary(df_raw, summary)

    # ------------------------------------------------------------------
    # Format and save CSV
    # ------------------------------------------------------------------
    df_out = _format_df(df_raw)

    if args.output:
        output_path = args.output
    else:
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        output_dir = os.path.join(project_root, get_output_dir(POLICY_NAME))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, get_eval_csv_filename(POLICY_NAME))

    df_out.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Rows: {len(df_out)}  |  Columns: {list(df_out.columns)}")

    return df_out, summary


if __name__ == "__main__":
    main()
