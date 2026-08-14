"""
Monte Carlo evaluation of RL-Kalman (PPO) policies.

=============================================================================
EXPERIMENT TRACK: RL WITH KALMAN-ESTIMATED ENEMY STATE
=============================================================================

This script evaluates PPO models trained with Kalman tracking enabled.
The environment MUST use use_kalman_tracking=True to match training.

Key Difference from experiments/rl/evaluate_rl.py:
    - Uses PPOKalmanPolicyWrapper (semantic marker)
    - Environment created with use_kalman_tracking=True
    - info dict contains e_hat, v_hat, tracking_error
    - CSV includes mean_tracking_error and final_tracking_error columns

Uses the shared evaluation pipeline from experiments/eval_utils.py.
Same rollout logic as baseline and direct RL evaluations.

Usage:
    python experiments/rl_kalman/evaluate_rl_kalman.py \\
        --model results/rl_kalman/models/ppo_kalman_final.zip
    
    python experiments/rl_kalman/evaluate_rl_kalman.py \\
        --model <path> --n-episodes 1000 --model-version v1.0

Output:
    - results/rl_kalman/rl_kalman_results.csv
    - Printed summary with confidence intervals
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.rl_kalman import PPOKalmanPolicyWrapper

from experiments.eval_utils import (
    evaluate_policy,
    format_rl_kalman_comparison_df,
    save_trajectories,
    select_representative_trajectories,
    summarize_results,
)
from experiments.experiment_config import CONFIG, EVAL_CONFIG, KALMAN_CONFIG


# Default paths
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "rl_kalman" / "rl_kalman_results.csv"
DEFAULT_TRAJECTORY_DIR = PROJECT_ROOT / "results" / "rl_kalman" / "trajectories"

# Kalman configuration from shared config (must match training)
PROCESS_VAR = KALMAN_CONFIG["process_var"]
MEASUREMENT_VAR = KALMAN_CONFIG["measurement_var"]
LEAD_TIME = KALMAN_CONFIG["lead_time"]


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float, float]:
    """Compute mean, standard error, and confidence interval."""
    n = len(data)
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    
    mean = np.mean(data)
    std_error = stats.sem(data) if n > 1 else 0.0
    
    if n > 30:
        z = stats.norm.ppf((1 + confidence) / 2)
        ci_half = z * std_error
    else:
        t = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        ci_half = t * std_error
    
    return (mean, std_error, mean - ci_half, mean + ci_half)


def compute_proportion_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float, float]:
    """Compute confidence interval for proportion using Wilson score."""
    if total == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / total
    centre_adjusted_probability = p + z**2 / (2 * total)
    adjusted_standard_deviation = np.sqrt(
        (p * (1 - p) + z**2 / (4 * total)) / total
    )
    
    ci_lower = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    ci_upper = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    
    std_error = np.sqrt(p * (1 - p) / total)
    
    return (p, std_error, ci_lower, ci_upper)


def evaluate_rl_kalman(
    model_path: str,
    n_episodes: int = 1000,
    deterministic: bool = True,
    process_var: float = PROCESS_VAR,
    measurement_var: float = MEASUREMENT_VAR,
    lead_time: float = LEAD_TIME,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Evaluate RL-Kalman policy.
    
    Args:
        model_path: Path to trained PPO-Kalman model.
        n_episodes: Number of evaluation episodes.
        deterministic: Use deterministic actions.
        process_var: Kalman process noise variance (must match training).
        measurement_var: Kalman measurement noise variance (must match training).
        lead_time: Prediction lead time (must match training).
        verbose: Print progress.
    
    Returns:
        Tuple of (results_df, summary_dict).
    """
    if verbose:
        print("=" * 60)
        print("EVALUATING RL-KALMAN (PPO) POLICY")
        print("=" * 60)
        print(f"Model:           {model_path}")
        print(f"Episodes:        {n_episodes}")
        print(f"Deterministic:   {deterministic}")
        print(f"\nKalman Configuration:")
        print(f"  process_var:     {process_var}")
        print(f"  measurement_var: {measurement_var}")
        print(f"  lead_time:       {lead_time}")
        print("-" * 60)
    
    # Load policy
    policy = PPOKalmanPolicyWrapper.load(model_path, deterministic=deterministic)
    
    # Create environment factory with Kalman tracking
    def env_factory():
        config = EnvConfig(
            use_kalman_tracking=True,
            process_var=process_var,
            measurement_var=measurement_var,
            lead_time=lead_time,
        )
        return SoldierEnv(config=config)
    
    # Run evaluation
    seeds = range(CONFIG.SEED_OFFSET, CONFIG.SEED_OFFSET + n_episodes)
    df, summary, trajectories = evaluate_policy(
        env_factory=env_factory,
        policy=policy,
        seeds=seeds,
        verbose=verbose,
    )
    
    return df, summary, trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RL-Kalman (PPO) policy with Monte Carlo"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained PPO-Kalman model (.zip file)"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=EVAL_CONFIG["n_episodes"],
        help=f"Number of evaluation episodes (default: {EVAL_CONFIG['n_episodes']})"
    )
    parser.add_argument(
        "--model-version", type=str, default="unknown",
        help="Model version identifier for the CSV output (default: unknown)"
    )
    parser.add_argument(
        "--process-var", type=float, default=PROCESS_VAR,
        help=f"Kalman process noise variance (default: {PROCESS_VAR})"
    )
    parser.add_argument(
        "--measurement-var", type=float, default=MEASUREMENT_VAR,
        help=f"Kalman measurement noise variance (default: {MEASUREMENT_VAR})"
    )
    parser.add_argument(
        "--lead-time", type=float, default=LEAD_TIME,
        help=f"Prediction lead time (default: {LEAD_TIME})"
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    parser.add_argument(
        "--save-trajectories", action="store_true",
        help="Save representative trajectories"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    df_raw, summary, trajectories = evaluate_rl_kalman(
        model_path=args.model,
        n_episodes=args.n_episodes,
        deterministic=not args.stochastic,
        process_var=args.process_var,
        measurement_var=args.measurement_var,
        lead_time=args.lead_time,
        verbose=not args.quiet,
    )
    
    # Format to RL-Kalman comparison schema
    df = format_rl_kalman_comparison_df(df_raw, model_version=args.model_version)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Save trajectories if requested
    if args.save_trajectories and trajectories:
        traj_dir = Path(DEFAULT_TRAJECTORY_DIR)
        traj_dir.mkdir(parents=True, exist_ok=True)
        representative = select_representative_trajectories(trajectories, df_raw)
        save_trajectories(representative, str(traj_dir))
        print(f"Trajectories saved to: {traj_dir}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (RL-KALMAN)")
    print("=" * 70)
    
    # Success rate with CI
    n_success = int(summary["success_rate"] * summary["n_episodes"])
    p, se, ci_low, ci_high = compute_proportion_ci(
        n_success, summary["n_episodes"]
    )
    
    print(f"\nSuccess Rate:        {p:.1%} (95% CI: [{ci_low:.1%}, {ci_high:.1%}])")
    print(f"Failure Rate:        {summary['failure_rate']:.1%}")
    print(f"Timeout Rate:        {summary['timeout_rate']:.1%}")
    print(f"\nMean Episode Length: {summary['mean_episode_length']:.1f} ± {summary['std_episode_length']:.1f}")
    print(f"Mean Detection Time: {summary['mean_detection_time']:.1f}")
    print(f"Mean Intercept Time: {summary.get('mean_intercept_time', float('nan')):.1f}")
    
    # Print tracking error statistics if available
    if "mean_tracking_error" in df_raw.columns:
        valid_tracking = df_raw["mean_tracking_error"].dropna()
        if len(valid_tracking) > 0:
            mean_err, se_err, ci_lo, ci_hi = compute_confidence_interval(valid_tracking.values)
            print(f"\nMean Tracking Error: {mean_err:.4f} (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
            
            final_tracking = df_raw["final_tracking_error"].dropna()
            if len(final_tracking) > 0:
                mean_final, _, _, _ = compute_confidence_interval(final_tracking.values)
                print(f"Mean Final Track Err: {mean_final:.4f}")
    
    print(f"\nEpisodes Evaluated:  {summary['n_episodes']}")
    print(f"Model Version:       {args.model_version}")
    print("=" * 70)


if __name__ == "__main__":
    main()
