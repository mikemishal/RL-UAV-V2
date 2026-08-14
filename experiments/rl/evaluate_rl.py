"""
Monte Carlo evaluation of trained RL (PPO) policies.

This script evaluates a trained PPO model using the same evaluation pipeline
as the baseline policies, ensuring direct comparability.

Features:
    - Uses shared eval_utils.py (same as baseline evaluation)
    - Computes standard error and 95% confidence intervals
    - Outputs CSV with identical schema to baseline
    - Prints detailed summary statistics

Usage:
    # Default: 1000 episodes, deterministic actions
    python experiments/rl/evaluate_rl.py --model results/rl/models/ppo_defender_final.zip
    
    # Quick test with fewer episodes
    python experiments/rl/evaluate_rl.py --model <path> --n-episodes 100
    
    # Use stochastic actions (exploration noise)
    python experiments/rl/evaluate_rl.py --model <path> --stochastic

Output:
    - results/rl/rl_results.csv (same schema as baseline_results.csv)
    - Printed summary with confidence intervals
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from uav_defend.envs import SoldierEnv
from uav_defend.policies.rl import PPOPolicyWrapper

from experiments.eval_utils import (
    Trajectory,
    evaluate_policy,
    format_comparison_df,
    run_episode_with_trajectory,
    save_trajectories,
    select_representative_trajectories,
    summarize_results,
)
from experiments.experiment_config import CONFIG, EVAL_CONFIG


# Default paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, 'results', 'rl', 'rl_results.csv')
DEFAULT_TRAJECTORY_DIR = os.path.join(PROJECT_ROOT, 'results', 'rl', 'trajectories')


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float, float]:
    """
    Compute mean, standard error, and confidence interval.
    
    Args:
        data: Array of samples.
        confidence: Confidence level (default 0.95 for 95% CI).
    
    Returns:
        Tuple of (mean, std_error, ci_lower, ci_upper).
    """
    n = len(data)
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    
    mean = np.mean(data)
    std_error = stats.sem(data) if n > 1 else 0.0
    
    # t-distribution for small samples, normal for large
    if n > 30:
        z = stats.norm.ppf((1 + confidence) / 2)
        ci_half = z * std_error
    else:
        t = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        ci_half = t * std_error
    
    ci_lower = mean - ci_half
    ci_upper = mean + ci_half
    
    return (mean, std_error, ci_lower, ci_upper)


def compute_proportion_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float, float]:
    """
    Compute confidence interval for a proportion using Wilson score.
    
    Args:
        successes: Number of successes.
        total: Total number of trials.
        confidence: Confidence level.
    
    Returns:
        Tuple of (proportion, std_error, ci_lower, ci_upper).
    """
    if total == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Wilson score interval (better for proportions near 0 or 1)
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)
    
    # Standard error (normal approximation)
    std_error = np.sqrt(p * (1 - p) / total) if total > 0 else 0.0
    
    return (p, std_error, ci_lower, ci_upper)


def print_detailed_summary(df: pd.DataFrame, summary: dict) -> None:
    """
    Print detailed summary statistics with confidence intervals.
    
    Args:
        df: Results DataFrame.
        summary: Summary dict from summarize_results().
    """
    n = len(df)
    
    print("\n" + "=" * 70)
    print("RL POLICY EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total episodes: {n}")
    print("=" * 70)
    
    # Outcome counts
    oc = summary["outcome_counts"]
    n_success = oc.get("intercepted", 0)
    n_soldier_caught = oc.get("soldier_caught", 0)
    n_unsafe = oc.get("unsafe_intercept", 0)
    n_timeout = oc.get("timeout", 0)
    n_fail = n_soldier_caught + n_unsafe
    
    print(f"\n{'OUTCOME DISTRIBUTION':}")
    print(f"  {'Outcome':<20} {'Count':>8} {'Rate':>10} {'95% CI':>20}")
    print(f"  {'-'*60}")
    
    # Success rate with CI
    p, se, ci_lo, ci_hi = compute_proportion_ci(n_success, n)
    print(f"  {'Intercepted (WIN)':<20} {n_success:>8} {p*100:>9.1f}% [{ci_lo*100:>5.1f}%, {ci_hi*100:>5.1f}%]")
    
    # Failure rate with CI
    p, se, ci_lo, ci_hi = compute_proportion_ci(n_fail, n)
    print(f"  {'Failure (total)':<20} {n_fail:>8} {p*100:>9.1f}% [{ci_lo*100:>5.1f}%, {ci_hi*100:>5.1f}%]")
    
    # Breakdown
    p_caught, _, ci_lo, ci_hi = compute_proportion_ci(n_soldier_caught, n)
    print(f"    {'- soldier_caught':<18} {n_soldier_caught:>8} {p_caught*100:>9.1f}% [{ci_lo*100:>5.1f}%, {ci_hi*100:>5.1f}%]")
    
    p_unsafe, _, ci_lo, ci_hi = compute_proportion_ci(n_unsafe, n)
    print(f"    {'- unsafe_intercept':<18} {n_unsafe:>8} {p_unsafe*100:>9.1f}% [{ci_lo*100:>5.1f}%, {ci_hi*100:>5.1f}%]")
    
    # Timeout
    p_timeout, _, ci_lo, ci_hi = compute_proportion_ci(n_timeout, n)
    print(f"  {'Timeout':<20} {n_timeout:>8} {p_timeout*100:>9.1f}% [{ci_lo*100:>5.1f}%, {ci_hi*100:>5.1f}%]")
    
    # Episode length
    print(f"\n{'EPISODE LENGTH':}")
    mean, se, ci_lo, ci_hi = compute_confidence_interval(df["episode_length"].values)
    print(f"  Mean:           {mean:>8.2f} steps (SE: {se:.2f})")
    print(f"  95% CI:         [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  Std:            {df['episode_length'].std():>8.2f}")
    print(f"  Min/Max:        {df['episode_length'].min()} / {df['episode_length'].max()}")
    
    # Detection time (only for detected episodes)
    detected_df = df[df["detected"] == 1]
    valid_detection = detected_df[detected_df["detection_time"] > 0]["detection_time"]
    
    print(f"\n{'DETECTION TIME':} (n={len(valid_detection)} detected)")
    if len(valid_detection) > 0:
        mean, se, ci_lo, ci_hi = compute_confidence_interval(valid_detection.values)
        print(f"  Mean:           {mean:>8.2f} steps (SE: {se:.2f})")
        print(f"  95% CI:         [{ci_lo:.2f}, {ci_hi:.2f}]")
    else:
        print(f"  No detections recorded")
    
    # Intercept time (only for successful episodes)
    success_df = df[df["success"] == 1]
    valid_intercept = success_df[success_df["intercept_time"] > 0]["intercept_time"]
    
    print(f"\n{'INTERCEPT TIME':} (n={len(valid_intercept)} successful)")
    if len(valid_intercept) > 0:
        mean, se, ci_lo, ci_hi = compute_confidence_interval(valid_intercept.values)
        print(f"  Mean:           {mean:>8.2f} steps (SE: {se:.2f})")
        print(f"  95% CI:         [{ci_lo:.2f}, {ci_hi:.2f}]")
    else:
        print(f"  No successful intercepts")
    
    # Distance metrics
    print(f"\n{'DISTANCE METRICS':}")
    
    mean, se, ci_lo, ci_hi = compute_confidence_interval(df["min_enemy_soldier_dist"].values)
    print(f"  Min d(enemy, soldier):")
    print(f"    Mean:         {mean:>8.3f} (SE: {se:.3f})")
    print(f"    95% CI:       [{ci_lo:.3f}, {ci_hi:.3f}]")
    
    mean, se, ci_lo, ci_hi = compute_confidence_interval(df["min_defender_enemy_dist"].values)
    print(f"  Min d(defender, enemy):")
    print(f"    Mean:         {mean:>8.3f} (SE: {se:.3f})")
    print(f"    95% CI:       [{ci_lo:.3f}, {ci_hi:.3f}]")
    
    # Reward (if available)
    if "total_reward" in df.columns:
        print(f"\n{'TOTAL REWARD':}")
        mean, se, ci_lo, ci_hi = compute_confidence_interval(df["total_reward"].values)
        print(f"  Mean:           {mean:>8.2f} (SE: {se:.2f})")
        print(f"  95% CI:         [{ci_lo:.2f}, {ci_hi:.2f}]")
    
    print("\n" + "=" * 70)


def evaluate_rl(
    model_path: str,
    n_episodes: int = 1000,
    seed_offset: int = 0,
    output_path: str | None = None,
    deterministic: bool = True,
    verbose: bool = True,
    save_trajectories_flag: bool = False,
    n_trajectory_episodes: int = 50,
    trajectory_dir: str | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run Monte Carlo evaluation of a trained PPO model.
    
    Args:
        model_path: Path to trained PPO model (.zip file).
        n_episodes: Number of evaluation episodes.
        seed_offset: Starting seed for reproducibility.
        output_path: Path to save results CSV.
        deterministic: If True, use deterministic actions (no exploration).
        verbose: Print progress.
        save_trajectories_flag: If True, save representative trajectories.
        n_trajectory_episodes: Number of episodes to capture for trajectory selection.
        trajectory_dir: Directory to save trajectories.
    
    Returns:
        Tuple of (results DataFrame, summary dict).
    """
    # Load model
    if verbose:
        print(f"Loading model: {model_path}")
    
    policy = PPOPolicyWrapper.load(model_path, deterministic=deterministic)
    
    if verbose:
        mode = "deterministic" if deterministic else "stochastic"
        print(f"Mode: {mode}")
    
    # Create environment factory
    env_factory = lambda: SoldierEnv()
    
    # Generate seeds
    seeds = list(range(seed_offset, seed_offset + n_episodes))
    
    # Trajectory collection setup
    captured_trajectories = []
    n_to_capture = min(n_trajectory_episodes, n_episodes) if save_trajectories_flag else 0
    
    if save_trajectories_flag and verbose:
        print(f"Trajectory capture: first {n_to_capture} episodes")
    
    # Run evaluation - capture trajectories for first N episodes if requested
    if save_trajectories_flag and n_to_capture > 0:
        # Run trajectory-capturing episodes first
        capture_seeds = seeds[:n_to_capture]
        remaining_seeds = seeds[n_to_capture:]
        
        if verbose:
            print(f"\nCapturing {n_to_capture} trajectories...")
        
        df_capture, summary_capture, captured_trajectories = evaluate_policy(
            env_factory=env_factory,
            policy=policy,
            seeds=capture_seeds,
            verbose=verbose,
            capture_trajectories=True,
        )
        
        # Run remaining episodes without trajectory capture (faster)
        if len(remaining_seeds) > 0:
            if verbose:
                print(f"\nRunning remaining {len(remaining_seeds)} episodes...")
            
            df_remaining, _, _ = evaluate_policy(
                env_factory=env_factory,
                policy=policy,
                seeds=remaining_seeds,
                verbose=verbose,
                capture_trajectories=False,
            )
            
            # Combine results
            df_raw = pd.concat([df_capture, df_remaining], ignore_index=True)
        else:
            df_raw = df_capture
        
        # Recompute summary on full dataset
        summary = summarize_results(df_raw)
    else:
        # Standard evaluation without trajectory capture
        df_raw, summary, _ = evaluate_policy(
            env_factory=env_factory,
            policy=policy,
            seeds=seeds,
            verbose=verbose,
        )
    
    # Format to comparison schema (same as baseline)
    df = format_comparison_df(df_raw)
    
    # Print detailed summary with CIs
    print_detailed_summary(df_raw, summary)
    
    # Save results
    if output_path is None:
        output_path = DEFAULT_OUTPUT
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Save representative trajectories
    if save_trajectories_flag and captured_trajectories:
        if trajectory_dir is None:
            trajectory_dir = DEFAULT_TRAJECTORY_DIR
        
        os.makedirs(trajectory_dir, exist_ok=True)
        
        # Use captured episodes DataFrame for selection
        df_for_selection = df_raw.head(n_to_capture)
        
        # Select representative trajectories
        selected = select_representative_trajectories(
            trajectories=captured_trajectories,
            results_df=df_for_selection,
        )
        
        if verbose:
            print(f"\nSaving {len(selected)} representative trajectories:")
            for category, traj in selected.items():
                print(f"  - {category}: seed={traj.seed}, outcome={traj.outcome}")
        
        # Save trajectories
        saved_paths = save_trajectories(
            trajectories=selected,
            output_dir=trajectory_dir,
            policy_name="ppo",
        )
        
        if verbose:
            print(f"\nTrajectories saved to: {trajectory_dir}")
            for path in saved_paths:
                print(f"  - {os.path.basename(path)}")
    
    return df, summary


def main():
    # Get defaults from shared config
    default_episodes = EVAL_CONFIG["n_episodes"]
    default_seed = EVAL_CONFIG["seed_offset"]
    default_traj_episodes = EVAL_CONFIG["trajectory_episodes"]
    
    parser = argparse.ArgumentParser(
        description="Monte Carlo evaluation of trained RL policies"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained PPO model (.zip file)"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=default_episodes,
        help=f"Number of evaluation episodes (default: {default_episodes})"
    )
    parser.add_argument(
        "--seed-offset", type=int, default=default_seed,
        help=f"Starting seed (default: {default_seed})"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: results/rl/rl_results.csv)"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic actions (with exploration noise)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--save-trajectories", action="store_true",
        help="Save representative trajectories for visualization"
    )
    parser.add_argument(
        "--n-trajectory-episodes", type=int, default=default_traj_episodes,
        help=f"Number of episodes to capture for trajectory selection (default: {default_traj_episodes})"
    )
    parser.add_argument(
        "--trajectory-dir", type=str, default=None,
        help="Directory to save trajectories (default: results/rl/trajectories/)"
    )
    
    args = parser.parse_args()
    
    df, summary = evaluate_rl(
        model_path=args.model,
        n_episodes=args.n_episodes,
        seed_offset=args.seed_offset,
        output_path=args.output,
        deterministic=not args.stochastic,
        verbose=not args.quiet,
        save_trajectories_flag=args.save_trajectories,
        n_trajectory_episodes=args.n_trajectory_episodes,
        trajectory_dir=args.trajectory_dir,
    )
    
    return df, summary


if __name__ == "__main__":
    main()
