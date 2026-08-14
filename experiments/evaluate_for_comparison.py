"""
Unified policy evaluation for baseline vs RL comparison.

This script is the single entry point for evaluating policies in a way that
produces identical, aligned CSV files for head-to-head comparison.

Key guarantees:
    - Baseline and RL policies use the EXACT same seeds
    - CSV schema is identical (see COMPARISON_COLUMNS below)
    - Output paths are standardized:
        - Baseline: results/baseline/1k_episodes/baseline_results.csv
        - RL:       results/rl/rl_results.csv

Usage:
    # Evaluate greedy baseline (1000 episodes, seeds 0-999)
    python experiments/evaluate_for_comparison.py --policy greedy --n-episodes 1000

    # Evaluate trained PPO with same seeds
    python experiments/evaluate_for_comparison.py --policy ppo --model-path results/rl/models/ppo_defender_final.zip --n-episodes 1000

    # Compare both (outputs aligned CSVs)
    python experiments/evaluate_for_comparison.py --policy greedy ppo --model-path results/rl/models/ppo_defender_final.zip --n-episodes 1000

CSV Schema (identical for baseline and RL):
    seed                  - Random seed for episode
    outcome               - Episode outcome (intercepted, soldier_caught, unsafe_intercept, timeout)
    success               - 1 if intercepted, 0 otherwise
    episode_length        - Number of steps
    detected              - 1 if enemy was detected, 0 otherwise
    detection_time        - Step when detected (-1 if never)
    intercept_time        - Step when intercepted (-1 if never)
    enemy_speed           - Enemy speed (v_e)
    defender_speed        - Defender speed (v_d)
    detection_radius      - Detection radius
    intercept_radius      - Intercept radius
    threat_radius         - Threat radius
    min_enemy_soldier_dist   - Minimum d(enemy, soldier) during episode
    min_defender_enemy_dist  - Minimum d(defender, enemy) during episode
    num_episodes          - Total number of episodes in this evaluation run
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from uav_defend.envs import SoldierEnv
from uav_defend.policies import get_policy, list_policies, get_policy_name

from experiments.eval_utils import (
    evaluate_policy,
    print_summary,
    format_comparison_df,
)


# Standard output paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
OUTPUT_PATHS = {
    "greedy": os.path.join(PROJECT_ROOT, "results", "baseline", "1k_episodes", "baseline_results.csv"),
    "random": os.path.join(PROJECT_ROOT, "results", "baseline", "random_results.csv"),
    "ppo": os.path.join(PROJECT_ROOT, "results", "rl", "rl_results.csv"),
}


def get_output_path(policy_name: str) -> str:
    """Get the standard output path for a policy type."""
    policy_key = policy_name.lower()
    if policy_key in OUTPUT_PATHS:
        return OUTPUT_PATHS[policy_key]
    # Default: results/evaluation/<policy>_results.csv
    return os.path.join(PROJECT_ROOT, "results", "evaluation", f"{policy_key}_results.csv")


def evaluate_for_comparison(
    policy_name: str,
    n_episodes: int,
    seed_offset: int = 0,
    model_path: str | None = None,
    output_path: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate a policy and save results in the standard comparison format.
    
    Args:
        policy_name: Policy type ("greedy", "random", "ppo").
        n_episodes: Number of episodes.
        seed_offset: Starting seed (default 0).
        model_path: Path to trained model (required for PPO).
        output_path: Custom output path (default uses standard paths).
        verbose: Print progress.
    
    Returns:
        Tuple of (formatted_df, summary).
    """
    # Create policy
    if policy_name.lower() == "ppo":
        if model_path is None:
            raise ValueError("--model-path required for PPO policy")
        policy = get_policy("ppo", model_path=model_path)
    else:
        policy = get_policy(policy_name)
    
    # Create environment factory
    env_factory = lambda: SoldierEnv()
    
    # Generate seeds (identical for all policies)
    seeds = list(range(seed_offset, seed_offset + n_episodes))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION FOR COMPARISON: {policy_name.upper()}")
        print(f"{'='*60}")
        print(f"  Policy:     {policy_name}")
        print(f"  Seed range: [{seeds[0]}, {seeds[-1]}]")
        print(f"  N episodes: {n_episodes}")
        print(f"{'='*60}")
    
    # Run evaluation
    df_raw, summary, _ = evaluate_policy(
        env_factory=env_factory,
        policy=policy,
        seeds=seeds,
        verbose=verbose,
    )
    
    # Format to comparison schema
    df = format_comparison_df(df_raw)
    
    # Print summary
    print_summary(summary, title=f"{policy_name.upper()} SUMMARY")
    
    # Determine output path
    if output_path is None:
        output_path = get_output_path(policy_name)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Schema: {', '.join(df.columns)}")
    
    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate policies for comparison (unified schema, same seeds)"
    )
    parser.add_argument(
        "--policy", type=str, nargs="+", required=True,
        help=f"Policy name(s) to evaluate. Available: {', '.join(list_policies())}"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=1000,
        help="Number of episodes (default: 1000)"
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="Starting seed (default: 0)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained model (required for PPO)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory for all policies"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    results = {}
    
    for policy_name in args.policy:
        # Determine output path
        if args.output_dir:
            output_path = os.path.join(args.output_dir, f"{policy_name.lower()}_results.csv")
        else:
            output_path = None  # Use default
        
        df, summary = evaluate_for_comparison(
            policy_name=policy_name,
            n_episodes=args.n_episodes,
            seed_offset=args.seed_offset,
            model_path=args.model_path,
            output_path=output_path,
            verbose=not args.quiet,
        )
        
        results[policy_name] = {"df": df, "summary": summary}
    
    # If multiple policies, print comparison
    if len(args.policy) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Policy':<15} {'Success':>10} {'Failure':>10} {'Ep Len':>10} {'Det Time':>10}")
        print("-" * 70)
        for name, data in results.items():
            s = data["summary"]
            det_time = s.get("mean_detection_time", float("nan"))
            det_str = f"{det_time:.1f}" if not pd.isna(det_time) else "N/A"
            print(f"{name:<15} {s['success_rate']*100:>9.1f}% "
                  f"{s['failure_rate']*100:>9.1f}% "
                  f"{s['mean_episode_length']:>10.1f} "
                  f"{det_str:>10}")
        print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
