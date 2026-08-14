"""
Monte Carlo evaluation of the greedy baseline policy.

Runs N episodes with different seeds and collects performance statistics.
Results are saved to results/baseline/baseline_results.csv.

This script uses the shared evaluation utilities from experiments/eval_utils.py,
making it easy to compare baseline and RL policies with identical evaluation code.

Usage:
    python experiments/baseline/evaluate_baseline.py [--n-episodes N] [--seed-offset OFFSET]
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from uav_defend.envs import SoldierEnv
from uav_defend.policies import GreedyInterceptPolicy

from experiments.eval_utils import (
    evaluate_policy,
    print_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo evaluation of greedy baseline policy"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=100,
        help="Number of episodes to run (default: 100)"
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="Starting seed for deterministic evaluation (default: 0)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: results/baseline/baseline_results.csv)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Create environment factory and policy
    env_factory = lambda: SoldierEnv()
    policy = GreedyInterceptPolicy()
    
    # Generate seeds
    seeds = range(args.seed_offset, args.seed_offset + args.n_episodes)
    
    # Run evaluation using shared utilities
    df, summary, _ = evaluate_policy(
        env_factory=env_factory,
        policy=policy,
        seeds=seeds,
        verbose=not args.quiet,
    )
    
    # Print summary
    print_summary(summary, title="BASELINE POLICY SUMMARY")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        # Default path: results/baseline/baseline_results.csv
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        output_dir = os.path.join(project_root, 'results', 'baseline')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'baseline_results.csv')
    
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return df, summary


if __name__ == "__main__":
    main()

