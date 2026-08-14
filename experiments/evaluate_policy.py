"""
Generic policy evaluation script.

Evaluates any policy with the standard `act(obs, info)` interface.
Supports baseline policies (greedy, random) and RL policies (PPO).

This is the unified evaluation entry point for all policy types.

Usage:
    # Evaluate greedy baseline
    python experiments/evaluate_policy.py --policy greedy --n-episodes 1000
    
    # Evaluate random baseline
    python experiments/evaluate_policy.py --policy random --n-episodes 1000
    
    # Evaluate trained PPO model
    python experiments/evaluate_policy.py --policy ppo --model-path models/policies/ppo_defender.zip
    
    # Compare multiple policies
    python experiments/evaluate_policy.py --policy greedy random --n-episodes 500
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
    compare_policies,
    select_representative_trajectories,
    save_trajectories,
)


def evaluate_single_policy(
    policy_name: str,
    n_episodes: int,
    seed_offset: int,
    output_dir: str,
    model_path: str | None = None,
    verbose: bool = True,
    save_traj: bool = False,
    traj_output_dir: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate a single policy.
    
    Args:
        policy_name: Name of the policy to evaluate.
        n_episodes: Number of episodes.
        seed_offset: Starting seed.
        output_dir: Directory to save results.
        model_path: Path to model file (required for PPO).
        verbose: Print progress.
        save_traj: If True, save representative trajectories.
        traj_output_dir: Directory for trajectories (default: results/baseline/trajectories/).
    
    Returns:
        Tuple of (DataFrame, summary dict).
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
    
    # Generate seeds
    seeds = range(seed_offset, seed_offset + n_episodes)
    
    # Run evaluation (capture trajectories if requested)
    df, summary, trajectories = evaluate_policy(
        env_factory=env_factory,
        policy=policy,
        seeds=seeds,
        verbose=verbose,
        capture_trajectories=save_traj,
    )
    
    # Print summary
    policy_display_name = get_policy_name(policy).upper()
    print_summary(summary, title=f"{policy_display_name} POLICY SUMMARY")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{get_policy_name(policy)}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save representative trajectories if requested
    if save_traj and trajectories:
        if traj_output_dir is None:
            project_root = os.path.join(os.path.dirname(__file__), '..')
            traj_output_dir = os.path.join(project_root, 'results', 'baseline', 'trajectories')
        
        # Select representative examples
        selected = select_representative_trajectories(trajectories, df)
        
        # Save them
        saved_paths = save_trajectories(
            selected,
            output_dir=traj_output_dir,
            policy_name=get_policy_name(policy),
        )
        
        print(f"\nSaved {len(saved_paths)} representative trajectories:")
        for path in saved_paths:
            print(f"  - {os.path.basename(path)}")
    
    return df, summary


def evaluate_multiple_policies(
    policy_names: list[str],
    n_episodes: int,
    seed_offset: int,
    output_dir: str,
    model_path: str | None = None,
    verbose: bool = True,
    save_traj: bool = False,
    traj_output_dir: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """
    Evaluate multiple policies on the same seeds.
    
    Args:
        policy_names: List of policy names.
        n_episodes: Number of episodes per policy.
        seed_offset: Starting seed.
        output_dir: Directory to save results.
        model_path: Path to model file (for PPO).
        verbose: Print progress.
        save_traj: If True, save representative trajectories.
        traj_output_dir: Directory for trajectories.
    
    Returns:
        Tuple of (dict of DataFrames, dict of summaries).
    """
    # Create policies
    policies = {}
    for name in policy_names:
        if name.lower() == "ppo":
            if model_path is None:
                raise ValueError("--model-path required for PPO policy")
            policies[name] = get_policy("ppo", model_path=model_path)
        else:
            policies[name] = get_policy(name)
    
    # Create environment factory
    env_factory = lambda: SoldierEnv()
    
    # Generate seeds (same for all policies for fair comparison)
    seeds = range(seed_offset, seed_offset + n_episodes)
    
    # Run comparison
    dfs, summaries, trajectories_dict = compare_policies(
        env_factory=env_factory,
        policies=policies,
        seeds=seeds,
        verbose=verbose,
        capture_trajectories=save_traj,
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    for name, df in dfs.items():
        csv_path = os.path.join(output_dir, f"{name.lower()}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
    
    # Save comparison summary
    comparison_df = pd.DataFrame([
        {
            "policy": name,
            "success_rate": s["success_rate"],
            "failure_rate": s["failure_rate"],
            "mean_episode_length": s["mean_episode_length"],
            "mean_detection_time": s["mean_detection_time"],
            "mean_intercept_time": s["mean_intercept_time"],
        }
        for name, s in summaries.items()
    ])
    comparison_path = os.path.join(output_dir, "policy_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("POLICY COMPARISON")
    print("=" * 60)
    print(f"{'Policy':<15} {'Success':>10} {'Failure':>10} {'Ep Len':>10}")
    print("-" * 60)
    for name, s in summaries.items():
        print(f"{name:<15} {s['success_rate']*100:>9.1f}% "
              f"{s['failure_rate']*100:>9.1f}% "
              f"{s['mean_episode_length']:>10.1f}")
    print("=" * 60)
    
    # Save representative trajectories if requested
    if save_traj and trajectories_dict:
        if traj_output_dir is None:
            project_root = os.path.join(os.path.dirname(__file__), '..')
            traj_output_dir = os.path.join(project_root, 'results', 'baseline', 'trajectories')
        
        print("\nSaving representative trajectories...")
        for policy_name, trajectories in trajectories_dict.items():
            selected = select_representative_trajectories(trajectories, dfs[policy_name])
            saved_paths = save_trajectories(
                selected,
                output_dir=traj_output_dir,
                policy_name=policy_name.lower(),
            )
            print(f"  {policy_name}: {len(saved_paths)} trajectories")
    
    return dfs, summaries


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate policies on the UAV defense environment"
    )
    parser.add_argument(
        "--policy", type=str, nargs="+", default=["greedy"],
        help=f"Policy name(s) to evaluate. Available: {', '.join(list_policies())}"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=100,
        help="Number of episodes per policy (default: 100)"
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="Starting seed for deterministic evaluation (default: 0)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained model file (required for PPO policy)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/evaluation/)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--save-trajectories", action="store_true",
        help="Save representative trajectories (success, failure, borderline)"
    )
    parser.add_argument(
        "--traj-output-dir", type=str, default=None,
        help="Directory for trajectories (default: results/baseline/trajectories/)"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        project_root = os.path.join(os.path.dirname(__file__), '..')
        output_dir = os.path.join(project_root, 'results', 'evaluation')
    
    # Single or multiple policy evaluation
    if len(args.policy) == 1:
        df, summary = evaluate_single_policy(
            policy_name=args.policy[0],
            n_episodes=args.n_episodes,
            seed_offset=args.seed_offset,
            output_dir=output_dir,
            model_path=args.model_path,
            verbose=not args.quiet,
            save_traj=args.save_trajectories,
            traj_output_dir=args.traj_output_dir,
        )
        return df, summary
    else:
        dfs, summaries = evaluate_multiple_policies(
            policy_names=args.policy,
            n_episodes=args.n_episodes,
            seed_offset=args.seed_offset,
            output_dir=output_dir,
            model_path=args.model_path,
            verbose=not args.quiet,
            save_traj=args.save_trajectories,
            traj_output_dir=args.traj_output_dir,
        )
        return dfs, summaries


if __name__ == "__main__":
    main()
