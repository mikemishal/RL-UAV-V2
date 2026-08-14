"""
Single-episode rollout script for RL-Kalman policy debugging.

=============================================================================
PURPOSE: QUALITATIVE DEBUGGING BEFORE MONTE CARLO EVALUATION
=============================================================================

This script runs a single episode with a trained PPO-Kalman model,
printing detailed step-by-step information for debugging and visualization.

Use this to:
    - Verify the policy is behaving reasonably
    - Debug unexpected failures
    - Inspect Kalman tracking quality
    - Generate trajectory data for visualization

Usage:
    # Basic single episode
    python experiments/rl_kalman/run_rl_kalman_episode.py \\
        --model results/rl_kalman/models/ppo_kalman_final.zip
    
    # With specific seed
    python experiments/rl_kalman/run_rl_kalman_episode.py \\
        --model results/rl_kalman/models/ppo_kalman_final.zip --seed 123
    
    # Save trajectory for visualization
    python experiments/rl_kalman/run_rl_kalman_episode.py \\
        --model results/rl_kalman/models/ppo_kalman_final.zip --save-trajectory
    
    # Verbose mode (print every step)
    python experiments/rl_kalman/run_rl_kalman_episode.py \\
        --model results/rl_kalman/models/ppo_kalman_final.zip --verbose

Output:
    - Printed episode summary
    - Optional: results/rl_kalman/debug/trajectory_{seed}.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.rl_kalman import PPOKalmanPolicyWrapper

from experiments.experiment_config import KALMAN_CONFIG

# Kalman configuration from shared config
PROCESS_VAR = KALMAN_CONFIG["process_var"]
MEASUREMENT_VAR = KALMAN_CONFIG["measurement_var"]
LEAD_TIME = KALMAN_CONFIG["lead_time"]

# Output directory
DEBUG_DIR = PROJECT_ROOT / "results" / "rl_kalman" / "debug"


def run_single_episode(
    model_path: str,
    seed: int = 42,
    process_var: float = PROCESS_VAR,
    measurement_var: float = MEASUREMENT_VAR,
    lead_time: float = LEAD_TIME,
    verbose: bool = False,
    save_trajectory: bool = False,
) -> dict:
    """
    Run a single episode with the RL-Kalman policy.
    
    Args:
        model_path: Path to trained PPO-Kalman model.
        seed: Random seed for episode.
        process_var: Kalman process noise variance.
        measurement_var: Kalman measurement noise variance.
        lead_time: Prediction lead time.
        verbose: If True, print step-by-step details.
        save_trajectory: If True, save trajectory data to JSON.
    
    Returns:
        Dictionary with episode results.
    """
    print("=" * 70)
    print("SINGLE EPISODE ROLLOUT (RL-KALMAN)")
    print("=" * 70)
    print(f"Model:           {model_path}")
    print(f"Seed:            {seed}")
    print(f"Kalman config:   process_var={process_var}, measurement_var={measurement_var}")
    print("-" * 70)
    
    # Load policy
    print("\nLoading policy...")
    policy = PPOKalmanPolicyWrapper.load(model_path, deterministic=True)
    print("Policy loaded successfully.")
    
    # Create environment with Kalman tracking
    config = EnvConfig(
        use_kalman_tracking=True,
        process_var=process_var,
        measurement_var=measurement_var,
        lead_time=lead_time,
    )
    env = SoldierEnv(config=config)
    
    # Reset environment
    obs, info = env.reset(seed=seed)
    
    # Trajectory storage
    trajectory = {
        "seed": seed,
        "config": {
            "process_var": process_var,
            "measurement_var": measurement_var,
            "lead_time": lead_time,
        },
        "steps": [],
    }
    
    # Episode tracking
    done = False
    step_count = 0
    detection_time = None
    tracking_errors = []
    
    print("\n" + "-" * 70)
    print("RUNNING EPISODE")
    print("-" * 70)
    
    if verbose:
        print(f"\n{'Step':>5} {'Detected':>8} {'D-E Dist':>10} {'E-S Dist':>10} "
              f"{'Track Err':>10} {'Action':>20}")
        print("-" * 70)
    
    while not done:
        # Get action from policy
        action = policy.act(obs, info)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        # Track detection time
        if info["enemy_detected"] and detection_time is None:
            detection_time = step_count
        
        # Track Kalman tracking error
        if info.get("tracking_error") is not None:
            tracking_errors.append(info["tracking_error"])
        
        # Store step data
        step_data = {
            "step": step_count,
            "soldier_pos": info["soldier_pos"].tolist(),
            "defender_pos": info["defender_pos"].tolist(),
            "enemy_pos": info["enemy_pos"].tolist(),
            "detected": info["enemy_detected"],
            "e_hat": info["e_hat"].tolist() if info["e_hat"] is not None else None,
            "v_hat": info["v_hat"].tolist() if info["v_hat"] is not None else None,
            "tracking_error": info.get("tracking_error"),
            "defender_enemy_dist": info["defender_enemy_dist"],
            "enemy_soldier_dist": info["enemy_soldier_dist"],
            "action": action.tolist(),
            "reward": reward,
        }
        trajectory["steps"].append(step_data)
        
        # Verbose output
        if verbose:
            detected_str = "YES" if info["enemy_detected"] else "no"
            track_err = f"{info.get('tracking_error', 0.0):.3f}" if info.get("tracking_error") else "-"
            action_str = f"[{action[0]:+.2f}, {action[1]:+.2f}]"
            print(f"{step_count:>5} {detected_str:>8} {info['defender_enemy_dist']:>10.2f} "
                  f"{info['enemy_soldier_dist']:>10.2f} {track_err:>10} {action_str:>20}")
    
    # Episode results
    outcome = info.get("outcome", "unknown")
    intercept_time = step_count if outcome == "intercepted" else None
    mean_tracking_error = np.mean(tracking_errors) if tracking_errors else None
    
    # Final distances
    final_defender_enemy_dist = info["defender_enemy_dist"]
    final_enemy_soldier_dist = info["enemy_soldier_dist"]
    
    # Store summary in trajectory
    trajectory["summary"] = {
        "outcome": outcome,
        "episode_length": step_count,
        "detection_time": detection_time,
        "intercept_time": intercept_time,
        "mean_tracking_error": mean_tracking_error,
        "final_defender_enemy_dist": final_defender_enemy_dist,
        "final_enemy_soldier_dist": final_enemy_soldier_dist,
        "n_tracking_samples": len(tracking_errors),
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("EPISODE RESULTS")
    print("=" * 70)
    
    # Outcome with color indication
    outcome_display = {
        "intercepted": "INTERCEPTED (SUCCESS)",
        "soldier_caught": "SOLDIER CAUGHT (FAILURE)",
        "unsafe_intercept": "UNSAFE INTERCEPT (FAILURE)",
        "timeout": "TIMEOUT (FAILURE)",
    }.get(outcome, outcome.upper())
    
    print(f"\n  Outcome:              {outcome_display}")
    print(f"  Episode Length:       {step_count} steps")
    print(f"  Detection Time:       {detection_time if detection_time else 'Never detected'}")
    
    if intercept_time:
        print(f"  Intercept Time:       {intercept_time} steps")
        time_to_intercept = intercept_time - (detection_time or 0)
        print(f"  Time After Detection: {time_to_intercept} steps")
    
    print(f"\n  Mean Tracking Error:  {mean_tracking_error:.4f}" if mean_tracking_error else 
          "\n  Mean Tracking Error:  N/A (no tracking)")
    
    print(f"\n  Final Distances:")
    print(f"    Defender → Enemy:   {final_defender_enemy_dist:.2f}")
    print(f"    Enemy → Soldier:    {final_enemy_soldier_dist:.2f}")
    
    # Save trajectory if requested
    if save_trajectory:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        traj_path = DEBUG_DIR / f"trajectory_{seed}.json"
        
        with open(traj_path, "w") as f:
            json.dump(trajectory, f, indent=2)
        
        print(f"\n  Trajectory saved to: {traj_path}")
    
    print("\n" + "=" * 70)
    
    env.close()
    
    return trajectory["summary"]


def main():
    parser = argparse.ArgumentParser(
        description="Single-episode rollout for RL-Kalman policy debugging"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained PPO-Kalman model (.zip file)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for episode (default: 42)"
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
        "--verbose", "-v", action="store_true",
        help="Print step-by-step details"
    )
    parser.add_argument(
        "--save-trajectory", "-s", action="store_true",
        help="Save trajectory data to JSON for visualization"
    )
    
    args = parser.parse_args()
    
    # Verify model exists
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    
    # Run episode
    summary = run_single_episode(
        model_path=args.model,
        seed=args.seed,
        process_var=args.process_var,
        measurement_var=args.measurement_var,
        lead_time=args.lead_time,
        verbose=args.verbose,
        save_trajectory=args.save_trajectory,
    )
    
    # Return exit code based on outcome
    if summary["outcome"] == "intercepted":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
