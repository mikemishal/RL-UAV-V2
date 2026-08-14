"""
Single-episode rollout script for Kalman baseline policy debugging.

Runs exactly one episode with the KalmanGreedyInterceptPolicy and prints
detailed diagnostics including Kalman tracking metrics. Useful for
verifying the Kalman baseline works correctly before Monte Carlo evaluation.

This script is the Kalman-tracking counterpart of run_baseline_episode.py:
    - run_baseline_episode.py         : GreedyInterceptPolicy,        use_kalman_tracking=False
    - run_kalman_baseline_episode.py  : KalmanGreedyInterceptPolicy,  use_kalman_tracking=True

Usage:
    python experiments/baseline/run_kalman_baseline_episode.py [--seed SEED] [--save] [--quiet]
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.policies.baseline import KalmanGreedyInterceptPolicy


def run_episode(seed: int = 42, verbose: bool = True) -> dict:
    """
    Run a single episode with the Kalman greedy baseline policy.

    The environment is configured with use_kalman_tracking=True so that
    the Kalman filter runs and exposes e_hat / v_hat / tracking_error
    through the info dict on every step.

    Args:
        seed: Random seed for reproducibility.
        verbose: If True, print step-by-step progress.

    Returns:
        Dictionary with keys:
            "trajectory": per-step arrays of all entity positions and metrics.
            "summary":    episode-level statistics.
    """
    config = EnvConfig(use_kalman_tracking=True)
    env = SoldierEnv(config=config)
    policy = KalmanGreedyInterceptPolicy()

    # ----- Trajectory storage -----
    trajectory = {
        "soldier_pos": [],
        "defender_pos": [],
        "enemy_pos": [],
        "e_hat": [],          # Kalman-filtered enemy position (None before detection)
        "v_hat": [],          # Kalman-filtered enemy velocity  (None before detection)
        "actions": [],
        "rewards": [],
        "detected": [],
        "tracking_error": [], # Euclidean filter error post-detection (None before detection)
        "enemy_soldier_dist": [],
        "defender_enemy_dist": [],
    }

    # Reset environment and policy
    obs, info = env.reset(seed=seed)
    policy.reset()

    # Timing bookmarks
    detection_step: int | None = None
    intercept_step: int | None = None

    # Store initial state (step 0)
    trajectory["soldier_pos"].append(info["soldier_pos"].copy())
    trajectory["defender_pos"].append(info["defender_pos"].copy())
    trajectory["enemy_pos"].append(info["enemy_pos"].copy())
    trajectory["e_hat"].append(info["e_hat"].copy() if info["e_hat"] is not None else None)
    trajectory["v_hat"].append(info["v_hat"].copy() if info["v_hat"] is not None else None)
    trajectory["detected"].append(info["enemy_detected"])
    trajectory["tracking_error"].append(info["tracking_error"])
    trajectory["enemy_soldier_dist"].append(info["enemy_soldier_dist"])
    trajectory["defender_enemy_dist"].append(info["defender_enemy_dist"])

    if verbose:
        print("=" * 65)
        print(f"KALMAN BASELINE EPISODE ROLLOUT  (seed={seed})")
        print("=" * 65)
        print(f"  Config: use_kalman_tracking=True, "
              f"process_var={config.process_var}, "
              f"measurement_var={config.measurement_var}")
        print(f"  Initial enemy position:          {info['enemy_pos']}")
        print(f"  Initial enemy-soldier distance:  {info['enemy_soldier_dist']:.2f} m")
        print("-" * 65)

    step = 0
    total_reward = 0.0
    done = False

    while not done:
        action = policy.act(obs, info)
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        total_reward += reward

        # Store per-step data
        trajectory["actions"].append(action.copy())
        trajectory["rewards"].append(reward)
        trajectory["soldier_pos"].append(info["soldier_pos"].copy())
        trajectory["defender_pos"].append(info["defender_pos"].copy())
        trajectory["enemy_pos"].append(info["enemy_pos"].copy())
        trajectory["e_hat"].append(info["e_hat"].copy() if info["e_hat"] is not None else None)
        trajectory["v_hat"].append(info["v_hat"].copy() if info["v_hat"] is not None else None)
        trajectory["detected"].append(info["enemy_detected"])
        trajectory["tracking_error"].append(info["tracking_error"])
        trajectory["enemy_soldier_dist"].append(info["enemy_soldier_dist"])
        trajectory["defender_enemy_dist"].append(info["defender_enemy_dist"])

        # First detection event
        if info["enemy_detected"] and detection_step is None:
            detection_step = step
            if verbose:
                print(f"Step {step:4d}: [DETECTED]  "
                      f"d_de={info['defender_enemy_dist']:.2f} m  "
                      f"e_hat={info['e_hat']}  "
                      f"track_err={info['tracking_error']:.3f} m")

        # Periodic status
        if verbose and step % 50 == 0:
            status = "tracking " if info["enemy_detected"] else "searching"
            track_str = (f"  track_err={info['tracking_error']:.3f} m"
                         if info["tracking_error"] is not None else "")
            print(f"Step {step:4d}: [{status}]  "
                  f"d_es={info['enemy_soldier_dist']:.2f} m  "
                  f"d_de={info['defender_enemy_dist']:.2f} m  "
                  f"r={reward:+.2f}{track_str}")

    # Record intercept timing
    if info["outcome"] == "intercepted":
        intercept_step = step

    # Compute mean tracking error over all post-detection steps
    post_detection_errors = [
        e for e in trajectory["tracking_error"] if e is not None
    ]
    mean_tracking_error = float(np.mean(post_detection_errors)) if post_detection_errors else None

    summary = {
        "seed": seed,
        "outcome": info["outcome"],
        "episode_length": step,
        "total_reward": total_reward,
        "detection_occurred": detection_step is not None,
        "detection_step": detection_step,
        "intercept_step": intercept_step,
        "final_enemy_soldier_dist": info["enemy_soldier_dist"],
        "final_defender_enemy_dist": info["defender_enemy_dist"],
        "final_tracking_error": info["tracking_error"],
        "mean_tracking_error": mean_tracking_error,
        "num_tracked_steps": len(post_detection_errors),
    }

    if verbose:
        print("-" * 65)
        print("EPISODE SUMMARY")
        print("-" * 65)
        print(f"  Outcome:               {summary['outcome']}")
        print(f"  Episode length:        {summary['episode_length']} steps")
        print(f"  Total reward:          {summary['total_reward']:.2f}")
        print(f"  Detection occurred:    {'Yes' if summary['detection_occurred'] else 'No'}")
        if summary["detection_step"] is not None:
            print(f"  Detection step:        {summary['detection_step']}")
        if summary["intercept_step"] is not None:
            print(f"  Intercept step:        {summary['intercept_step']}")
        print(f"  Final d(enemy,soldier):  {summary['final_enemy_soldier_dist']:.2f} m")
        print(f"  Final d(defender,enemy): {summary['final_defender_enemy_dist']:.2f} m")
        if summary["mean_tracking_error"] is not None:
            print(f"  Mean tracking error:   {summary['mean_tracking_error']:.4f} m "
                  f"(over {summary['num_tracked_steps']} tracked steps)")
        if summary["final_tracking_error"] is not None:
            print(f"  Final tracking error:  {summary['final_tracking_error']:.4f} m")
        print("=" * 65)

    return {"trajectory": trajectory, "summary": summary}


def save_trajectory(data: dict, filepath: str) -> None:
    """
    Save trajectory data to a .npz file for offline plotting.

    None-valued entries in e_hat, v_hat, and tracking_error (pre-detection
    steps) are stored as NaN so array shape is uniform.

    Args:
        data: Return value of run_episode().
        filepath: Destination .npz path. Parent directory is created if needed.
    """
    traj = data["trajectory"]

    def _fill_none_2d(lst: list) -> np.ndarray:
        """Replace None entries with [NaN, NaN] and stack to (N, 2) array."""
        return np.array([
            v if v is not None else [np.nan, np.nan] for v in lst
        ], dtype=np.float32)

    def _fill_none_scalar(lst: list) -> np.ndarray:
        """Replace None entries with NaN and return 1-D array."""
        return np.array([
            v if v is not None else np.nan for v in lst
        ], dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    np.savez(
        filepath,
        # Entity positions
        soldier_pos=np.array(traj["soldier_pos"], dtype=np.float32),
        defender_pos=np.array(traj["defender_pos"], dtype=np.float32),
        enemy_pos=np.array(traj["enemy_pos"], dtype=np.float32),
        # Kalman estimates (NaN before detection)
        e_hat=_fill_none_2d(traj["e_hat"]),
        v_hat=_fill_none_2d(traj["v_hat"]),
        tracking_error=_fill_none_scalar(traj["tracking_error"]),
        # Control & reward
        actions=np.array(traj["actions"], dtype=np.float32) if traj["actions"] else np.empty((0, 2), dtype=np.float32),
        rewards=np.array(traj["rewards"], dtype=np.float32),
        # Flags & distances
        detected=np.array(traj["detected"], dtype=bool),
        enemy_soldier_dist=np.array(traj["enemy_soldier_dist"], dtype=np.float32),
        defender_enemy_dist=np.array(traj["defender_enemy_dist"], dtype=np.float32),
        # Episode summary scalars
        seed=np.int32(data["summary"]["seed"]),
        episode_length=np.int32(data["summary"]["episode_length"]),
        total_reward=np.float32(data["summary"]["total_reward"]),
        detection_step=np.int32(data["summary"]["detection_step"] if data["summary"]["detection_step"] is not None else -1),
        mean_tracking_error=np.float32(
            data["summary"]["mean_tracking_error"] if data["summary"]["mean_tracking_error"] is not None else np.nan
        ),
    )
    print(f"Trajectory saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a single Kalman baseline episode for debugging"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save trajectory to experiments/baseline/results/trajectories/",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress step-by-step output",
    )

    args = parser.parse_args()

    data = run_episode(seed=args.seed, verbose=not args.quiet)

    if args.save:
        results_dir = os.path.join(os.path.dirname(__file__), "results", "trajectories")
        filepath = os.path.join(results_dir, f"kalman_baseline_trajectory_seed{args.seed}.npz")
        save_trajectory(data, filepath)

    return data


if __name__ == "__main__":
    main()
