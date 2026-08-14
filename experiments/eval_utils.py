"""
Reusable evaluation utilities for policy evaluation.

This module provides modular functions for running episodes,
evaluating policies, and summarizing results. Works with any
policy that implements the `act(obs, info) -> action` interface.

Usage:
    from experiments.eval_utils import run_episode, evaluate_policy, summarize_results
    
    # Evaluate a policy
    df, summary = evaluate_policy(
        env_factory=lambda: SoldierEnv(),
        policy=GreedyInterceptPolicy(),
        seeds=range(100),
    )
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, Any

import numpy as np
import pandas as pd


@dataclass
class Trajectory:
    """Container for episode trajectory data."""
    
    seed: int
    outcome: str
    episode_length: int
    
    # Position histories (T x 2 arrays)
    soldier_positions: np.ndarray
    defender_positions: np.ndarray
    enemy_positions: np.ndarray
    estimated_enemy_positions: np.ndarray  # Kalman estimate
    
    # Additional metadata
    detection_time: int = -1
    intercept_time: int = -1
    total_reward: float = 0.0
    
    def save(self, filepath: str) -> None:
        """Save trajectory to .npz file."""
        np.savez(
            filepath,
            seed=self.seed,
            outcome=self.outcome,
            episode_length=self.episode_length,
            soldier_positions=self.soldier_positions,
            defender_positions=self.defender_positions,
            enemy_positions=self.enemy_positions,
            estimated_enemy_positions=self.estimated_enemy_positions,
            detection_time=self.detection_time,
            intercept_time=self.intercept_time,
            total_reward=self.total_reward,
        )
    
    @classmethod
    def load(cls, filepath: str) -> "Trajectory":
        """Load trajectory from .npz file."""
        data = np.load(filepath, allow_pickle=True)
        return cls(
            seed=int(data["seed"]),
            outcome=str(data["outcome"]),
            episode_length=int(data["episode_length"]),
            soldier_positions=data["soldier_positions"],
            defender_positions=data["defender_positions"],
            enemy_positions=data["enemy_positions"],
            estimated_enemy_positions=data["estimated_enemy_positions"],
            detection_time=int(data["detection_time"]),
            intercept_time=int(data["intercept_time"]),
            total_reward=float(data["total_reward"]),
        )


class Policy(Protocol):
    """Protocol for policies compatible with the evaluation utilities."""
    
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Return action given observation and info."""
        ...
    
    def reset(self) -> None:
        """Reset any internal state."""
        ...


def run_episode(env, policy: Policy, seed: int) -> dict:
    """
    Run a single episode and collect metrics.
    
    This is the core evaluation primitive. It runs one complete episode
    with the given policy and returns a dictionary of metrics.
    
    Args:
        env: A Gymnasium-compatible environment instance.
        policy: A policy with `act(obs, info)` and `reset()` methods.
        seed: Random seed for environment reset.
    
    Returns:
        Dictionary containing episode metrics:
            - seed: The random seed used
            - outcome: Episode outcome string
            - success: 1 if intercepted, else 0
            - episode_length: Number of steps
            - total_reward: Cumulative reward
            - detected: 1 if enemy was detected, else 0
            - detection_time: Step when detected (or -1)
            - intercept_time: Step when intercepted (or -1)
            - min_enemy_soldier_dist: Minimum distance during episode
            - min_defender_enemy_dist: Minimum distance during episode
            - final_enemy_soldier_dist: Distance at episode end
            - final_defender_enemy_dist: Distance at episode end
            - env_config: Dictionary of environment parameters
            
        When Kalman tracking is enabled (tracking_error in info):
            - mean_tracking_error: Average tracking error during episode
            - final_tracking_error: Tracking error at episode end
    """
    obs, info = env.reset(seed=seed)
    policy.reset()
    
    # Track metrics
    total_reward = 0.0
    min_enemy_soldier_dist = info["enemy_soldier_dist"]
    min_defender_enemy_dist = info["defender_enemy_dist"]
    detection_time = None
    intercept_time = None
    
    # Kalman tracking error (when available)
    tracking_errors = []
    final_tracking_error = None
    
    step = 0
    done = False
    
    while not done:
        action = policy.act(obs, info)
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        total_reward += reward
        
        # Update minimum distances
        min_enemy_soldier_dist = min(min_enemy_soldier_dist, info["enemy_soldier_dist"])
        min_defender_enemy_dist = min(min_defender_enemy_dist, info["defender_enemy_dist"])
        
        # Track detection timing
        if info["enemy_detected"] and detection_time is None:
            detection_time = step
        
        # Track Kalman tracking error (only when detected, i.e., tracking_error is not None)
        if info.get("tracking_error") is not None:
            tracking_errors.append(info["tracking_error"])
            final_tracking_error = info["tracking_error"]
        
        if done or truncated:
            break
    
    # Record intercept time if successful
    if info["outcome"] == "intercepted":
        intercept_time = step
    
    # Get config values
    config = env.config
    
    # Build base metrics dict
    metrics = {
        # Episode identifiers
        "seed": seed,
        
        # Outcome metrics
        "outcome": info["outcome"],
        "success": 1 if info["outcome"] == "intercepted" else 0,
        "episode_length": step,
        "total_reward": total_reward,
        
        # Detection and intercept timing
        "detected": 1 if detection_time is not None else 0,
        "detection_time": detection_time if detection_time is not None else -1,
        "intercept_time": intercept_time if intercept_time is not None else -1,
        
        # Distance metrics
        "min_enemy_soldier_dist": min_enemy_soldier_dist,
        "min_defender_enemy_dist": min_defender_enemy_dist,
        "final_enemy_soldier_dist": info["enemy_soldier_dist"],
        "final_defender_enemy_dist": info["defender_enemy_dist"],
        
        # Environment configuration (for reproducibility)
        "enemy_speed": config.v_e,
        "defender_speed": config.v_d,
        "detection_radius": config.detection_radius,
        "intercept_radius": config.intercept_radius,
        "threat_radius": config.threat_radius,
    }
    
    # Add Kalman tracking metrics if available
    if tracking_errors:
        metrics["mean_tracking_error"] = float(np.mean(tracking_errors))
        metrics["final_tracking_error"] = float(final_tracking_error) if final_tracking_error is not None else float("nan")
    
    return metrics


def run_episode_with_trajectory(env, policy: Policy, seed: int) -> tuple[dict, Trajectory]:
    """
    Run a single episode and capture full trajectory for visualization.
    
    Similar to run_episode but additionally records position history
    for all agents at each timestep.
    
    Args:
        env: A Gymnasium-compatible environment instance.
        policy: A policy with `act(obs, info)` and `reset()` methods.
        seed: Random seed for environment reset.
    
    Returns:
        Tuple of (metrics_dict, Trajectory).
    """
    obs, info = env.reset(seed=seed)
    policy.reset()
    
    # Helper to get e_hat with fallback to NaN (unobserved before detection)
    def get_e_hat(info_dict):
        e_hat = info_dict.get("e_hat")
        if e_hat is not None:
            return e_hat.copy()
        # Before detection, e_hat is unknown - use NaN to distinguish from zeros
        return np.array([np.nan, np.nan], dtype=np.float32)
    
    # Position histories (using correct internal attribute names)
    soldier_positions = [env._soldier_pos.copy()]
    defender_positions = [env._defender_pos.copy()]
    enemy_positions = [env._enemy_pos.copy()]
    estimated_enemy_positions = [get_e_hat(info)]
    
    # Track metrics
    total_reward = 0.0
    min_enemy_soldier_dist = info["enemy_soldier_dist"]
    min_defender_enemy_dist = info["defender_enemy_dist"]
    detection_time = None
    intercept_time = None
    
    # Kalman tracking error (when available)
    tracking_errors = []
    final_tracking_error = None
    
    step = 0
    done = False
    
    while not done:
        action = policy.act(obs, info)
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        total_reward += reward
        
        # Record positions (using correct internal attribute names)
        soldier_positions.append(env._soldier_pos.copy())
        defender_positions.append(env._defender_pos.copy())
        enemy_positions.append(env._enemy_pos.copy())
        estimated_enemy_positions.append(get_e_hat(info))
        
        # Update minimum distances
        min_enemy_soldier_dist = min(min_enemy_soldier_dist, info["enemy_soldier_dist"])
        min_defender_enemy_dist = min(min_defender_enemy_dist, info["defender_enemy_dist"])
        
        # Track detection timing
        if info["enemy_detected"] and detection_time is None:
            detection_time = step
        
        # Track Kalman tracking error (only when detected, i.e., tracking_error is not None)
        if info.get("tracking_error") is not None:
            tracking_errors.append(info["tracking_error"])
            final_tracking_error = info["tracking_error"]
        
        if done or truncated:
            break
    
    # Record intercept time if successful
    if info["outcome"] == "intercepted":
        intercept_time = step
    
    # Get config values
    config = env.config
    
    # Build metrics dict (same as run_episode)
    metrics = {
        "seed": seed,
        "outcome": info["outcome"],
        "success": 1 if info["outcome"] == "intercepted" else 0,
        "episode_length": step,
        "total_reward": total_reward,
        "detected": 1 if detection_time is not None else 0,
        "detection_time": detection_time if detection_time is not None else -1,
        "intercept_time": intercept_time if intercept_time is not None else -1,
        "min_enemy_soldier_dist": min_enemy_soldier_dist,
        "min_defender_enemy_dist": min_defender_enemy_dist,
        "final_enemy_soldier_dist": info["enemy_soldier_dist"],
        "final_defender_enemy_dist": info["defender_enemy_dist"],
        "enemy_speed": config.v_e,
        "defender_speed": config.v_d,
        "detection_radius": config.detection_radius,
        "intercept_radius": config.intercept_radius,
        "threat_radius": config.threat_radius,
    }
    
    # Add Kalman tracking metrics if available
    if tracking_errors:
        metrics["mean_tracking_error"] = float(np.mean(tracking_errors))
        metrics["final_tracking_error"] = float(final_tracking_error) if final_tracking_error is not None else float("nan")
    
    # Build trajectory
    trajectory = Trajectory(
        seed=seed,
        outcome=info["outcome"],
        episode_length=step,
        soldier_positions=np.array(soldier_positions),
        defender_positions=np.array(defender_positions),
        enemy_positions=np.array(enemy_positions),
        estimated_enemy_positions=np.array(estimated_enemy_positions),
        detection_time=detection_time if detection_time is not None else -1,
        intercept_time=intercept_time if intercept_time is not None else -1,
        total_reward=total_reward,
    )
    
    return metrics, trajectory


def select_representative_trajectories(
    trajectories: list[Trajectory],
    results_df: pd.DataFrame,
) -> dict[str, Trajectory]:
    """
    Select representative trajectories for qualitative analysis.
    
    Selects:
        - "success": A clean successful intercept
        - "failure": A clear failure (soldier caught)
        - "borderline": A late intercept (if available)
    
    Args:
        trajectories: List of captured trajectories.
        results_df: DataFrame with episode results (for filtering).
    
    Returns:
        Dictionary mapping category to Trajectory.
    """
    selected = {}
    
    # Index trajectories by seed for lookup
    traj_by_seed = {t.seed: t for t in trajectories}
    
    # 1. Success: find a cleanly successful episode (early intercept)
    success_df = results_df[results_df["outcome"] == "intercepted"]
    if len(success_df) > 0:
        # Prefer median intercept time for "typical" success
        median_intercept = success_df["intercept_time"].median()
        success_df = success_df.copy()
        success_df["dist_to_median"] = abs(success_df["intercept_time"] - median_intercept)
        best_success_seed = success_df.sort_values("dist_to_median").iloc[0]["seed"]
        if best_success_seed in traj_by_seed:
            selected["success"] = traj_by_seed[int(best_success_seed)]
    
    # 2. Failure: clear failure (soldier caught)
    failure_df = results_df[results_df["outcome"] == "soldier_caught"]
    if len(failure_df) > 0:
        # Prefer median episode length for "typical" failure
        median_length = failure_df["episode_length"].median()
        failure_df = failure_df.copy()
        failure_df["dist_to_median"] = abs(failure_df["episode_length"] - median_length)
        best_failure_seed = failure_df.sort_values("dist_to_median").iloc[0]["seed"]
        if best_failure_seed in traj_by_seed:
            selected["failure"] = traj_by_seed[int(best_failure_seed)]
    
    # 3. Borderline: late intercept (90th percentile intercept time)
    if len(success_df) > 0:
        p90_intercept = success_df["intercept_time"].quantile(0.9)
        borderline_df = success_df[success_df["intercept_time"] >= p90_intercept]
        if len(borderline_df) > 0:
            # Pick first borderline case
            borderline_seed = borderline_df.iloc[0]["seed"]
            if borderline_seed in traj_by_seed:
                selected["borderline"] = traj_by_seed[int(borderline_seed)]
    
    return selected


def save_trajectories(
    trajectories: dict[str, Trajectory],
    output_dir: str,
    policy_name: str = "",
) -> list[str]:
    """
    Save representative trajectories to disk.
    
    Args:
        trajectories: Dict mapping category name to Trajectory.
        output_dir: Directory to save trajectories.
        policy_name: Optional policy name prefix.
    
    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for category, traj in trajectories.items():
        prefix = f"{policy_name}_" if policy_name else ""
        filename = f"{prefix}{category}_seed{traj.seed}.npz"
        filepath = os.path.join(output_dir, filename)
        traj.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths


def evaluate_policy(
    env_factory: Callable,
    policy: Policy,
    seeds: Iterable[int],
    verbose: bool = True,
    progress_interval: int = 50,
    capture_trajectories: bool = False,
) -> tuple[pd.DataFrame, dict, list[Trajectory] | None]:
    """
    Evaluate a policy over multiple episodes with different seeds.
    
    This is the main evaluation function. It runs episodes in sequence,
    collects results, and returns both raw data and summary statistics.
    
    Args:
        env_factory: Callable that returns a new environment instance.
                    Example: `lambda: SoldierEnv()`
        policy: Policy instance with `act(obs, info)` and `reset()` methods.
        seeds: Iterable of integer seeds for each episode.
        verbose: If True, print progress updates.
        progress_interval: Print progress every N episodes.
        capture_trajectories: If True, capture full trajectory data for
                             later selection and visualization.
    
    Returns:
        Tuple of (df, summary, trajectories):
            - df: DataFrame with one row per episode
            - summary: Dictionary of aggregated statistics
            - trajectories: List of Trajectory objects (if capture_trajectories=True)
                           or None otherwise
    
    Example:
        >>> df, summary, _ = evaluate_policy(
        ...     env_factory=lambda: SoldierEnv(),
        ...     policy=GreedyInterceptPolicy(),
        ...     seeds=range(100),
        ... )
        >>> print(f"Success rate: {summary['success_rate']:.1%}")
    """
    seeds = list(seeds)
    n_episodes = len(seeds)
    
    # Create environment once (will be reset each episode)
    env = env_factory()
    
    if verbose:
        policy_name = type(policy).__name__
        print("=" * 60)
        print("POLICY EVALUATION")
        print("=" * 60)
        print(f"Policy:       {policy_name}")
        print(f"Episodes:     {n_episodes}")
        print(f"Seed range:   [{min(seeds)}, {max(seeds)}]")
        if capture_trajectories:
            print(f"Trajectories: capturing")
        print("-" * 60)
    
    results = []
    trajectories = [] if capture_trajectories else None
    start_time = time.time()
    
    for i, seed in enumerate(seeds):
        if capture_trajectories:
            result, traj = run_episode_with_trajectory(env, policy, seed)
            trajectories.append(traj)
        else:
            result = run_episode(env, policy, seed)
        results.append(result)
        
        # Progress update
        if verbose and (i + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_episodes - i - 1) / rate if rate > 0 else 0
            current_success = sum(r["success"] for r in results) / len(results)
            print(f"  Episode {i+1:4d}/{n_episodes}: "
                  f"success_rate={current_success:.1%}, "
                  f"ETA={eta:.1f}s")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print("-" * 60)
        print(f"Completed {n_episodes} episodes in {elapsed:.2f}s "
              f"({n_episodes/elapsed:.1f} eps/sec)")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Compute summary statistics
    summary = summarize_results(df)
    
    return df, summary, trajectories


def summarize_results(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics from evaluation results.
    
    Handles missing values gracefully (e.g., detection_time=-1 for
    episodes where detection never occurred).
    
    Args:
        df: DataFrame with episode results from evaluate_policy or run_episode.
    
    Returns:
        Dictionary containing:
            - n_episodes: Total number of episodes
            - outcome_counts: Dict of outcome -> count
            - success_rate: Fraction of successful episodes
            - failure_rate: Fraction of failed episodes (caught + unsafe)
            - timeout_rate: Fraction of timeout episodes
            - mean_episode_length: Average episode length
            - std_episode_length: Std dev of episode length
            - mean_total_reward: Average cumulative reward
            - detection_rate: Fraction of episodes with detection
            - mean_detection_time: Average detection time (detected only)
            - mean_intercept_time: Average intercept time (successful only)
            - mean_min_enemy_soldier_dist: Average minimum d(e,s)
            - mean_min_defender_enemy_dist: Average minimum d(d,e)
    """
    n_episodes = len(df)
    
    if n_episodes == 0:
        return {
            "n_episodes": 0,
            "outcome_counts": {},
            "success_rate": float("nan"),
            "failure_rate": float("nan"),
            "timeout_rate": float("nan"),
            "mean_episode_length": float("nan"),
            "std_episode_length": float("nan"),
            "mean_total_reward": float("nan"),
            "detection_rate": float("nan"),
            "mean_detection_time": float("nan"),
            "mean_intercept_time": float("nan"),
            "mean_min_enemy_soldier_dist": float("nan"),
            "mean_min_defender_enemy_dist": float("nan"),
        }
    
    # Outcome counts
    outcome_counts = df["outcome"].value_counts().to_dict()
    n_success = outcome_counts.get("intercepted", 0)
    n_soldier_caught = outcome_counts.get("soldier_caught", 0)
    n_unsafe = outcome_counts.get("unsafe_intercept", 0)
    n_timeout = outcome_counts.get("timeout", 0)
    
    # Rates
    success_rate = n_success / n_episodes
    failure_rate = (n_soldier_caught + n_unsafe) / n_episodes
    timeout_rate = n_timeout / n_episodes
    
    # Episode length stats
    mean_length = df["episode_length"].mean()
    std_length = df["episode_length"].std()
    
    # Reward stats
    mean_reward = df["total_reward"].mean() if "total_reward" in df.columns else float("nan")
    
    # Detection stats (filter out episodes where detection didn't occur: detection_time == -1)
    detected_df = df[df["detected"] == 1]
    if len(detected_df) > 0:
        # Filter valid detection times (> 0)
        valid_detection_times = detected_df["detection_time"][detected_df["detection_time"] > 0]
        mean_detection_time = valid_detection_times.mean() if len(valid_detection_times) > 0 else float("nan")
        detection_rate = len(detected_df) / n_episodes
    else:
        mean_detection_time = float("nan")
        detection_rate = 0.0
    
    # Intercept stats (only for successful episodes with valid intercept_time)
    success_df = df[df["success"] == 1]
    if len(success_df) > 0:
        valid_intercept_times = success_df["intercept_time"][success_df["intercept_time"] > 0]
        mean_intercept_time = valid_intercept_times.mean() if len(valid_intercept_times) > 0 else float("nan")
    else:
        mean_intercept_time = float("nan")
    
    # Distance stats
    mean_min_es = df["min_enemy_soldier_dist"].mean()
    mean_min_de = df["min_defender_enemy_dist"].mean()
    
    return {
        "n_episodes": n_episodes,
        "outcome_counts": outcome_counts,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "timeout_rate": timeout_rate,
        "mean_episode_length": mean_length,
        "std_episode_length": std_length,
        "mean_total_reward": mean_reward,
        "detection_rate": detection_rate,
        "mean_detection_time": mean_detection_time,
        "mean_intercept_time": mean_intercept_time,
        "mean_min_enemy_soldier_dist": mean_min_es,
        "mean_min_defender_enemy_dist": mean_min_de,
    }


def print_summary(summary: dict, title: str = "SUMMARY STATISTICS") -> None:
    """
    Print formatted summary statistics.
    
    Args:
        summary: Dictionary from summarize_results().
        title: Header title for the printout.
    """
    n = summary["n_episodes"]
    oc = summary["outcome_counts"]
    
    n_success = oc.get("intercepted", 0)
    n_soldier_caught = oc.get("soldier_caught", 0)
    n_unsafe = oc.get("unsafe_intercept", 0)
    n_timeout = oc.get("timeout", 0)
    
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    print(f"\n{'Outcome Distribution':}")
    print(f"  Intercepted (WIN):     {n_success:4d} ({n_success/n:6.1%})")
    print(f"  Soldier caught:        {n_soldier_caught:4d} ({n_soldier_caught/n:6.1%})")
    print(f"  Unsafe intercept:      {n_unsafe:4d} ({n_unsafe/n:6.1%})")
    print(f"  Timeout:               {n_timeout:4d} ({n_timeout/n:6.1%})")
    
    print(f"\n{'Aggregate Rates':}")
    print(f"  Success rate:          {summary['success_rate']:6.1%}")
    print(f"  Failure rate:          {summary['failure_rate']:6.1%}")
    print(f"  Timeout rate:          {summary['timeout_rate']:6.1%}")
    
    print(f"\n{'Episode Length':}")
    print(f"  Mean:                  {summary['mean_episode_length']:6.1f} steps")
    print(f"  Std:                   {summary['std_episode_length']:6.1f} steps")
    
    if not np.isnan(summary["mean_total_reward"]):
        print(f"\n{'Reward':}")
        print(f"  Mean total reward:     {summary['mean_total_reward']:6.1f}")
    
    print(f"\n{'Detection':}")
    print(f"  Detection rate:        {summary['detection_rate']:6.1%}")
    if not np.isnan(summary["mean_detection_time"]):
        print(f"  Mean detection time:   {summary['mean_detection_time']:6.1f} steps")
    
    print(f"\n{'Intercept (successful only)':}")
    if not np.isnan(summary["mean_intercept_time"]):
        print(f"  Mean intercept time:   {summary['mean_intercept_time']:6.1f} steps")
    else:
        print(f"  Mean intercept time:   N/A (no successful intercepts)")
    
    print(f"\n{'Distance Metrics':}")
    print(f"  Mean min d(e,s):       {summary['mean_min_enemy_soldier_dist']:6.2f}")
    print(f"  Mean min d(d,e):       {summary['mean_min_defender_enemy_dist']:6.2f}")
    
    print("=" * 60)


def format_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format a results DataFrame to the standard comparison schema.
    
    This ensures baseline and RL evaluation CSVs have identical columns
    for one-to-one comparison. The schema is designed for reproducibility
    and easy diff/merge.
    
    Standard comparison columns (in order):
        seed, outcome, success, episode_length, detected, detection_time,
        intercept_time, enemy_speed, defender_speed, detection_radius,
        intercept_radius, threat_radius, min_enemy_soldier_dist,
        min_defender_enemy_dist, num_episodes
    
    Args:
        df: Raw results DataFrame from evaluate_policy().
    
    Returns:
        DataFrame with exactly the comparison columns, in order.
    """
    # Add num_episodes column (same value for all rows)
    result = df.copy()
    result["num_episodes"] = len(df)
    
    # Define the exact column order for comparison
    comparison_columns = [
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
    ]
    
    # Select only the comparison columns (ignore extras like total_reward)
    return result[comparison_columns]


def format_rl_kalman_comparison_df(
    df: pd.DataFrame,
    model_version: str = "unknown",
) -> pd.DataFrame:
    """
    Format a results DataFrame to the RL-Kalman comparison schema.
    
    Extends the standard comparison schema with Kalman-specific columns:
    - mean_tracking_error: Average tracking error during episode
    - final_tracking_error: Tracking error at episode end
    - model_version: Version identifier for the trained model
    
    Standard columns (in order):
        seed, outcome, success, episode_length, detected, detection_time,
        intercept_time, enemy_speed, defender_speed, detection_radius,
        intercept_radius, threat_radius, min_enemy_soldier_dist,
        min_defender_enemy_dist, num_episodes, model_version,
        mean_tracking_error, final_tracking_error
    
    Args:
        df: Raw results DataFrame from evaluate_policy() with Kalman env.
        model_version: Model version string for identification.
    
    Returns:
        DataFrame with RL-Kalman comparison columns, in order.
    """
    result = df.copy()
    result["num_episodes"] = len(df)
    result["model_version"] = model_version
    
    # Fill missing Kalman columns with NaN if not present
    if "mean_tracking_error" not in result.columns:
        result["mean_tracking_error"] = float("nan")
    if "final_tracking_error" not in result.columns:
        result["final_tracking_error"] = float("nan")
    
    # Define the exact column order for RL-Kalman comparison
    comparison_columns = [
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
        "model_version",
        "mean_tracking_error",
        "final_tracking_error",
    ]
    
    return result[comparison_columns]


def compare_policies(
    env_factory: Callable,
    policies: dict[str, Policy],
    seeds: Iterable[int],
    verbose: bool = True,
    capture_trajectories: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict], dict[str, list[Trajectory]] | None]:
    """
    Compare multiple policies on the same set of seeds.
    
    Useful for fair comparison between baseline and RL policies.
    
    Args:
        env_factory: Callable that returns a new environment instance.
        policies: Dictionary mapping policy names to policy instances.
        seeds: Iterable of seeds (same seeds used for all policies).
        verbose: If True, print progress.
        capture_trajectories: If True, capture trajectories for each policy.
    
    Returns:
        Tuple of (dfs, summaries, trajectories_dict):
            - dfs: Dict mapping policy name to results DataFrame
            - summaries: Dict mapping policy name to summary dict
            - trajectories_dict: Dict mapping policy name to list of Trajectory
                                (if capture_trajectories=True), else None
    """
    seeds = list(seeds)
    dfs = {}
    summaries = {}
    trajectories_dict = {} if capture_trajectories else None
    
    for name, policy in policies.items():
        if verbose:
            print(f"\n>>> Evaluating: {name}")
        
        df, summary, trajectories = evaluate_policy(
            env_factory=env_factory,
            policy=policy,
            seeds=seeds,
            verbose=verbose,
            capture_trajectories=capture_trajectories,
        )
        
        dfs[name] = df
        summaries[name] = summary
        if capture_trajectories:
            trajectories_dict[name] = trajectories
        
        if verbose:
            print_summary(summary, title=f"SUMMARY: {name}")
    
    return dfs, summaries, trajectories_dict
