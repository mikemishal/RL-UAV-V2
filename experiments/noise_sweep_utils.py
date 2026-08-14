"""Shared helpers for measurement-noise sweep experiments.

These utilities keep the four-method noise sweeps aligned on:
- Seed generation
- Monte Carlo summary row schema
- Standard error and 95% confidence interval calculation
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

Z_95 = 1.96


def make_eval_seeds(seed_offset: int, n_episodes: int) -> range:
    """Return the shared deterministic seed range used by all sweeps."""
    return range(seed_offset, seed_offset + n_episodes)


def binomial_stats(p: float, n: int) -> tuple[float, float, float]:
    """Return (se, ci_low, ci_high) for a binomial proportion."""
    if n <= 0:
        return 0.0, float("nan"), float("nan")

    se = float(np.sqrt(p * (1.0 - p) / n))
    ci_low = max(0.0, p - Z_95 * se)
    ci_high = min(1.0, p + Z_95 * se)
    return se, ci_low, ci_high


def build_noise_summary_row(
    *,
    policy_name: str,
    noise_value: float,
    summary: dict,
) -> dict:
    """Build a shared schema row from evaluate_policy summary output."""
    n = int(summary["n_episodes"])
    p_success = float(summary["success_rate"])
    p_failure = float(summary["failure_rate"])

    success_se, success_ci_low, success_ci_high = binomial_stats(p_success, n)
    failure_se, failure_ci_low, failure_ci_high = binomial_stats(p_failure, n)

    return {
        "policy_name": policy_name,
        "noise_value": float(noise_value),
        "measurement_var": float(noise_value),
        "num_episodes": n,
        "n_episodes": n,
        "n_success": int(round(p_success * n)),
        "success_rate": p_success,
        "failure_rate": p_failure,
        "timeout_rate": float(summary.get("timeout_rate", float("nan"))),
        "mean_episode_length": float(summary["mean_episode_length"]),
        "std_episode_length": float(summary["std_episode_length"]),
        "standard_error": success_se,
        "success_se": success_se,
        "success_ci_low": success_ci_low,
        "success_ci_high": success_ci_high,
        "failure_se": failure_se,
        "failure_ci_low": failure_ci_low,
        "failure_ci_high": failure_ci_high,
    }


def extract_tracking_metrics(df: pd.DataFrame) -> dict:
    """Extract tracking metrics from episode-level DataFrame when available."""
    metrics = {
        "mean_tracking_error": float("nan"),
        "final_tracking_error": float("nan"),
    }

    if "mean_tracking_error" in df.columns:
        metrics["mean_tracking_error"] = float(df["mean_tracking_error"].mean(skipna=True))
    if "final_tracking_error" in df.columns:
        metrics["final_tracking_error"] = float(df["final_tracking_error"].mean(skipna=True))

    return metrics
