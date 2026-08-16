"""
Correction script for the measurement-noise estimator-aggregation bug.

Reads the ALREADY-LOCKED raw_episode_results.csv from
results/robustness/measurement_noise/ (READ-ONLY; never modified) and
recomputes CORRECTED estimator-summary statistics using
experiments/estimator_stats.py's sample-weighted (pooled, timestep-correct)
aggregation. Writes all outputs to results/corrections/measurement_noise_estimator/.

NO EPISODES ARE RERUN. NO NEW SEEDS ARE EVALUATED. This is an
analysis/reporting-only correction of experiments/run_robustness_measurement_noise_sweep.py's
compute_estimator_summary(), which incorrectly computed
sqrt(mean(per_episode_mean_error^2)) instead of reconstructing the true
pooled RMSE from each episode's own (n_i, m_i, r_i).

Usage:
    python experiments/run_measurement_noise_estimator_correction.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robustness_measurement_noise_protocol import (
    MEASUREMENT_VAR_VALUES,
    MEASUREMENT_SWEEP_SEED_OFFSET,
    MEASUREMENT_SWEEP_EPISODES,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    measurement_std,
)
from experiments.estimator_stats import (
    theoretical_raw_measurement_mean_error,
    theoretical_raw_measurement_rmse,
    sample_weighted_mean_error,
    sample_weighted_rmse,
    episode_weighted_mean_error,
    mean_episode_rmse,
    equal_weight_across_seeds,
)

ORIGINAL_SWEEP_COMMIT = "e5632d75cfc93decae942f1330427f436d7b8522"
RAW_DATA_PATH = PROJECT_ROOT / "results" / "robustness" / "measurement_noise" / "raw_episode_results.csv"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "corrections" / "measurement_noise_estimator"

EXPECTED_ROW_COUNT = 60_000
EXPECTED_SEED_RANGE = (MEASUREMENT_SWEEP_SEED_OFFSET, MEASUREMENT_SWEEP_SEED_OFFSET + MEASUREMENT_SWEEP_EPISODES - 1)


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def load_raw_data() -> pd.DataFrame:
    """Load the locked raw episode results READ-ONLY -- never write back to this path."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Locked raw data not found: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    assert len(df) == EXPECTED_ROW_COUNT, f"expected {EXPECTED_ROW_COUNT} rows, found {len(df)} -- raw file may have changed"
    seed_min, seed_max = int(df["eval_seed"].min()), int(df["eval_seed"].max())
    assert (seed_min, seed_max) == EXPECTED_SEED_RANGE, f"seed range changed: found {seed_min}..{seed_max}"
    return df


def compute_corrected_estimator_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variance in MEASUREMENT_VAR_VALUES:
        theoretical_mean = theoretical_raw_measurement_mean_error(variance)
        theoretical_rmse = theoretical_raw_measurement_rmse(variance)
        for method in SCRIPTED_METHODS:
            sub = df[(df["measurement_var"] == variance) & (df["policy_instance"] == method)].dropna(
                subset=["mean_position_estimation_error", "position_estimation_rmse"]
            )
            n = sub["n_position_estimation_samples"].to_numpy(dtype=np.float64)
            m = sub["mean_position_estimation_error"].to_numpy(dtype=np.float64)
            r = sub["position_estimation_rmse"].to_numpy(dtype=np.float64)
            is_direct = method in ("greedy", "pn", "lead")
            rows.append({
                "measurement_var": variance, "measurement_std": measurement_std(variance), "method": method,
                "n_episodes_with_estimates": int(len(sub)), "n_total_estimation_samples": int(n.sum()),
                "mean_error_episode_weighted": episode_weighted_mean_error(m),
                "mean_error_sample_weighted": sample_weighted_mean_error(n, m),
                "mean_episode_rmse": mean_episode_rmse(r),
                "rmse_sample_weighted": sample_weighted_rmse(n, r),
                "theoretical_raw_mean_error": theoretical_mean if is_direct else float("nan"),
                "theoretical_raw_rmse": theoretical_rmse if is_direct else float("nan"),
                "theoretical_benchmark_note": (
                    "Direct/raw-measurement theoretical benchmark" if is_direct
                    else "RAW SENSOR theoretical benchmark for reference only -- NOT a Kalman prediction benchmark"
                ),
            })
    return pd.DataFrame(rows)


def compute_corrected_ppo_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variance in MEASUREMENT_VAR_VALUES:
        for method in PPO_METHODS:
            sub_method = df[(df["measurement_var"] == variance) & (df["paper_method"] == method)]
            for seed in (42, 43, 44):
                sub = sub_method[sub_method["training_seed"] == seed].dropna(
                    subset=["mean_position_estimation_error", "position_estimation_rmse"]
                )
                n = sub["n_position_estimation_samples"].to_numpy(dtype=np.float64)
                m = sub["mean_position_estimation_error"].to_numpy(dtype=np.float64)
                r = sub["position_estimation_rmse"].to_numpy(dtype=np.float64)
                rows.append({
                    "measurement_var": variance, "method": method, "training_seed": seed,
                    "n_total_estimation_samples": int(n.sum()),
                    "mean_error_sample_weighted": sample_weighted_mean_error(n, m),
                    "rmse_sample_weighted": sample_weighted_rmse(n, r),
                    "mean_error_episode_weighted": episode_weighted_mean_error(m),
                    "mean_episode_rmse": mean_episode_rmse(r),
                })
    return pd.DataFrame(rows)


def compute_corrected_ppo_paper_level_summary(ppo_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Paper-level PPO/PPO-Kalman estimator stats: EQUAL-WEIGHT mean of the 3 per-seed statistics."""
    rows = []
    for variance in MEASUREMENT_VAR_VALUES:
        theoretical_mean = theoretical_raw_measurement_mean_error(variance)
        theoretical_rmse = theoretical_raw_measurement_rmse(variance)
        for method in PPO_METHODS:
            sub = ppo_seed_df[(ppo_seed_df["measurement_var"] == variance) & (ppo_seed_df["method"] == method)]
            assert len(sub) == 3, f"expected 3 training-seed rows, found {len(sub)}"
            mean_error, mean_error_sd = equal_weight_across_seeds(sub["mean_error_sample_weighted"].to_numpy())
            rmse, rmse_sd = equal_weight_across_seeds(sub["rmse_sample_weighted"].to_numpy())
            rows.append({
                "measurement_var": variance, "measurement_std": measurement_std(variance), "method": method,
                "n_total_estimation_samples": int(sub["n_total_estimation_samples"].sum()),
                "mean_error_sample_weighted": mean_error,
                "train_seed_mean_error_sd": mean_error_sd,
                "rmse_sample_weighted": rmse,
                "train_seed_rmse_sd": rmse_sd,
                "theoretical_raw_mean_error": theoretical_mean if method == "ppo" else float("nan"),
                "theoretical_raw_rmse": theoretical_rmse if method == "ppo" else float("nan"),
            })
    return pd.DataFrame(rows)


def compute_sensor_sanity_check(estimator_summary: pd.DataFrame, ppo_paper_summary: pd.DataFrame) -> pd.DataFrame:
    """Direct-track (greedy/pn/lead/ppo) observed vs theoretical mean-error and RMSE."""
    rows = []
    for variance in MEASUREMENT_VAR_VALUES:
        theoretical_mean = theoretical_raw_measurement_mean_error(variance)
        theoretical_rmse = theoretical_raw_measurement_rmse(variance)
        for method in ("greedy", "pn", "lead"):
            row = estimator_summary[
                (estimator_summary["measurement_var"] == variance) & (estimator_summary["method"] == method)
            ].iloc[0]
            observed_mean = row["mean_error_sample_weighted"]
            observed_rmse = row["rmse_sample_weighted"]
            rows.append({
                "measurement_var": variance, "method": method,
                "observed_mean_error": observed_mean, "theoretical_mean_error": theoretical_mean,
                "mean_error_abs_diff": observed_mean - theoretical_mean,
                "mean_error_rel_diff_pct": 100.0 * (observed_mean - theoretical_mean) / theoretical_mean,
                "observed_rmse": observed_rmse, "theoretical_rmse": theoretical_rmse,
                "rmse_abs_diff": observed_rmse - theoretical_rmse,
                "rmse_rel_diff_pct": 100.0 * (observed_rmse - theoretical_rmse) / theoretical_rmse,
            })
        ppo_row = ppo_paper_summary[
            (ppo_paper_summary["measurement_var"] == variance) & (ppo_paper_summary["method"] == "ppo")
        ].iloc[0]
        observed_mean = ppo_row["mean_error_sample_weighted"]
        observed_rmse = ppo_row["rmse_sample_weighted"]
        rows.append({
            "measurement_var": variance, "method": "ppo",
            "observed_mean_error": observed_mean, "theoretical_mean_error": theoretical_mean,
            "mean_error_abs_diff": observed_mean - theoretical_mean,
            "mean_error_rel_diff_pct": 100.0 * (observed_mean - theoretical_mean) / theoretical_mean,
            "observed_rmse": observed_rmse, "theoretical_rmse": theoretical_rmse,
            "rmse_abs_diff": observed_rmse - theoretical_rmse,
            "rmse_rel_diff_pct": 100.0 * (observed_rmse - theoretical_rmse) / theoretical_rmse,
        })
    return pd.DataFrame(rows)


def compute_estimator_comparisons(estimator_summary: pd.DataFrame, ppo_paper_summary: pd.DataFrame) -> pd.DataFrame:
    """Kalman-vs-raw RMSE comparison (descriptive), using the CORRECTED rmse_sample_weighted."""
    rows = []
    for variance in MEASUREMENT_VAR_VALUES:
        greedy_rmse = estimator_summary[
            (estimator_summary["measurement_var"] == variance) & (estimator_summary["method"] == "greedy")
        ]["rmse_sample_weighted"].iloc[0]
        kalman_greedy_rmse = estimator_summary[
            (estimator_summary["measurement_var"] == variance) & (estimator_summary["method"] == "kalman_greedy")
        ]["rmse_sample_weighted"].iloc[0]
        ppo_rmse = ppo_paper_summary[
            (ppo_paper_summary["measurement_var"] == variance) & (ppo_paper_summary["method"] == "ppo")
        ]["rmse_sample_weighted"].iloc[0]
        ppo_kalman_rmse = ppo_paper_summary[
            (ppo_paper_summary["measurement_var"] == variance) & (ppo_paper_summary["method"] == "ppo_kalman")
        ]["rmse_sample_weighted"].iloc[0]
        rows.append({
            "measurement_var": variance,
            "greedy_raw_rmse": greedy_rmse, "kalman_greedy_rmse": kalman_greedy_rmse,
            "kalman_greedy_minus_greedy_rmse": kalman_greedy_rmse - greedy_rmse,
            "kalman_greedy_beats_greedy": bool(kalman_greedy_rmse < greedy_rmse),
            "ppo_raw_rmse": ppo_rmse, "ppo_kalman_rmse": ppo_kalman_rmse,
            "ppo_kalman_minus_ppo_rmse": ppo_kalman_rmse - ppo_rmse,
            "ppo_kalman_beats_ppo": bool(ppo_kalman_rmse < ppo_rmse),
        })
    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 70)
    print("MEASUREMENT-NOISE ESTIMATOR-AGGREGATION CORRECTION (analysis-only)")
    print("=" * 70)
    print(f"Reading locked raw data (READ-ONLY): {RAW_DATA_PATH}")
    df = load_raw_data()
    print(f"  {len(df)} rows loaded; seed range {df['eval_seed'].min()}..{df['eval_seed'].max()}")

    print("\nComputing corrected scripted-method estimator summary...")
    estimator_summary = compute_corrected_estimator_summary(df)

    print("Computing corrected PPO per-training-seed estimator summary...")
    ppo_seed_summary = compute_corrected_ppo_seed_summary(df)

    print("Computing corrected PPO paper-level (equal-weight) estimator summary...")
    ppo_paper_summary = compute_corrected_ppo_paper_level_summary(ppo_seed_summary)

    # Merge scripted + PPO paper-level rows into one estimator_summary table.
    ppo_for_merge = ppo_paper_summary.rename(columns={
        "train_seed_mean_error_sd": "_train_seed_mean_error_sd",
        "train_seed_rmse_sd": "_train_seed_rmse_sd",
    })
    for col in ("n_episodes_with_estimates", "mean_error_episode_weighted", "mean_episode_rmse", "theoretical_benchmark_note"):
        ppo_for_merge[col] = float("nan") if col != "theoretical_benchmark_note" else "PPO paper-level = equal-weight mean of 3 training-seed sample-weighted statistics"
    combined_estimator_summary = pd.concat([estimator_summary, ppo_for_merge], ignore_index=True, sort=False)

    print("Computing sensor sanity check (Direct raw vs theoretical)...")
    sensor_sanity = compute_sensor_sanity_check(estimator_summary, ppo_paper_summary)

    print("Computing Kalman-vs-raw estimator comparisons (descriptive)...")
    comparisons = compute_estimator_comparisons(estimator_summary, ppo_paper_summary)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined_estimator_summary.to_csv(OUTPUT_ROOT / "corrected_estimator_summary.csv", index=False)
    ppo_seed_summary.to_csv(OUTPUT_ROOT / "corrected_ppo_estimator_seed_summary.csv", index=False)
    sensor_sanity.to_csv(OUTPUT_ROOT / "corrected_sensor_sanity_check.csv", index=False)
    comparisons.to_csv(OUTPUT_ROOT / "corrected_estimator_comparisons.csv", index=False)

    max_rel_diff = sensor_sanity[["mean_error_rel_diff_pct", "rmse_rel_diff_pct"]].abs().max().max()
    large_mismatch = bool(max_rel_diff > 5.0)
    if large_mismatch:
        print(f"\n*** WARNING: unexplained sensor sanity-check mismatch of {max_rel_diff:.2f}% detected. STOP and report. ***")

    manifest = {
        "original_sweep_commit": ORIGINAL_SWEEP_COMMIT,
        "correction_source_commit": _get_git_commit(),
        "correction_branch": _get_git_branch(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Estimator RMSE aggregation used squared per-episode mean errors rather than "
            "the recorded per-episode squared-error RMSE statistic."
        ),
        "raw_data_source": str(RAW_DATA_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "episodes_rerun": "NO EPISODES RERUN",
        "seed_range": f"{EXPECTED_SEED_RANGE[0]}..{EXPECTED_SEED_RANGE[1]}",
        "raw_row_count": EXPECTED_ROW_COUNT,
        "measurement_var_values": list(MEASUREMENT_VAR_VALUES),
        "primary_paper_methods": list(PRIMARY_PAPER_METHODS),
        "max_sensor_sanity_check_rel_diff_pct": max_rel_diff,
        "large_unexplained_mismatch_flag": large_mismatch,
    }
    (OUTPUT_ROOT / "correction_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    final_summary = {
        "label": "MEASUREMENT-NOISE ESTIMATOR-AGGREGATION CORRECTION",
        "manifest": manifest,
        "estimator_summary": combined_estimator_summary.to_dict(orient="records"),
        "ppo_estimator_seed_summary": ppo_seed_summary.to_dict(orient="records"),
        "sensor_sanity_check": sensor_sanity.to_dict(orient="records"),
        "estimator_comparisons": comparisons.to_dict(orient="records"),
    }
    (OUTPUT_ROOT / "corrected_estimator_summary.json").write_text(json.dumps(final_summary, indent=2, default=str))

    print(f"\nAll corrected outputs written to {OUTPUT_ROOT}")
    print("=" * 70)
    return 1 if large_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
