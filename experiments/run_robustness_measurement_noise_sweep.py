"""
Locked measurement-noise robustness sweep runner.

Evaluates the SIX primary paper methods (10 concrete policy instances: 4
scripted + 3 PPO + 3 PPO-Kalman, same frozen 400k PPO models as the final
nominal experiment) across 6 measurement-variance values, on a previously
untouched seed range. measurement_var is the ONLY environment parameter that
changes (process_var/lead_time are held FIXED at their nominal values, and
the Kalman filter's declared measurement covariance R is always set to the
SAME swept measurement_var -- this is not filter tuning). PPO/Kalman remain
exactly as trained/frozen -- zero-shot sensing-noise generalization test.

Usage:
    # Full locked sweep (seeds 31000-31999, all 6 variances):
    python experiments/run_robustness_measurement_noise_sweep.py

    # Small NON-FINAL dry run (diagnostic seeds < 31000, variance subset):
    python experiments/run_robustness_measurement_noise_sweep.py --seed-start 16000 --n-episodes 5 \
        --variances 0.25 0.5 2.0 --output-root results/_dry_run_robustness_measurement_noise
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig

from experiments.robustness_measurement_noise_protocol import (
    MEASUREMENT_VAR_VALUES,
    NOMINAL_MEASUREMENT_VAR,
    FIXED_PROCESS_VAR,
    FIXED_LEAD_TIME,
    MEASUREMENT_SWEEP_SEED_OFFSET,
    MEASUREMENT_SWEEP_EPISODES,
    MEASUREMENT_SWEEP_SEEDS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    NOISE_POLICY_INSTANCES,
    EXPECTED_N_POLICY_INSTANCES,
    PRIMARY_COMPARISONS,
    SECONDARY_COMPARISONS,
    ALL_COMPARISONS,
    ROBUSTNESS_NOISE_OUTPUT_ROOT,
    EXPERIMENT_MANIFEST_FILENAME,
    MODEL_HASHES_FILENAME,
    RAW_EPISODE_RESULTS_FILENAME,
    NOISE_SUMMARY_FILENAME,
    PPO_TRAINING_SEED_SUMMARY_FILENAME,
    PAIRWISE_COMPARISONS_FILENAME,
    ESTIMATOR_SUMMARY_FILENAME,
    MEASUREMENT_NOISE_SUMMARY_FILENAME,
    measurement_std,
    theoretical_raw_measurement_rmse,
    theoretical_raw_measurement_mean_error,
    build_policy,
    build_sweep_env_config,
    assert_only_measurement_var_differs,
)
from experiments.estimator_stats import (
    sample_weighted_mean_error,
    sample_weighted_rmse,
    episode_weighted_mean_error,
    mean_episode_rmse,
    equal_weight_across_seeds,
)
from experiments.measurement_noise_eval_utils import run_measurement_noise_episode, sha256_of_file
from experiments.final_stats import (
    wilson_interval,
    mcnemar_exact,
    paired_bootstrap_ci,
    hierarchical_bootstrap_track,
    hierarchical_paired_bootstrap_vs_scripted,
    hierarchical_paired_bootstrap_matched_seeds,
)

_NOMINAL_ENV_CONFIG = EnvConfig()
_FINAL_NOMINAL_HASHES_PATH = PROJECT_ROOT / "results" / "final_nominal" / "model_hashes.json"


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


def build_env(estimator: str, measurement_var: float) -> SoldierEnv:
    config = build_sweep_env_config(estimator, measurement_var)
    assert_only_measurement_var_differs(config, _NOMINAL_ENV_CONFIG)
    return SoldierEnv(config=config)


def compute_model_hashes_and_verify() -> list[dict]:
    nominal_hashes_by_path = {}
    if _FINAL_NOMINAL_HASHES_PATH.exists():
        for row in json.loads(_FINAL_NOMINAL_HASHES_PATH.read_text()):
            nominal_hashes_by_path[row["model_path"].replace("\\", "/")] = row["sha256"]

    rows = []
    for instance in NOISE_POLICY_INSTANCES:
        if instance.model_path is None:
            continue
        rel = str(instance.model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if not instance.model_path.exists():
            raise FileNotFoundError(f"Frozen model not found: {instance.model_path}")
        digest = sha256_of_file(instance.model_path)
        nominal_digest = nominal_hashes_by_path.get(rel)
        matches = (nominal_digest == digest) if nominal_digest else None
        if matches is False:
            raise RuntimeError(f"Model hash mismatch vs final nominal manifest for {rel}: {digest} != {nominal_digest}")
        rows.append({
            "paper_method": instance.paper_method, "policy_instance": instance.policy_instance,
            "training_seed": instance.training_seed, "model_path": rel, "sha256": digest,
            "matches_final_nominal_hash": matches,
        })
    return rows


def build_manifest(seeds: list[int], variances: list[float], output_root: Path, model_hashes: list[dict]) -> dict:
    import gymnasium
    import stable_baselines3
    import torch

    opened = len(seeds) > 0 and min(seeds) >= MEASUREMENT_SWEEP_SEED_OFFSET
    env = build_env("measurement", NOMINAL_MEASUREMENT_VAR)
    manifest = {
        "source_git_commit": _get_git_commit(),
        "source_git_branch": _get_git_branch(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gymnasium.__version__,
        "primary_paper_methods": list(PRIMARY_PAPER_METHODS),
        "nominal_env_config": dataclasses.asdict(_NOMINAL_ENV_CONFIG),
        "measurement_var_values": list(variances),
        "measurement_std_values": [measurement_std(v) for v in variances],
        "nominal_measurement_var": NOMINAL_MEASUREMENT_VAR,
        "fixed_process_var": FIXED_PROCESS_VAR,
        "fixed_lead_time": FIXED_LEAD_TIME,
        "ppo_model_hashes": model_hashes,
        "evaluation_seed_start": min(seeds) if seeds else None,
        "evaluation_seed_end": max(seeds) if seeds else None,
        "n_episodes_per_condition": len(seeds),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": list(PRIMARY_COMPARISONS),
        "secondary_comparisons": list(SECONDARY_COMPARISONS),
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "note_only_measurement_var_swept": (
            "Only measurement_var is swept; Kalman process_var and all controller, "
            "reward, dynamics, detection, and evasion parameters are frozen."
        ),
        "note_zero_shot_ppo": (
            "Frozen PPO policies were trained only at measurement_var=0.5; all other "
            "noise conditions are zero-shot robustness tests."
        ),
        "robustness_test_opened": opened,
        "robustness_type": "measurement_noise",
        "robustness_first_seed": MEASUREMENT_SWEEP_SEED_OFFSET if opened else None,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / EXPERIMENT_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, default=str))
    (output_root / MODEL_HASHES_FILENAME).write_text(json.dumps(model_hashes, indent=2, default=str))
    return manifest


def run_evaluation(seeds: list[int], variances: list[float], output_root: Path, verbose: bool = True) -> pd.DataFrame:
    rows = []
    dt = _NOMINAL_ENV_CONFIG.dt
    t_start = time.time()
    total = len(variances) * len(NOISE_POLICY_INSTANCES)
    i = 0
    for variance in variances:
        for instance in NOISE_POLICY_INSTANCES:
            i += 1
            env = build_env(instance.estimator, variance)
            policy = build_policy(instance)
            if verbose:
                print(f"[{i}/{total}] measurement_var={variance} {instance.policy_instance} "
                      f"(estimator={instance.estimator}) -- {len(seeds)} episodes...")
            for s in seeds:
                row = run_measurement_noise_episode(env, policy, seed=s, dt=dt)
                row["measurement_var"] = variance
                row["measurement_std"] = measurement_std(variance)
                row["paper_method"] = instance.paper_method
                row["policy_instance"] = instance.policy_instance
                row["estimator"] = instance.estimator
                row["training_seed"] = instance.training_seed if instance.training_seed is not None else float("nan")
                rows.append(row)
            env.close()
            if verbose:
                print(f"  done ({time.time() - t_start:.1f}s elapsed)")

    df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / RAW_EPISODE_RESULTS_FILENAME, index=False)
    return df


def run_integrity_checks(df: pd.DataFrame, seeds: list[int], variances: list[float]) -> dict:
    results = {}
    expected_rows = EXPECTED_N_POLICY_INSTANCES * len(variances) * len(seeds)
    results["expected_row_count"] = expected_rows
    results["actual_row_count"] = len(df)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    dup = df.duplicated(subset=["measurement_var", "policy_instance", "eval_seed"]).sum()
    results["duplicate_keys"] = int(dup)
    assert dup == 0, f"found {dup} duplicated (measurement_var, policy_instance, eval_seed) keys"

    seed_set_expected = set(seeds)
    combo_seed_sets = df.groupby(["measurement_var", "policy_instance"])["eval_seed"].apply(set)
    common_seeds_ok = all(s == seed_set_expected for s in combo_seed_sets)
    results["common_random_numbers_ok"] = bool(common_seeds_ok)
    assert common_seeds_ok, "not all (measurement_var, policy_instance) combinations share the identical evaluation-seed set"
    results["n_variance_method_combinations"] = int(len(combo_seed_sets))
    assert len(combo_seed_sets) == len(variances) * EXPECTED_N_POLICY_INSTANCES, "missing (measurement_var, policy_instance) combination"

    outcome_sum = df.groupby(["measurement_var", "policy_instance"])[
        ["success", "soldier_caught", "unsafe_intercept", "timeout"]
    ].sum().sum(axis=1)
    outcome_ok = bool((outcome_sum == len(seeds)).all())
    results["outcome_accounting_ok"] = outcome_ok
    assert outcome_ok, "outcome categories do not sum to n_episodes for every (measurement_var, policy_instance)"

    required = ["success", "soldier_caught", "unsafe_intercept", "timeout", "episode_length_steps", "total_reward"]
    nan_inf_bad = {c: int((~np.isfinite(df[c])).sum()) for c in required if (~np.isfinite(df[c])).sum()}
    results["nan_inf_in_required_fields"] = nan_inf_bad
    assert not nan_inf_bad, f"NaN/Inf found in required-always-defined fields: {nan_inf_bad}"

    return results


def run_detection_noise_independence_check(df: pd.DataFrame, variances: list[float]) -> dict:
    """
    Item 35: detection is defined by TRUE geometric range vs detection_radius
    and must not shift with measurement_var. Since all 4 scripted methods
    share the same pre-detection escort law (see acquisition ablation), their
    detection_step at a fixed seed should match ACROSS ALL measurement_var
    values for a given method (using the same measurement_var-independent
    escort trajectory).
    """
    tol = 1e-4
    results = {}
    ref_variance = variances[0]
    for method in SCRIPTED_METHODS:
        ref_df = df[(df["measurement_var"] == ref_variance) & (df["policy_instance"] == method)].sort_values("eval_seed").reset_index(drop=True)
        mismatches = 0
        checked = 0
        for variance in variances[1:]:
            other_df = df[(df["measurement_var"] == variance) & (df["policy_instance"] == method)].sort_values("eval_seed").reset_index(drop=True)
            assert (ref_df["eval_seed"].to_numpy() == other_df["eval_seed"].to_numpy()).all()
            checked += len(other_df)
            mismatch = ~np.isclose(
                ref_df["detection_step"].to_numpy(), other_df["detection_step"].to_numpy(), equal_nan=True, atol=tol
            )
            mismatches += int(mismatch.sum())
        results[method] = {"checked_episode_pairs": checked, "mismatches": mismatches}
    return results


def _success_vector(df: pd.DataFrame, variance: float, policy_instance: str) -> np.ndarray:
    sub = df[(df["measurement_var"] == variance) & (df["policy_instance"] == policy_instance)].sort_values("eval_seed")
    return sub["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, variance: float, paper_method: str) -> np.ndarray:
    sub = df[(df["measurement_var"] == variance) & (df["paper_method"] == paper_method)]
    rows = []
    for seed in (42, 43, 44):
        s = sub[sub["training_seed"] == seed].sort_values("eval_seed")
        rows.append(s["success"].to_numpy(dtype=np.float64))
    return np.stack(rows)


def _metric_vector(df: pd.DataFrame, variance: float, policy_instance: str, column: str) -> np.ndarray:
    sub = df[(df["measurement_var"] == variance) & (df["policy_instance"] == policy_instance)].sort_values("eval_seed")
    return sub[column].to_numpy(dtype=np.float64)


def _metric_matrix(df: pd.DataFrame, variance: float, paper_method: str, column: str) -> np.ndarray:
    sub = df[(df["measurement_var"] == variance) & (df["paper_method"] == paper_method)]
    rows = []
    for seed in (42, 43, 44):
        s = sub[sub["training_seed"] == seed].sort_values("eval_seed")
        rows.append(s[column].to_numpy(dtype=np.float64))
    return np.stack(rows)


def compute_noise_summary(df: pd.DataFrame, variances: list[float]) -> pd.DataFrame:
    rows = []
    for variance in variances:
        for method in PRIMARY_PAPER_METHODS:
            if method in SCRIPTED_METHODS:
                sub = df[(df["measurement_var"] == variance) & (df["policy_instance"] == method)]
                n = len(sub)
                successes = int(sub["success"].sum())
                ci_lo, ci_hi = wilson_interval(successes, n)
                rows.append({
                    "measurement_var": variance, "measurement_std": measurement_std(variance), "method": method,
                    "n_eval_episodes": n, "n_training_seeds": 0,
                    "success_rate": successes / n, "success_ci_low": ci_lo, "success_ci_high": ci_hi,
                    "training_seed_sd": float("nan"),
                    "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                    "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                    "timeout_rate": float(sub["timeout"].mean()),
                    "mean_episode_length_seconds": float(sub["episode_length_seconds"].mean()),
                    "mean_intercept_time_seconds": float(sub["intercept_time_seconds"].mean()),
                })
            else:
                success_matrix = _success_matrix(df, variance, method)
                observed, ci_lo, ci_hi = hierarchical_bootstrap_track(success_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
                per_seed_rates = np.nanmean(success_matrix, axis=1)

                def _ew(col):
                    m = _metric_matrix(df, variance, method, col)
                    return float(np.nanmean(np.nanmean(m, axis=1)))

                rows.append({
                    "measurement_var": variance, "measurement_std": measurement_std(variance), "method": method,
                    "n_eval_episodes": success_matrix.shape[1], "n_training_seeds": 3,
                    "success_rate": observed, "success_ci_low": ci_lo, "success_ci_high": ci_hi,
                    "training_seed_sd": float(np.std(per_seed_rates)),
                    "soldier_caught_rate": _ew("soldier_caught"),
                    "unsafe_intercept_rate": _ew("unsafe_intercept"),
                    "timeout_rate": _ew("timeout"),
                    "mean_episode_length_seconds": _ew("episode_length_seconds"),
                    "mean_intercept_time_seconds": _ew("intercept_time_seconds"),
                })
    return pd.DataFrame(rows)


def compute_ppo_training_seed_summary(df: pd.DataFrame, variances: list[float]) -> pd.DataFrame:
    rows = []
    for variance in variances:
        for method in PPO_METHODS:
            sub_method = df[(df["measurement_var"] == variance) & (df["paper_method"] == method)]
            for seed in (42, 43, 44):
                sub = sub_method[sub_method["training_seed"] == seed]
                n = len(sub)
                successes = int(sub["success"].sum())
                ci_lo, ci_hi = wilson_interval(successes, n)
                errs = sub["mean_position_estimation_error"].dropna()
                rows.append({
                    "measurement_var": variance, "paper_method": method, "training_seed": seed,
                    "n_eval_episodes": n, "success_count": successes, "success_rate": successes / n,
                    "success_ci_low": ci_lo, "success_ci_high": ci_hi,
                    "mean_position_estimation_error": float(errs.mean()) if len(errs) else float("nan"),
                    "position_estimation_rmse": (
                        float(np.sqrt(np.mean(np.square(errs)))) if len(errs) else float("nan")
                    ),
                    "mean_episode_length_seconds": float(sub["episode_length_seconds"].mean()),
                })
    return pd.DataFrame(rows)


def compute_pairwise_comparisons(df: pd.DataFrame, variances: list[float]) -> pd.DataFrame:
    rows = []
    for variance in variances:
        for name, method_a, method_b in ALL_COMPARISONS:
            a_is_ppo = method_a in PPO_METHODS
            b_is_ppo = method_b in PPO_METHODS
            row = {"measurement_var": variance, "comparison_name": name, "method_a": method_a, "method_b": method_b,
                   "diff_seed_42_pp": float("nan"), "diff_seed_43_pp": float("nan"), "diff_seed_44_pp": float("nan"),
                   "n10_a_wins": float("nan"), "n01_a_loses": float("nan"), "mcnemar_p_value": float("nan")}
            if not a_is_ppo and not b_is_ppo:
                vec_a = _success_vector(df, variance, method_a)
                vec_b = _success_vector(df, variance, method_b)
                observed, lo, hi = paired_bootstrap_ci(vec_a, vec_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
                n10 = int(np.sum((vec_a == 1) & (vec_b == 0)))
                n01 = int(np.sum((vec_a == 0) & (vec_b == 1)))
                p_value = mcnemar_exact(n01=n01, n10=n10)
                row.update({
                    "comparison_type": "fixed_paired",
                    "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                    "n10_a_wins": n10, "n01_a_loses": n01, "mcnemar_p_value": p_value,
                })
            elif a_is_ppo and b_is_ppo:
                a_matrix = _success_matrix(df, variance, method_a)
                b_matrix = _success_matrix(df, variance, method_b)
                observed, lo, hi, per_seed_diffs = hierarchical_paired_bootstrap_matched_seeds(
                    a_matrix, b_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
                )
                row.update({
                    "comparison_type": "hierarchical_paired_matched_seeds",
                    "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                    "diff_seed_42_pp": per_seed_diffs[0] * 100,
                    "diff_seed_43_pp": per_seed_diffs[1] * 100,
                    "diff_seed_44_pp": per_seed_diffs[2] * 100,
                })
            else:
                ppo_method, scripted_method = (method_a, method_b) if a_is_ppo else (method_b, method_a)
                ppo_matrix = _success_matrix(df, variance, ppo_method)
                scripted_vec = _success_vector(df, variance, scripted_method)
                observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(
                    ppo_matrix, scripted_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
                )
                if not a_is_ppo:
                    observed, lo, hi = -observed, -hi, -lo
                row.update({
                    "comparison_type": "hierarchical_paired_vs_scripted",
                    "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                })
            rows.append(row)
    return pd.DataFrame(rows)


def compute_estimator_summary(df: pd.DataFrame, variances: list[float]) -> pd.DataFrame:
    """
    CORRECTED aggregation (fix/measurement-noise-estimator-aggregation):
    reconstructs the true sample-weighted (pooled, timestep-correct) mean
    error and RMSE from each episode's own (n_i, m_i, r_i), rather than
    computing sqrt(mean(per_episode_mean_error^2)) -- see
    experiments/estimator_stats.py for the full derivation.
    """
    rows = []
    for variance in variances:
        theoretical_mean = theoretical_raw_measurement_mean_error(variance)
        theoretical_rmse = theoretical_raw_measurement_rmse(variance)
        for method in PRIMARY_PAPER_METHODS:
            if method in SCRIPTED_METHODS:
                sub = df[(df["measurement_var"] == variance) & (df["policy_instance"] == method)].dropna(
                    subset=["mean_position_estimation_error", "position_estimation_rmse"]
                )
                n = sub["n_position_estimation_samples"].to_numpy(dtype=np.float64)
                m = sub["mean_position_estimation_error"].to_numpy(dtype=np.float64)
                r = sub["position_estimation_rmse"].to_numpy(dtype=np.float64)
                rows.append({
                    "measurement_var": variance, "method": method,
                    "mean_error_sample_weighted": sample_weighted_mean_error(n, m),
                    "rmse_sample_weighted": sample_weighted_rmse(n, r),
                    "mean_error_episode_weighted": episode_weighted_mean_error(m),
                    "mean_episode_rmse": mean_episode_rmse(r),
                    "theoretical_raw_measurement_mean_error": theoretical_mean if method in ("greedy", "pn", "lead") else float("nan"),
                    "theoretical_raw_measurement_rmse": theoretical_rmse if method in ("greedy", "pn", "lead") else float("nan"),
                })
            else:
                sub_method = df[(df["measurement_var"] == variance) & (df["paper_method"] == method)]
                seed_rmse, seed_mean = [], []
                for seed in (42, 43, 44):
                    seed_sub = sub_method[sub_method["training_seed"] == seed].dropna(
                        subset=["mean_position_estimation_error", "position_estimation_rmse"]
                    )
                    n = seed_sub["n_position_estimation_samples"].to_numpy(dtype=np.float64)
                    m = seed_sub["mean_position_estimation_error"].to_numpy(dtype=np.float64)
                    r = seed_sub["position_estimation_rmse"].to_numpy(dtype=np.float64)
                    seed_mean.append(sample_weighted_mean_error(n, m))
                    seed_rmse.append(sample_weighted_rmse(n, r))
                mean_error, _ = equal_weight_across_seeds(np.array(seed_mean))
                rmse, rmse_sd = equal_weight_across_seeds(np.array(seed_rmse))
                rows.append({
                    "measurement_var": variance, "method": method,
                    "mean_error_sample_weighted": mean_error,
                    "rmse_sample_weighted": rmse,
                    "train_seed_rmse_sd": rmse_sd,
                    "theoretical_raw_measurement_mean_error": theoretical_mean if method == "ppo" else float("nan"),
                    "theoretical_raw_measurement_rmse": theoretical_rmse if method == "ppo" else float("nan"),
                })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked measurement-noise robustness sweep")
    parser.add_argument("--seed-start", type=int, default=MEASUREMENT_SWEEP_SEED_OFFSET)
    parser.add_argument("--n-episodes", type=int, default=MEASUREMENT_SWEEP_EPISODES)
    parser.add_argument("--variances", type=float, nargs="+", default=list(MEASUREMENT_VAR_VALUES))
    parser.add_argument("--output-root", type=str, default=str(ROBUSTNESS_NOISE_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_episodes))
    variances = args.variances
    output_root = Path(args.output_root)
    is_final = args.seed_start >= MEASUREMENT_SWEEP_SEED_OFFSET

    print("=" * 70)
    print("MEASUREMENT-NOISE ROBUSTNESS SWEEP" if is_final else "DRY RUN (non-final seeds)")
    print("=" * 70)
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes/condition)")
    print(f"Measurement variances: {variances}")
    print(f"Policy instances: {len(NOISE_POLICY_INSTANCES)}")
    print(f"Output root: {output_root}")
    print("=" * 70)

    print("\nHashing frozen PPO models and verifying vs final nominal manifest...")
    model_hashes = compute_model_hashes_and_verify()
    for h in model_hashes:
        print(f"  {h['policy_instance']}: sha256={h['sha256'][:16]}... matches_final_nominal={h['matches_final_nominal_hash']}")

    manifest = build_manifest(seeds, variances, output_root, model_hashes)
    print(f"\nManifest written. robustness_test_opened={manifest['robustness_test_opened']}")

    print("\nRunning evaluation...")
    df = run_evaluation(seeds, variances, output_root)

    print("\nRunning integrity checks...")
    integrity = run_integrity_checks(df, seeds, variances)
    print(json.dumps(integrity, indent=2, default=str))

    print("\nRunning detection noise-independence check (item 35)...")
    detection_check = run_detection_noise_independence_check(df, variances)
    print(json.dumps(detection_check, indent=2, default=str))

    print("\nComputing summaries...")
    noise_summary = compute_noise_summary(df, variances)
    noise_summary.to_csv(output_root / NOISE_SUMMARY_FILENAME, index=False)

    ppo_seed_summary = compute_ppo_training_seed_summary(df, variances)
    ppo_seed_summary.to_csv(output_root / PPO_TRAINING_SEED_SUMMARY_FILENAME, index=False)

    pairwise = compute_pairwise_comparisons(df, variances)
    pairwise.to_csv(output_root / PAIRWISE_COMPARISONS_FILENAME, index=False)

    estimator_summary = compute_estimator_summary(df, variances)
    estimator_summary.to_csv(output_root / ESTIMATOR_SUMMARY_FILENAME, index=False)

    final_summary = {
        "label": "MEASUREMENT-NOISE ROBUSTNESS SWEEP" if is_final else "DRY RUN (non-final seeds, diagnostic only)",
        "integrity_checks": integrity,
        "detection_noise_independence_check": detection_check,
        "noise_summary": noise_summary.to_dict(orient="records"),
        "ppo_training_seed_summary": ppo_seed_summary.to_dict(orient="records"),
        "pairwise_comparisons": pairwise.to_dict(orient="records"),
        "estimator_summary": estimator_summary.to_dict(orient="records"),
    }
    (output_root / MEASUREMENT_NOISE_SUMMARY_FILENAME).write_text(json.dumps(final_summary, indent=2, default=str))

    print(f"\nAll outputs written to {output_root}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
