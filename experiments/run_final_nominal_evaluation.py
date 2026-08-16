"""
Locked final nominal-evaluation runner (publication-specific, not a legacy script).

Evaluates all 8 paper-level methods (12 concrete policy instances: 6 scripted
+ 3 PPO + 3 PPO-Kalman) on the SAME set of environment seeds (common random
numbers), using the frozen 400k PPO models selected in
experiments/final_experiment_protocol.py.

Usage:
    # Full locked final evaluation (seeds 20000..24999, 5000 episodes/instance):
    python experiments/run_final_nominal_evaluation.py

    # Small NON-FINAL dry run (diagnostic seeds < 20000 only):
    python experiments/run_final_nominal_evaluation.py --seed-start 13000 --n-episodes 5 \
        --output-root results/_dry_run_final_nominal

Outputs (under --output-root, default results/final_nominal/, gitignored):
    experiment_manifest.json, model_hashes.json, raw_episode_results.csv,
    method_summary.csv, ppo_training_seed_summary.csv, pairwise_comparisons.csv,
    estimator_summary.csv, final_nominal_summary.json
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
from uav_defend.policies.registry import get_policy

from experiments.method_specs import METHOD_SPECS
from experiments.final_experiment_protocol import (
    FINAL_NOMINAL_SEED_OFFSET,
    FINAL_NOMINAL_EPISODES,
    FINAL_NOMINAL_SEEDS,
    PPO_TRAINING_SEEDS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    POLICY_INSTANCES,
    PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    PRIMARY_COMPARISONS,
    SECONDARY_COMPARISONS,
    ALL_COMPARISONS,
    FIXED_PAIRED_COMPARISONS,
    HIERARCHICAL_PAIRED_COMPARISONS,
    EXPECTED_N_POLICY_INSTANCES,
    FINAL_NOMINAL_OUTPUT_ROOT,
    EXPERIMENT_MANIFEST_FILENAME,
    MODEL_HASHES_FILENAME,
    RAW_EPISODE_RESULTS_FILENAME,
    METHOD_SUMMARY_FILENAME,
    PPO_TRAINING_SEED_SUMMARY_FILENAME,
    PAIRWISE_COMPARISONS_FILENAME,
    ESTIMATOR_SUMMARY_FILENAME,
    FINAL_NOMINAL_SUMMARY_FILENAME,
)
from experiments.final_eval_utils import run_final_episode, sha256_of_file
from experiments.final_stats import (
    wilson_interval,
    mcnemar_exact,
    paired_bootstrap_ci,
    hierarchical_bootstrap_track,
    hierarchical_paired_bootstrap_vs_scripted,
    hierarchical_paired_bootstrap_matched_seeds,
)

_NOMINAL_ENV_CONFIG = EnvConfig()  # defaults == training/validation/diagnostic config


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


def build_env(estimator: str) -> SoldierEnv:
    return SoldierEnv(config=EnvConfig(use_kalman_tracking=(estimator == "kalman")))


def build_policy(instance):
    if instance.model_path is not None:
        return get_policy(instance.policy_name, model_path=str(instance.model_path), deterministic=True)
    return get_policy(instance.policy_name)


def compute_model_hashes() -> list[dict]:
    """Hash the 6 frozen 400k best_model.zip files (item 3). Must be called
    before evaluating any final-test seed."""
    rows = []
    for instance in POLICY_INSTANCES:
        if instance.model_path is None:
            continue
        model_path = instance.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Frozen model not found: {model_path}")
        manifest_path = model_path.parent / "training_manifest.json"
        training_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        rows.append({
            "paper_method": instance.paper_method,
            "policy_instance": instance.policy_instance,
            "track": "kalman" if instance.estimator == "kalman" else "direct",
            "training_seed": instance.training_seed,
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_of_file(model_path),
            "training_manifest_source_commit": training_manifest.get("git_commit"),
            "best_validation_success_rate": training_manifest.get("best_validation_success_rate"),
            "best_checkpoint_timestep": training_manifest.get("best_checkpoint_timestep"),
        })
    return rows


def build_manifest(seeds: list[int], output_root: Path, model_hashes: list[dict]) -> dict:
    import gymnasium
    import stable_baselines3
    import torch

    opened = len(seeds) > 0 and min(seeds) >= FINAL_NOMINAL_SEED_OFFSET
    obs_space_shape = build_env("measurement").observation_space.shape
    act_space_shape = build_env("measurement").action_space.shape

    manifest = {
        "source_git_commit": _get_git_commit(),
        "source_git_branch": _get_git_branch(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gymnasium.__version__,
        "env_config": dataclasses.asdict(_NOMINAL_ENV_CONFIG),
        "ppo_model_hashes": model_hashes,
        "ppo_training_seeds": list(PPO_TRAINING_SEEDS),
        "evaluation_seed_start": min(seeds) if seeds else None,
        "evaluation_seed_end": max(seeds) if seeds else None,
        "n_episodes_per_instance": len(seeds),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": CONFIDENCE_LEVEL,
        "primary_comparisons": list(PRIMARY_COMPARISONS),
        "secondary_comparisons": list(SECONDARY_COMPARISONS),
        "method_registry_snapshot": [dataclasses.asdict(spec) for spec in METHOD_SPECS],
        "observation_shape": list(obs_space_shape),
        "action_shape": list(act_space_shape),
        "final_test_opened": opened,
        "final_test_first_seed": FINAL_NOMINAL_SEED_OFFSET if opened else None,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / EXPERIMENT_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, default=str))
    (output_root / MODEL_HASHES_FILENAME).write_text(json.dumps(model_hashes, indent=2, default=str))
    return manifest


def run_evaluation(seeds: list[int], output_root: Path, verbose: bool = True) -> pd.DataFrame:
    rows = []
    dt = _NOMINAL_ENV_CONFIG.dt
    t_start = time.time()
    for i, instance in enumerate(POLICY_INSTANCES, start=1):
        env = build_env(instance.estimator)
        policy = build_policy(instance)
        if verbose:
            print(f"[{i}/{len(POLICY_INSTANCES)}] {instance.policy_instance} "
                  f"(paper_method={instance.paper_method}, estimator={instance.estimator}) "
                  f"-- {len(seeds)} episodes...")
        for s in seeds:
            row = run_final_episode(env, policy, seed=s, dt=dt)
            row["paper_method"] = instance.paper_method
            row["policy_instance"] = instance.policy_instance
            row["estimator"] = instance.estimator
            row["training_seed"] = instance.training_seed if instance.training_seed is not None else float("nan")
            rows.append(row)
        env.close()
        if verbose:
            elapsed = time.time() - t_start
            print(f"  done ({elapsed:.1f}s elapsed)")

    df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / RAW_EPISODE_RESULTS_FILENAME, index=False)
    return df


def run_integrity_checks(df: pd.DataFrame, seeds: list[int]) -> dict:
    """Items 27/38/39: row count, duplicates, common-random-numbers, outcome accounting."""
    results = {}

    expected_rows = EXPECTED_N_POLICY_INSTANCES * len(seeds)
    results["expected_row_count"] = expected_rows
    results["actual_row_count"] = len(df)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    dup = df.duplicated(subset=["policy_instance", "eval_seed"]).sum()
    results["duplicate_policy_instance_seed_pairs"] = int(dup)
    assert dup == 0, f"found {dup} duplicated (policy_instance, eval_seed) pairs"

    seed_set_expected = set(seeds)
    seed_sets_by_instance = df.groupby("policy_instance")["eval_seed"].apply(set)
    common_seeds_ok = all(s == seed_set_expected for s in seed_sets_by_instance)
    results["common_random_numbers_ok"] = bool(common_seeds_ok)
    assert common_seeds_ok, "not all 12 policy instances share the identical evaluation-seed set"

    outcome_sum = df.groupby("policy_instance")[["success", "soldier_caught", "unsafe_intercept", "timeout"]].sum().sum(axis=1)
    outcome_accounting_ok = bool((outcome_sum == len(seeds)).all())
    results["outcome_accounting_ok"] = outcome_accounting_ok
    assert outcome_accounting_ok, "outcome categories do not sum to n_episodes for every policy instance"

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_nan_finite_required = ["success", "soldier_caught", "unsafe_intercept", "timeout",
                                "episode_length_steps", "total_reward"]
    nan_inf_bad = {}
    for col in non_nan_finite_required:
        bad = (~np.isfinite(df[col])).sum()
        if bad:
            nan_inf_bad[col] = int(bad)
    results["nan_inf_in_required_fields"] = nan_inf_bad
    assert not nan_inf_bad, f"NaN/Inf found in required-always-defined fields: {nan_inf_bad}"

    return results


def _success_vector(df: pd.DataFrame, paper_method: str) -> np.ndarray:
    sub = df[df["paper_method"] == paper_method].sort_values("eval_seed")
    return sub["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, paper_method: str) -> tuple[np.ndarray, list[int]]:
    sub = df[df["paper_method"] == paper_method]
    seed_labels = sorted(int(s) for s in sub["training_seed"].unique())
    rows = []
    for s in seed_labels:
        rows.append(sub[sub["training_seed"] == s].sort_values("eval_seed")["success"].to_numpy(dtype=np.float64))
    return np.stack(rows), seed_labels


def _scripted_method_row(df_method: pd.DataFrame) -> dict:
    n = len(df_method)
    successes = int(df_method["success"].sum())
    ci_lo, ci_hi = wilson_interval(successes, n)
    return {
        "n_eval_episodes": n,
        "n_training_seeds": 0,
        "success_rate": successes / n,
        "success_ci_low": ci_lo,
        "success_ci_high": ci_hi,
        "training_seed_sd": float("nan"),
        "soldier_caught_rate": float(df_method["soldier_caught"].mean()),
        "unsafe_intercept_rate": float(df_method["unsafe_intercept"].mean()),
        "timeout_rate": float(df_method["timeout"].mean()),
        "mean_episode_length_seconds": float(df_method["episode_length_seconds"].mean()),
        "mean_detection_time_seconds": float(df_method["detection_time_seconds"].mean()),
        "mean_intercept_time_seconds": float(df_method["intercept_time_seconds"].mean()),
        "mean_min_enemy_soldier_distance": float(df_method["min_enemy_soldier_dist"].mean()),
    }


def _ppo_method_row(df_method: pd.DataFrame) -> dict:
    success_matrix, seed_labels = _success_matrix(df_method, df_method["paper_method"].iloc[0])
    per_seed_rates = success_matrix.mean(axis=1)
    observed, ci_lo, ci_hi = hierarchical_bootstrap_track(success_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)

    seed_dfs = [df_method[df_method["training_seed"] == s].sort_values("eval_seed") for s in seed_labels]

    def _equal_weight_mean(col: str) -> float:
        return float(np.mean([d[col].mean() for d in seed_dfs]))

    return {
        "n_eval_episodes": FINAL_NOMINAL_EPISODES,
        "n_training_seeds": 3,
        "success_rate": observed,
        "success_ci_low": ci_lo,
        "success_ci_high": ci_hi,
        "training_seed_sd": float(np.std(per_seed_rates)),
        "soldier_caught_rate": _equal_weight_mean("soldier_caught"),
        "unsafe_intercept_rate": _equal_weight_mean("unsafe_intercept"),
        "timeout_rate": _equal_weight_mean("timeout"),
        "mean_episode_length_seconds": _equal_weight_mean("episode_length_seconds"),
        "mean_detection_time_seconds": _equal_weight_mean("detection_time_seconds"),
        "mean_intercept_time_seconds": _equal_weight_mean("intercept_time_seconds"),
        "mean_min_enemy_soldier_distance": _equal_weight_mean("min_enemy_soldier_dist"),
    }


def compute_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in PAPER_METHODS:
        spec = next(s for s in METHOD_SPECS if s.key == method)
        df_method = df[df["paper_method"] == method]
        if method in SCRIPTED_METHODS:
            stats_row = _scripted_method_row(df_method)
        else:
            stats_row = _ppo_method_row(df_method)
        stats_row = {"method": method, "estimator": spec.estimator, **stats_row}
        rows.append(stats_row)
    return pd.DataFrame(rows)


def compute_ppo_training_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in PPO_METHODS:
        df_method = df[df["paper_method"] == method]
        for seed in PPO_TRAINING_SEEDS:
            sub = df_method[df_method["training_seed"] == seed]
            n = len(sub)
            successes = int(sub["success"].sum())
            ci_lo, ci_hi = wilson_interval(successes, n)
            rows.append({
                "paper_method": method,
                "training_seed": seed,
                "n_eval_episodes": n,
                "success_count": successes,
                "success_rate": successes / n,
                "success_ci_low": ci_lo,
                "success_ci_high": ci_hi,
            })
    return pd.DataFrame(rows)


def compute_pairwise_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, method_a, method_b in ALL_COMPARISONS:
        a_is_ppo = method_a in PPO_METHODS
        b_is_ppo = method_b in PPO_METHODS
        row = {"comparison_name": name, "method_a": method_a, "method_b": method_b,
               "diff_seed_42_pp": float("nan"), "diff_seed_43_pp": float("nan"), "diff_seed_44_pp": float("nan")}

        if not a_is_ppo and not b_is_ppo:
            # Both fixed/scripted: matched-seed paired bootstrap + exact McNemar.
            vec_a = _success_vector(df, method_a)
            vec_b = _success_vector(df, method_b)
            observed, lo, hi = paired_bootstrap_ci(vec_a, vec_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
            a_win_b_lose = int(np.sum((vec_a == 1) & (vec_b == 0)))  # n10
            a_lose_b_win = int(np.sum((vec_a == 0) & (vec_b == 1)))  # n01
            p_value = mcnemar_exact(n01=a_lose_b_win, n10=a_win_b_lose)
            row.update({
                "comparison_type": "fixed_paired",
                "observed_diff_pp": observed * 100,
                "ci_low_pp": lo * 100,
                "ci_high_pp": hi * 100,
                "n10_a_wins": a_win_b_lose,
                "n01_a_loses": a_lose_b_win,
                "mcnemar_p_value": p_value,
            })
        elif a_is_ppo and b_is_ppo:
            # G: PPO-Kalman vs PPO, matched training-seed labels.
            a_matrix, a_labels = _success_matrix(df, method_a)
            b_matrix, b_labels = _success_matrix(df, method_b)
            assert a_labels == b_labels == list(PPO_TRAINING_SEEDS)
            observed, lo, hi, per_seed_diffs = hierarchical_paired_bootstrap_matched_seeds(
                a_matrix, b_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
            )
            row.update({
                "comparison_type": "hierarchical_paired_matched_seeds",
                "observed_diff_pp": observed * 100,
                "ci_low_pp": lo * 100,
                "ci_high_pp": hi * 100,
                "n10_a_wins": float("nan"),
                "n01_a_loses": float("nan"),
                "mcnemar_p_value": float("nan"),
                "diff_seed_42_pp": per_seed_diffs[0] * 100,
                "diff_seed_43_pp": per_seed_diffs[1] * 100,
                "diff_seed_44_pp": per_seed_diffs[2] * 100,
            })
        else:
            # PPO/PPO-Kalman vs a fixed scripted comparator.
            ppo_method, scripted_method = (method_a, method_b) if a_is_ppo else (method_b, method_a)
            ppo_matrix, _ = _success_matrix(df, ppo_method)
            scripted_vec = _success_vector(df, scripted_method)
            observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(
                ppo_matrix, scripted_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
            )
            if a_is_ppo:
                obs_signed, lo_signed, hi_signed = observed, lo, hi
            else:
                obs_signed, lo_signed, hi_signed = -observed, -hi, -lo
            row.update({
                "comparison_type": "hierarchical_paired_vs_scripted",
                "observed_diff_pp": obs_signed * 100,
                "ci_low_pp": lo_signed * 100,
                "ci_high_pp": hi_signed * 100,
                "n10_a_wins": float("nan"),
                "n01_a_loses": float("nan"),
                "mcnemar_p_value": float("nan"),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def compute_estimator_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in PAPER_METHODS:
        df_method = df[df["paper_method"] == method]
        if method in SCRIPTED_METHODS:
            row = {
                "method": method,
                "defender_path_length": float(df_method["defender_path_length"].mean()),
                "mean_defender_speed": float(df_method["mean_defender_speed"].mean()),
                "mean_defender_accel": float(df_method["mean_defender_accel"].mean()),
                "mean_defender_turn_rate": float(df_method["mean_defender_turn_rate"].mean()),
                "accel_saturation_fraction": float(df_method["fraction_defender_accel_saturated"].mean()),
                "turn_saturation_fraction": float(df_method["fraction_defender_turn_saturated"].mean()),
                "climb_saturation_fraction": float(df_method["fraction_defender_climb_saturated"].mean()),
                "evasion_episode_fraction": float(df_method["evasion_activated_episode"].mean()),
                "mean_position_estimation_error": float(df_method["mean_position_estimation_error"].mean()),
                "position_estimation_rmse": float(
                    np.sqrt(np.mean(np.square(df_method["mean_position_estimation_error"].dropna())))
                ) if df_method["mean_position_estimation_error"].notna().any() else float("nan"),
            }
        else:
            seed_labels = sorted(int(s) for s in df_method["training_seed"].unique())
            seed_dfs = [df_method[df_method["training_seed"] == s] for s in seed_labels]

            def _ew(col):
                return float(np.mean([d[col].mean() for d in seed_dfs]))

            row = {
                "method": method,
                "defender_path_length": _ew("defender_path_length"),
                "mean_defender_speed": _ew("mean_defender_speed"),
                "mean_defender_accel": _ew("mean_defender_accel"),
                "mean_defender_turn_rate": _ew("mean_defender_turn_rate"),
                "accel_saturation_fraction": _ew("fraction_defender_accel_saturated"),
                "turn_saturation_fraction": _ew("fraction_defender_turn_saturated"),
                "climb_saturation_fraction": _ew("fraction_defender_climb_saturated"),
                "evasion_episode_fraction": _ew("evasion_activated_episode"),
                "mean_position_estimation_error": _ew("mean_position_estimation_error"),
                "position_estimation_rmse": float(np.mean([
                    np.sqrt(np.mean(np.square(d["mean_position_estimation_error"].dropna())))
                    if d["mean_position_estimation_error"].notna().any() else np.nan
                    for d in seed_dfs
                ])),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked final nominal evaluation (8 paper methods, common random numbers)")
    parser.add_argument("--seed-start", type=int, default=FINAL_NOMINAL_SEED_OFFSET)
    parser.add_argument("--n-episodes", type=int, default=FINAL_NOMINAL_EPISODES)
    parser.add_argument("--output-root", type=str, default=str(FINAL_NOMINAL_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_episodes))
    output_root = Path(args.output_root)

    is_final = args.seed_start >= FINAL_NOMINAL_SEED_OFFSET
    print("=" * 70)
    print("FINAL NOMINAL EVALUATION" if is_final else "DRY RUN (non-final seeds)")
    print("=" * 70)
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes/instance)")
    print(f"Policy instances: {len(POLICY_INSTANCES)}")
    print(f"Output root: {output_root}")
    print("=" * 70)

    print("\nHashing frozen PPO models (before opening test set)...")
    model_hashes = compute_model_hashes()
    for h in model_hashes:
        print(f"  {h['policy_instance']}: sha256={h['sha256'][:16]}... "
              f"best_success={h['best_validation_success_rate']} @ t={h['best_checkpoint_timestep']}")

    manifest = build_manifest(seeds, output_root, model_hashes)
    print(f"\nManifest written. final_test_opened={manifest['final_test_opened']}")

    print("\nRunning evaluation...")
    df = run_evaluation(seeds, output_root)

    print("\nRunning integrity checks...")
    integrity = run_integrity_checks(df, seeds)
    print(json.dumps(integrity, indent=2, default=str))

    print("\nComputing summaries...")
    method_summary = compute_method_summary(df)
    method_summary.to_csv(output_root / METHOD_SUMMARY_FILENAME, index=False)

    ppo_seed_summary = compute_ppo_training_seed_summary(df)
    ppo_seed_summary.to_csv(output_root / PPO_TRAINING_SEED_SUMMARY_FILENAME, index=False)

    pairwise = compute_pairwise_comparisons(df)
    pairwise.to_csv(output_root / PAIRWISE_COMPARISONS_FILENAME, index=False)

    estimator_summary = compute_estimator_summary(df)
    estimator_summary.to_csv(output_root / ESTIMATOR_SUMMARY_FILENAME, index=False)

    final_summary = {
        "label": "FINAL NOMINAL EVALUATION" if is_final else "DRY RUN (non-final seeds, diagnostic only)",
        "integrity_checks": integrity,
        "method_summary": method_summary.to_dict(orient="records"),
        "ppo_training_seed_summary": ppo_seed_summary.to_dict(orient="records"),
        "pairwise_comparisons": pairwise.to_dict(orient="records"),
        "estimator_summary": estimator_summary.to_dict(orient="records"),
    }
    (output_root / FINAL_NOMINAL_SUMMARY_FILENAME).write_text(json.dumps(final_summary, indent=2, default=str))

    print(f"\nAll outputs written to {output_root}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
